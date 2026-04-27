# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""alerting.py — 掉线告警（SMTP）

对外暴露：
    AlertManager(cfg).notify_disconnect(reason)
    AlertManager(cfg).notify_recover()

设计要点：
    1. SMTP 凭据全部从环境变量读取（与 scripts/test_smtp.py 一致），yaml 不存放密码。
    2. 内置冷却窗口（cooldown），防止重复轰炸邮箱。
    3. 状态机：仅在 OK→DOWN / DOWN→OK 切换时发邮件。
    4. 阻塞 IO 通过 asyncio.to_thread 放线程池，不卡事件循环。
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
import secrets
import smtplib
import ssl
import time
from datetime import datetime
from email.message import EmailMessage
from email.utils import make_msgid
from pathlib import Path

logger = logging.getLogger("AICQ.alerting")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


class AlertManager:
    """掉线告警发送器。

    cfg 字段（来自 config.yaml 的 alerting 段）：
        enabled: bool          总开关
        cooldown: int          两次同类告警最小间隔（秒）
        subject_prefix: str    主题前缀
    """

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.cooldown: int = int(cfg.get("cooldown", 600))
        self.subject_prefix: str = cfg.get("subject_prefix", "[AIcarus 告警]")
        self._last_disconnect_at: float = 0.0
        self._lock = asyncio.Lock()
        # 状态机：True=已知离线，False=已知在线，None=未知
        self._is_down: bool | None = None
        # ── 远程指令（Phase 3）所需的 token + Message-ID 注册表 ──
        # token → (msg_id, expires_at_wall_time)；一次性，使用即焚
        self._pending_tokens: dict[str, tuple[str, float]] = {}
        # 最近发出的告警 Message-ID 集合（24h 滑窗），用于 In-Reply-To 校验
        self._recent_msgids: dict[str, float] = {}
        # 邮件指令模块在此读取参数（避免循环 import）
        ec_cfg = (cfg.get("email_control") or {})
        self.email_control_cfg: dict = dict(ec_cfg)
        # 默认 10 分钟：正常握手场景下 token 是发 REQUEST 后即时下发，10min 足够。
        # 安全靠“发件人白名单 + 严格 In-Reply-To + 单次使用”，不靠超时。
        self._token_ttl: int = max(
            60, min(7 * 24 * 3600, int(ec_cfg.get("token_ttl_seconds", 600)))
        )
        self._msgid_ttl: int = max(self._token_ttl, 24 * 3600)

    # ── token / msgid 维护（Phase 3 远程指令用）─────────────
    def _gc_expired(self) -> None:
        now = time.time()
        self._pending_tokens = {
            k: v for k, v in self._pending_tokens.items() if v[1] > now
        }
        self._recent_msgids = {
            k: v for k, v in self._recent_msgids.items() if v > now
        }

    def _new_token_and_msgid(self) -> tuple[str, str]:
        """生成一次性 token + Message-ID，并登记到注册表。"""
        self._gc_expired()
        token = secrets.token_hex(16)
        msg_id = make_msgid(domain="aicarus.local")
        self._pending_tokens[token] = (msg_id, time.time() + self._token_ttl)
        self._recent_msgids[msg_id] = time.time() + self._msgid_ttl
        logger.info(
            "已签发远程指令 token=%s msgid=%s pending_count=%d",
            token, msg_id, len(self._pending_tokens),
        )
        return token, msg_id

    def consume_token(self, token: str | None, in_reply_to: str | None) -> str:
        """远程指令鉴权。返回值：
          - "ok"            : token 有效，已销毁
          - "missing"       : token 不在注册表（没发过 / 已过期 / bot 重启）
          - "used"          : token 曾存在但已被销毁（处理中不会同时走这里，预留）
          - "irt_mismatch"  : token 有效但 In-Reply-To 不匹配（软提示，仍销毁 + 放行）
        策略：token 本身 = 16 字节随机 + 一次性 + 10min TTL + 发件人白名单，
        已足够安全。In-Reply-To 只作为诊断信息，不作为拒绝门。
        """
        self._gc_expired()
        if not token:
            return "missing"
        token = token.strip().lower()
        rec = self._pending_tokens.get(token)
        if rec is None:
            return "missing"
        bound_msgid, _exp = rec
        norm_irt = (in_reply_to or "").strip().strip("<>")
        norm_bound = bound_msgid.strip().strip("<>")
        # 一次性：无论如何都销毁 token
        self._pending_tokens.pop(token, None)
        if norm_irt and norm_irt != norm_bound:
            return "irt_mismatch"
        return "ok"

    def list_recent_msgids(self) -> list[str]:
        self._gc_expired()
        return list(self._recent_msgids.keys())

    def _command_footer(self, token: str) -> str:
        ec = self.email_control_cfg
        if not ec.get("enabled"):
            return ""
        cmds = ec.get("allowed_commands") or ["RESTART", "STATUS"]
        cmd_list = " / ".join(str(c).upper() for c in cmds)
        ttl_sec = self._token_ttl
        if ttl_sec >= 3600:
            ttl_human = f"{ttl_sec // 3600} 小时"
        else:
            ttl_human = f"{max(1, ttl_sec // 60)} 分钟"
        return (
            "\n\n──────── 远程指令（回复本邮件以触发）────────\n"
            f"在【正文首行】或【主题】写以下任一指令：{cmd_list}\n"
            f"必须包含以下一次性令牌（{ttl_human}内有效，用一次即失效）：\n"
            f"TOKEN: {token}\n"
            "────────────────────────────────────────\n"
        )

    async def notify_disconnect(self, reason: str) -> None:
        """通知掉线。仅在状态切换或冷却结束后真正发邮件。"""
        if not self.enabled:
            return
        loop = asyncio.get_event_loop()
        async with self._lock:
            now = loop.time()
            # 已经处于 down 状态，且未超过冷却 → 跳过
            if self._is_down is True and (now - self._last_disconnect_at) < self.cooldown:
                logger.debug("掉线告警冷却中，跳过 (%s)", reason)
                return
            self._is_down = True
            self._last_disconnect_at = now

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"NapCat 掉线: {reason}"
        token, msg_id = self._new_token_and_msgid()
        body = (
            f"时间: {ts}\n"
            f"原因: {reason}\n\n"
            f"请检查 NapCat 进程 / QQ 账号风控状态。\n"
            f"恢复后会再发送一封 [恢复] 邮件。"
        ) + self._command_footer(token)
        await asyncio.to_thread(
            self._send_sync, subject, body, None, msg_id,
        )

    async def notify_recover(self) -> None:
        """通知恢复连接。仅在之前确认过 down 时才发。"""
        if not self.enabled:
            return
        async with self._lock:
            if self._is_down is not True:
                # 从未发过掉线告警，无需发恢复通知
                return
            self._is_down = False

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "NapCat 恢复连接"
        body = f"时间: {ts}\nNapCat 已重新上线，监控解除。"
        await asyncio.to_thread(self._send_sync, subject, body)

    async def notify_qrcode(
        self,
        reason: str,
        qr_path: "Path | str",
        recovery_hint: str = "等待窗口内",
    ) -> None:
        """自动重启后未恢复时，把 NapCat 生成的登录二维码发到收件人邮箱。

        与 notify_disconnect 共享冷却窗口：但二维码强制突破冷却（用户需要立即扫码）。
        """
        if not self.enabled:
            return
        path = Path(qr_path)
        if not path.is_file():
            logger.warning("二维码文件不存在，跳过邮件: %s", path)
            return
        try:
            data = await asyncio.to_thread(path.read_bytes)
        except OSError as e:
            logger.warning("读取二维码文件失败 %s: %s", path, e)
            return

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "NapCat 需要扫码登录"
        body = (
            f"时间: {ts}\n"
            f"原因: {reason}\n\n"
            f"已自动执行 NapCat 重启脚本，但 {recovery_hint}未恢复连接，"
            f"通常意味着账号 token 失效，需要扫描附件中的二维码完成登录。\n"
            f"二维码文件路径: {path}\n"
            f"修改时间: {datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        maintype, _, subtype = mime.partition("/")
        attachment = (path.name, data, maintype, subtype or "png")
        token, msg_id = self._new_token_and_msgid()
        body += self._command_footer(token)
        await asyncio.to_thread(
            self._send_sync, subject, body, [attachment], msg_id,
        )

    async def notify_disconnect_followup(self, message: str) -> None:
        """在已发出掉线告警之后追加一条说明（如自动重启失败但未拿到二维码）。

        受 notify_disconnect 同一冷却窗口约束，避免轰炸。
        """
        if not self.enabled:
            return
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if (now - self._last_disconnect_at) < self.cooldown:
                logger.debug("followup 告警冷却中，跳过")
                return
            self._last_disconnect_at = now
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "NapCat 自动重启后续"
        token, msg_id = self._new_token_and_msgid()
        body = f"时间: {ts}\n{message}" + self._command_footer(token)
        await asyncio.to_thread(
            self._send_sync, subject, body, None, msg_id,
        )

    async def notify_command_result(
        self,
        command: str,
        message: str,
        in_reply_to: str | None = None,
    ) -> None:
        """远程指令执行结果回执。无 cooldown，每条指令都回复。

        - in_reply_to 非空：作为对原指令邮件的回信（不带新 token）
        - in_reply_to 为空（如 REQUEST 握手回执）：作为新会话发送，
          自带新 token + Message-ID，供用户继续回复以触发实际指令。
        """
        if not self.enabled:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = f"指令执行: {command}"
        body = f"时间: {ts}\n指令: {command}\n结果: {message}"
        new_msgid: str | None = None
        if in_reply_to is None:
            token, new_msgid = self._new_token_and_msgid()
            body += self._command_footer(token)
        await asyncio.to_thread(
            self._send_sync, subject, body, None, new_msgid, in_reply_to,
        )

    # ── 内部 ────────────────────────────────────────────────

    def _send_sync(
        self,
        subject: str,
        body: str,
        attachments: list[tuple[str, bytes, str, str]] | None = None,
        message_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> None:
        host = _env("AICQ_SMTP_HOST")
        try:
            port = int(_env("AICQ_SMTP_PORT", "465") or "465")
        except ValueError:
            logger.error("AICQ_SMTP_PORT 不是有效数字，告警邮件未发送")
            return
        use_ssl = _bool_env("AICQ_SMTP_USE_SSL", port == 465)
        user = _env("AICQ_SMTP_USER")
        password = _env("AICQ_SMTP_PASSWORD")
        sender_raw = _env("AICQ_SMTP_SENDER") or user
        # 容错：用户可能只填了中文显示名（如 “吹雪”）不带 @。
        # 不处理会导致 SMTPUTF8 报错（服务器不支持 UTF-8 地址），
        # 这里包装为 RFC 标准的 "<显示名>" <user@host> 格式。
        if "@" not in sender_raw and user:
            from email.utils import formataddr
            sender = formataddr((sender_raw, user))
        else:
            sender = sender_raw
        recipients_raw = _env("AICQ_SMTP_RECIPIENTS")
        recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

        if not (host and user and password and recipients):
            logger.warning(
                "SMTP 配置不完整（host/user/password/recipients 至少一项缺失），"
                "告警未发送 subject=%s", subject,
            )
            return

        msg = EmailMessage()
        msg["Subject"] = f"{self.subject_prefix} {subject}"
        msg["From"] = sender
        msg["To"] = ", ".join(recipients)
        if message_id:
            msg["Message-ID"] = message_id
        if in_reply_to:
            irt = in_reply_to.strip()
            if not irt.startswith("<"):
                irt = f"<{irt}>"
            msg["In-Reply-To"] = irt
            msg["References"] = irt
        msg.set_content(body)
        for filename, data, maintype, subtype in (attachments or []):
            try:
                msg.add_attachment(
                    data, maintype=maintype, subtype=subtype, filename=filename,
                )
            except (TypeError, ValueError):
                logger.warning("附件添加失败，已跳过: %s", filename)

        try:
            if use_ssl:
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL(host, port, context=ctx, timeout=15) as s:
                    s.login(user, password)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(host, port, timeout=15) as s:
                    s.ehlo()
                    if port != 25:
                        s.starttls(context=ssl.create_default_context())
                        s.ehlo()
                    s.login(user, password)
                    s.send_message(msg)
            logger.info("已发送告警邮件: %s", subject)
        except (smtplib.SMTPException, OSError) as e:
            logger.exception("发送告警邮件失败: %s", e)
