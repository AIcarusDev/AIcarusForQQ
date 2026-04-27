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
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
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
        body = (
            f"时间: {ts}\n"
            f"原因: {reason}\n\n"
            f"请检查 NapCat 进程 / QQ 账号风控状态。\n"
            f"恢复后会再发送一封 [恢复] 邮件。"
        )
        await asyncio.to_thread(self._send_sync, subject, body)

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
        await asyncio.to_thread(
            self._send_sync, subject, body, [attachment]
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
        body = f"时间: {ts}\n{message}"
        await asyncio.to_thread(self._send_sync, subject, body)

    # ── 内部 ────────────────────────────────────────────────

    def _send_sync(
        self,
        subject: str,
        body: str,
        attachments: list[tuple[str, bytes, str, str]] | None = None,
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
        sender = _env("AICQ_SMTP_SENDER") or user
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
