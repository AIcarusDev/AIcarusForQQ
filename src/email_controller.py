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

"""email_controller.py — 邮件远程指令通道（Phase 3）

设计目标：
    - 通过 IMAP 周期性轮询收件箱，识别**针对告警邮件的回复**，触发预定义指令。
    - 安全模型（多重闸）：
        1) 发件人必须在 `AICQ_SMTP_RECIPIENTS` 白名单内
        2) `In-Reply-To` 必须命中近期发出的告警邮件 Message-ID（24h 内）
        3) 邮件正文/主题必须包含告警邮件附带的一次性 token，且未过期、未使用
        4) 指令必须在白名单（默认 RESTART/STATUS，可加 STOP/KILL_AICQ）
    - 任一闸门失败即沉默丢弃；通过则调度对应动作，并以 In-Reply-To 回执结果。

为何不用 IMAP IDLE：
    - imaplib 原生不支持 IDLE，自实现脆弱（30 分钟必断、需自己处理 server 推送）。
    - 用户场景：通知慢一点（≤30s）完全可接受。
    - 轮询模型简单可靠：每 N 秒登录 → SEARCH UNSEEN → fetch → 标记 Seen → 关闭。
"""

from __future__ import annotations

import asyncio
import contextlib
import email
import imaplib
import logging
import os
import re
import ssl
from email.utils import parseaddr
from typing import Any

import app_state

logger = logging.getLogger("AICQ.email_controller")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# 解析正文中 "TOKEN: <hex>" 的正则；大小写不敏感；不用 \b 因为 hex 后面可能紧跟 HTML 残留
_TOKEN_RE = re.compile(r"TOKEN\s*[:：]\s*([0-9a-fA-F]{16,64})", re.IGNORECASE)
# 指令识别：单独成行的大写关键字；只识别白名单内的
_COMMAND_RE = re.compile(r"\b(REQUEST|RESTART|STOP|STATUS|KILL_AICQ)\b", re.IGNORECASE)


class _SkipMail(Exception):
    """信号异常：表示这封邮件应被跳过处理（不标记 Seen、不入 dedupe）。"""


class EmailController:
    """邮件指令控制器。

    cfg 字段（来自 alerting.email_control）:
        enabled: bool                 总开关
        allowed_commands: list[str]   允许执行的指令（默认 [RESTART, STATUS]）
        token_ttl_seconds: int        token 有效期（已由 AlertManager 持有）
        poll_interval: int            IMAP 轮询周期（秒），默认 30
        reuse_smtp_credentials: bool  IMAP USER/PASSWORD 复用 SMTP 同名字段
    """

    def __init__(
        self,
        alerting_cfg: dict | None,
        supervisor: Any = None,
        alert: Any = None,
    ) -> None:
        ec = ((alerting_cfg or {}).get("email_control") or {})
        self.enabled: bool = bool(ec.get("enabled", False))
        self.allowed_commands: set[str] = {
            str(c).upper().strip()
            for c in (ec.get("allowed_commands") or ["RESTART", "STATUS"])
            if str(c).strip()
        }
        self.poll_interval: int = max(10, int(ec.get("poll_interval", 30)))
        self.reuse_smtp: bool = bool(ec.get("reuse_smtp_credentials", True))

        self._supervisor = supervisor
        self._alert = alert
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        # 已处理过的 UID（按当前会话）+ 已用 token 集合（防回放）
        self._consumed_uids: set[bytes] = set()
        self._consumed_tokens: set[str] = set()

    # ── 配置 / 注入 ─────────────────────────────────────────

    def attach(self, supervisor: Any = None, alert: Any = None) -> None:
        if supervisor is not None:
            self._supervisor = supervisor
        if alert is not None:
            self._alert = alert

    def is_configured(self) -> bool:
        if not self.enabled:
            return False
        host, _port, _ssl, user, password = self._read_imap_conn()
        return bool(host and user and password)

    def _read_imap_conn(self) -> tuple[str, int, bool, str, str]:
        host = _env("AICQ_IMAP_HOST")
        try:
            port = int(_env("AICQ_IMAP_PORT", "993") or "993")
        except ValueError:
            port = 993
        use_ssl = _bool_env("AICQ_IMAP_USE_SSL", port == 993)
        user = _env("AICQ_IMAP_USER")
        password = _env("AICQ_IMAP_PASSWORD")
        if self.reuse_smtp:
            if not user:
                user = _env("AICQ_SMTP_USER")
            if not password:
                password = _env("AICQ_SMTP_PASSWORD")
        return host, port, use_ssl, user, password

    def _allowed_senders(self) -> set[str]:
        raw = _env("AICQ_SMTP_RECIPIENTS")
        return {r.strip().lower() for r in raw.split(",") if r.strip()}

    # ── 生命周期 ────────────────────────────────────────────

    async def start(self) -> None:
        if not self.is_configured():
            logger.debug("EmailController 未启用或配置不全，跳过启动")
            return
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="email_controller_loop")
        logger.info(
            "EmailController 已启动：每 %ds 轮询 IMAP，允许指令=%s",
            self.poll_interval, sorted(self.allowed_commands),
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except asyncio.TimeoutError:
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        self._task = None
        logger.info("EmailController 已停止")

    # ── 主循环 ──────────────────────────────────────────────

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self._poll_once)
            except Exception:
                logger.exception("EmailController 轮询失败")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval,
                )
            except asyncio.TimeoutError:
                continue

    # ── IMAP 轮询（同步线程内执行）─────────────────────────

    def _poll_once(self) -> None:
        host, port, use_ssl, user, password = self._read_imap_conn()
        if not (host and user and password):
            return

        if use_ssl:
            ctx = ssl.create_default_context()
            try:
                conn = imaplib.IMAP4_SSL(host, port, ssl_context=ctx, timeout=15)
            except TypeError:
                # 旧 Python 不支持 timeout 关键字
                conn = imaplib.IMAP4_SSL(host, port, ssl_context=ctx)
        else:
            try:
                conn = imaplib.IMAP4(host, port, timeout=15)
            except TypeError:
                conn = imaplib.IMAP4(host, port)

        try:
            conn.login(user, password)
            try:
                # 部分服务商（如 163）需要 ID 命令才允许后续操作
                conn.xatom("ID", '("name" "AICQ" "version" "1.0")')
            except (imaplib.IMAP4.error, AttributeError):
                pass
            typ, _ = conn.select("INBOX")
            if typ != "OK":
                logger.warning("IMAP SELECT INBOX 失败: %s", typ)
                return
            typ, data = conn.search(None, "UNSEEN")
            if typ != "OK" or not data or not data[0]:
                return
            for uid in data[0].split():
                if uid in self._consumed_uids:
                    continue
                skip_seen = False
                try:
                    self._handle_uid(conn, uid)
                except _SkipMail:
                    # 非白名单邮件：不标 Seen、不加入 dedupe（下次轮询由 UNSEEN 过滤掉也无所谓）
                    skip_seen = True
                except Exception:
                    logger.exception("处理邮件 UID=%s 失败", uid)
                finally:
                    if not skip_seen:
                        try:
                            conn.store(uid, "+FLAGS", "\\Seen")
                        except imaplib.IMAP4.error:
                            pass
                        self._consumed_uids.add(uid)
        finally:
            with contextlib.suppress(Exception):
                conn.close()
            with contextlib.suppress(Exception):
                conn.logout()

    def _handle_uid(self, conn: imaplib.IMAP4, uid: bytes) -> None:
        typ, msg_data = conn.fetch(uid, "(RFC822)")
        if typ != "OK" or not msg_data:
            return
        # msg_data 形如 [(b'1 (RFC822 {bytes}', b'<raw>'), b')']
        raw_bytes: bytes | None = None
        for part in msg_data:
            if isinstance(part, tuple) and len(part) >= 2 and isinstance(part[1], (bytes, bytearray)):
                raw_bytes = bytes(part[1])
                break
        if not raw_bytes:
            return
        msg = email.message_from_bytes(raw_bytes)

        from_addr = parseaddr(msg.get("From", ""))[1].lower().strip()
        if not from_addr or from_addr not in self._allowed_senders():
            # 非白名单发件人：静默跳过，不记任何日志，也不标记 Seen。
            # 原因：SMTP 邮箱可能不是专用邮箱，会收到营销/验证码等正常邮件，
            # 标 Seen 会污染用户未读状态。返回特殊值让上层跳过 Seen 标记。
            raise _SkipMail()

        in_reply_to = (msg.get("In-Reply-To") or "").strip()
        subject = (msg.get("Subject") or "").strip()
        body = self._extract_text_body(msg)
        # token 专用：不剩除引用历史，以便从“> TOKEN: xxx”这种回复引用中提取
        body_full = self._extract_text_body(msg, keep_quotes=True)

        logger.info(
            "邮件収到 from=%s subject=%r in_reply_to=%r body_len=%d full_len=%d",
            from_addr, subject, in_reply_to, len(body), len(body_full),
        )

        # 指令优先从正文首段抓，其次主题
        cmd = self._extract_command(body) or self._extract_command(subject)
        if not cmd:
            logger.debug("邮件指令拒绝：未识别到指令 subject=%r", subject)
            return
        if cmd not in self.allowed_commands:
            logger.warning("邮件指令拒绝：指令 %s 不在允许列表 %s", cmd, self.allowed_commands)
            return

        # REQUEST = 握手：白名单内即放行，不验 token；bot 会主动回一封带新 token 的邮件
        if cmd == "REQUEST":
            logger.info("邮件握手请求 from=%s", from_addr)
            self._dispatch_command("REQUEST", from_addr, msg.get("Message-ID", ""))
            return

        # token 提取：从包含引用历史的原始正文中搜（用户只需"回复"即可）
        # ⚠️ 关键：必须收集**所有候选**，因为回复链里同时存在多封 TOKEN 邮件，
        #         逐个交给注册表验证，命中哪个用哪个；绝不能把多个 token 的 hex 拼起来。
        candidates: list[str] = []
        seen_cand: set[str] = set()
        for src in (body_full, body, subject):
            for tok in self._extract_tokens(src):
                if tok not in seen_cand:
                    seen_cand.add(tok)
                    candidates.append(tok)
        if not candidates:
            # 调试输出：看看正文里有没有 "TOKEN" 字样，帮用户诊断
            sample = body_full[:600] if body_full else body[:600]
            logger.warning(
                "邮件指令拒绝：缺少 TOKEN cmd=%s from=%s\n【正文预览】\n%s",
                cmd, from_addr, sample,
            )
            return

        # 双重校验：token 必须命中、且其绑定的 Message-ID 与 In-Reply-To 一致
        if self._alert is None:
            logger.error("无 AlertManager 注入，token 无法校验")
            return

        token_norm: str | None = None
        result: str = "missing"
        for cand in candidates:
            if cand in self._consumed_tokens:
                continue  # 已用过的跳过，继续试下一个
            r = self._alert.consume_token(cand, in_reply_to)
            if r in ("ok", "irt_mismatch"):
                token_norm = cand
                result = r
                break
            result = r  # 记下最后一次失败原因
        if token_norm is None:
            logger.warning(
                "邮件指令拒绝：所有 token 候选均无效 cmd=%s candidates=%s recent_msgids=%s",
                cmd, candidates, self._alert.list_recent_msgids(),
            )
            return
        if result == "irt_mismatch":
            # 软提示：仍然放行，但记个警告便于调试
            logger.warning(
                "邮件指令：In-Reply-To 不匹配但 token 有效，放行 cmd=%s irt=%r",
                cmd, in_reply_to,
            )
        self._consumed_tokens.add(token_norm)

        logger.info(
            "邮件指令接受 cmd=%s from=%s msgid=%s",
            cmd, from_addr, msg.get("Message-ID", ""),
        )
        self._dispatch_command(cmd, from_addr, msg.get("Message-ID", ""))

    def _dispatch_command(self, cmd: str, from_addr: str, in_reply_to: str) -> None:
        loop = app_state.main_loop
        if loop is None:
            logger.error("main_loop 未初始化，无法分发指令 %s", cmd)
            return

        sup = self._supervisor
        alert = self._alert

        async def _reply(msg: str) -> None:
            if alert is None:
                return
            with contextlib.suppress(Exception):
                await alert.notify_command_result(cmd, msg, in_reply_to=in_reply_to)

        async def _reply_new_session(msg: str) -> None:
            """握手专用：开一封新邮件，alerting 内会另起 token + Message-ID。"""
            if alert is None:
                return
            with contextlib.suppress(Exception):
                await alert.notify_command_result(cmd, msg, in_reply_to=None)

        if cmd == "REQUEST":
            asyncio.run_coroutine_threadsafe(
                _reply_new_session(
                    "握手成功，已为本次会话生成一次性令牌。\n"
                    "请直接【回复本邮件】，在【正文首行】写入指令"
                    "（RESTART / STATUS / STOP / KILL_AICQ），并保留下方 TOKEN 行。"
                ),
                loop,
            )
            return

        if cmd == "RESTART":
            if sup is None or not sup.is_configured():
                asyncio.run_coroutine_threadsafe(
                    _reply("supervisor 未启用，无法执行重启"), loop,
                )
                return
            sup.request_restart(f"email_command/{from_addr}")
            asyncio.run_coroutine_threadsafe(
                _reply("已触发 NapCat 重启流程，稍后自动观察恢复情况"), loop,
            )

        elif cmd == "STOP":
            if sup is None or not sup.is_configured():
                asyncio.run_coroutine_threadsafe(
                    _reply("supervisor 未启用，无法执行停止"), loop,
                )
                return

            async def _do_stop() -> None:
                result = await sup.request_stop(f"email_command/{from_addr}")
                await _reply(result)

            asyncio.run_coroutine_threadsafe(_do_stop(), loop)

        elif cmd == "STATUS":
            client = app_state.napcat_client
            connected = bool(getattr(client, "connected", False))
            bot_id = getattr(client, "bot_id", "") if client else ""
            asyncio.run_coroutine_threadsafe(
                _reply(
                    f"NapCat connected={connected}, bot_id={bot_id or 'N/A'}"
                ),
                loop,
            )

        elif cmd == "KILL_AICQ":
            async def _do_kill() -> None:
                await _reply("AICQ 即将退出（需要本机有 supervisor 守护进程才能自动拉起）")
                # 给 SMTP flush 一点时间
                await asyncio.sleep(2.0)
                logger.warning("收到 KILL_AICQ 邮件指令，进程退出")
                # noinspection PyProtectedMember
                os._exit(0)

            asyncio.run_coroutine_threadsafe(_do_kill(), loop)

    # ── 解析工具 ────────────────────────────────────────────

    def _extract_text_body(self, msg: email.message.Message, keep_quotes: bool = False) -> str:
        """从 message 中抓 text/plain 正文；失败时回退 text/html 去标签。

        :param keep_quotes: True 时不剩除 "> "、"在 ... 写道:" 等引用限化符，
            以便 token 提取能读到被引用起来的原邮件 TOKEN 行。
        """
        text_parts: list[str] = []
        html_parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    text_parts.append(self._decode_part(part))
                elif ctype == "text/html":
                    html_parts.append(self._decode_part(part))
        else:
            ctype = msg.get_content_type()
            payload = self._decode_part(msg)
            if ctype == "text/plain":
                text_parts.append(payload)
            elif ctype == "text/html":
                html_parts.append(payload)
            else:
                text_parts.append(payload)

        if text_parts:
            raw = "\n".join(text_parts)
        elif html_parts:
            raw = re.sub(r"<[^>]+>", "", "\n".join(html_parts))
        else:
            raw = ""

        if keep_quotes:
            # 还原 HTML 里被转义的实体，顺便去掉行首可能的 "> " 以免与 TOKEN 默认词边界冲突
            raw = raw.replace("&gt;", ">").replace("&nbsp;", " ")
            return raw

        # 截掉引用历史（常见分隔：>, "在 ... 写道:", "On ... wrote:", "------"）
        cleaned_lines: list[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith(">"):
                continue
            if re.match(r"^(在|At|On)\s.+(写道|wrote)[:：]\s*$", stripped):
                break
            if re.match(r"^-{4,}\s*Original Message", stripped, re.IGNORECASE):
                break
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _decode_part(part: email.message.Message) -> str:
        try:
            payload = part.get_payload(decode=True)
            if not payload:
                return ""
            charset = part.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
        except (LookupError, ValueError, AttributeError):
            return ""

    @staticmethod
    def _extract_tokens(text: str) -> list[str]:
        """从文本里提取所有疑似 token 候选，按出现顺序去重返回。

        QQ/163 邮件回复会把整条历史拼接进来，因此正文里可能同时出现多个 TOKEN 行
        （旧的、新的都在）。绝不能把它们的 hex 字符串拼到一起——必须分别取出
        每个 TOKEN: 之后**严格连续**的 16~64 hex，全部交给注册表试一遍。
        """
        if not text:
            return []
        # 去掉 HTML 零宽 / 软连字符 / BOM 等会切断 hex 串的隐形字符
        normalized = (
            text.replace("\u200b", "")
                .replace("\u200c", "")
                .replace("\u200d", "")
                .replace("\u2060", "")
                .replace("\ufeff", "")
                .replace("\xad", "")
        )
        results: list[str] = []
        seen: set[str] = set()
        # 严格匹配：TOKEN: 之后必须是连续的 hex（允许中间只有空白）
        for m in re.finditer(
            r"TOKEN\s*[:：]\s*([0-9a-fA-F]{16,64})", normalized, re.IGNORECASE
        ):
            tok = m.group(1).lower()
            if tok not in seen:
                seen.add(tok)
                results.append(tok)
        return results

    @staticmethod
    def _extract_token(text: str) -> str | None:
        """单值版本（兜底用）。优先返回最后一个匹配（回复链中最新的一封）。"""
        toks = EmailController._extract_tokens(text)
        return toks[-1] if toks else None

    @staticmethod
    def _extract_command(text: str) -> str | None:
        # 只取去引用后第一段（前 5 行）的第一个匹配，避免抓到"上一封我的告警邮件"中的关键字
        head = "\n".join((text or "").splitlines()[:5])
        m = _COMMAND_RE.search(head)
        return m.group(1).upper() if m else None
