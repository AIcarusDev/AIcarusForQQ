"""QQ platform runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platforms.attention import AttentionEvent
from platforms.base import PlatformAccount, PlatformToolContext, PlatformWorldBlock

from .adapter import QQAdapterClient
from .adapter.config import runtime_adapter_config
from .prompt import render_platform_content
from .session_context import HOME_FOCUS


@dataclass
class QQRuntime:
    config: dict[str, Any]
    client: QQAdapterClient | None = None
    supervisor: Any = None
    _account_id: str = ""
    _account_name: str = ""

    platform: str = "qq"
    surface: str = "session"

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False))

    @property
    def connected(self) -> bool:
        return bool(self.client and self.client.connected)

    @property
    def account(self) -> PlatformAccount:
        client_account_id = str(getattr(self.client, "bot_id", "") or "") if self.client is not None else ""
        if client_account_id:
            account_id = client_account_id
            account_name = self._account_name if self._account_id in ("", client_account_id) else ""
        else:
            account_id = self._account_id
            account_name = self._account_name
        return PlatformAccount(
            platform=self.platform,
            account_id=account_id,
            account_name=account_name,
        )

    def update_account(self, account_id: str, account_name: str) -> None:
        self._account_id = str(account_id or "")
        self._account_name = str(account_name or "")

    @property
    def adapter_config(self) -> dict[str, Any]:
        return runtime_adapter_config(self.config)

    def main_focus(self):
        return HOME_FOCUS

    def ensure_client(self, *, bot_name: str) -> QQAdapterClient | None:
        if not self.enabled:
            self.client = None
            return None
        adapter_cfg = self.adapter_config
        if self.client is None:
            self.client = QQAdapterClient(
                bot_name=bot_name,
                adapter=adapter_cfg.get("adapter", "napcat"),
                adapter_name=adapter_cfg.get("name", ""),
            )
        else:
            self.client.bot_name = bot_name
            self.client.set_configured_adapter(
                adapter_cfg.get("adapter", "auto"),
                adapter_cfg.get("name", ""),
            )
        return self.client

    def tool_context(self, session: Any, app_config: dict[str, Any]) -> PlatformToolContext:
        import app_state

        return PlatformToolContext(
            platform=self.platform,
            runtime=self,
            session=session,
            config=app_config,
            loop=getattr(app_state, "main_loop", None),
        )

    async def _fetch_quoted_message(self, ref_id: str) -> dict | None:
        if self.client is None or not self.client.connected:
            return None

        try:
            msg_id_int = int(ref_id)
        except (ValueError, TypeError):
            return None

        msg_data = await self.client.send_api("get_msg", {"message_id": msg_id_int})
        if not msg_data:
            return None

        sender = msg_data.get("sender", {})
        sender_card = str(sender.get("card", "") or "")
        sender_nickname = str(sender.get("nickname", "") or "")
        sender_name = sender_card or sender_nickname or str(sender.get("user_id", "未知"))
        segs = msg_data.get("message") or []
        from platforms.qq.adapter import segments as qq_segments

        return {
            "message_id": ref_id,
            "sender_name": sender_name,
            "sender_card": sender_card,
            "sender_nickname": sender_nickname,
            "content": qq_segments.qq_adapter_segments_to_text(segs),
            "content_type": "text",
        }

    async def prefetch_quoted_messages(self, session: Any) -> None:
        from platforms.chat.quote_prefetch import prefetch_quoted_messages

        await prefetch_quoted_messages(session, self._fetch_quoted_message)

    def attention_events(self, *, now: Any = None) -> list[AttentionEvent]:
        return []

    def world_block(
        self,
        session: Any,
        *,
        current_time: str,
        chat_log: Any,
        forward_content: Any = "",
    ) -> PlatformWorldBlock:
        from llm.session import sessions

        current_key = session.key if getattr(session, "key", "") else ""
        account = self.account
        return PlatformWorldBlock(
            name=self.platform,
            attrs={
                "account_id": account.account_id,
                "account_name": account.account_name,
            },
            content=render_platform_content(
                session=session,
                sessions=sessions,
                current_key=current_key,
                chat_log=chat_log,
                forward_content=forward_content,
            ),
        )
