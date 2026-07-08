"""Local Core platform runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from platforms.attention import AttentionEvent
from platforms.base import PlatformAccount, PlatformToolContext, PlatformWorldBlock

from .prompt import render_dialogue
from .session_context import CORE_MAIN_FOCUS, core_surface_for_focus, is_closed_platform_focus


@dataclass
class CoreRuntime:
    config: dict[str, Any] = field(default_factory=dict)

    platform: str = "core"

    @property
    def enabled(self) -> bool:
        return True

    @property
    def connected(self) -> bool:
        return True

    @property
    def account(self) -> PlatformAccount:
        return PlatformAccount(
            platform=self.platform,
            account_id=str(self.config.get("account_id") or "core"),
            account_name=str(self.config.get("account_name") or "Core"),
        )

    def main_focus(self):
        return CORE_MAIN_FOCUS

    def surface(self, session: Any) -> str:
        return core_surface_for_focus(getattr(session, "focus", None))

    def tool_context(self, session: Any, app_config: dict[str, Any]) -> PlatformToolContext:
        import app_state

        return PlatformToolContext(
            platform=self.platform,
            runtime=self,
            session=session,
            config=app_config,
            loop=getattr(app_state, "main_loop", None),
        )

    async def prefetch_quoted_messages(self, session: Any) -> None:
        return None

    def attention_events(self, *, now: Any = None) -> list[AttentionEvent]:
        try:
            from llm.session import sessions
        except Exception:
            return []

        session = sessions.get(CORE_MAIN_FOCUS.key())
        if session is None or getattr(session, "unread_count", 0) <= 0:
            return []
        occurred_at = ""
        for message in reversed(getattr(session, "context_messages", []) or []):
            occurred_at = str(message.get("timestamp", "") or "").strip()
            if occurred_at:
                break
        return [AttentionEvent(name=self.platform, level="mention", occurred_at=occurred_at)]

    def world_block(
        self,
        session: Any,
        *,
        current_time: str,
        chat_log: Any,
        forward_content: Any = "",
    ) -> PlatformWorldBlock:
        if is_closed_platform_focus(getattr(session, "focus", None)):
            return PlatformWorldBlock(
                name="",
                attrs={"page": "none"},
                content=None,
            )
        return PlatformWorldBlock(
            name=self.platform,
            attrs={"transport": "webui"},
            content=render_dialogue(session),
        )
