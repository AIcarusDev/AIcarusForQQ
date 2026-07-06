"""Local Core platform runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from platforms.base import PlatformAccount, PlatformToolContext, PlatformWorldBlock

from .prompt import render_dialogue


@dataclass
class CoreRuntime:
    config: dict[str, Any] = field(default_factory=dict)

    platform: str = "core"
    surface: str = "session"

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

    def world_block(
        self,
        session: Any,
        *,
        current_time: str,
        chat_log: Any,
        forward_content: Any = "",
    ) -> PlatformWorldBlock:
        return PlatformWorldBlock(
            name=self.platform,
            attrs={"transport": "webui"},
            content=render_dialogue(session),
        )
