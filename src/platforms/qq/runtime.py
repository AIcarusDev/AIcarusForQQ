"""QQ platform runtime wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from platforms.base import PlatformAccount, PlatformToolContext

from .adapter import QQAdapterClient
from .adapter.config import runtime_adapter_config
from .prompt import render_platform_block


@dataclass
class QQRuntime:
    config: dict[str, Any]
    client: QQAdapterClient | None = None
    supervisor: Any = None

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
        if self.client is None:
            return PlatformAccount(self.platform)
        return PlatformAccount(
            platform=self.platform,
            account_id=str(getattr(self.client, "bot_id", "") or ""),
            account_name=str(getattr(self.client, "bot_name", "") or ""),
        )

    @property
    def adapter_config(self) -> dict[str, Any]:
        return runtime_adapter_config(self.config)

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
            self.client.adapter = adapter_cfg.get("adapter", "napcat")
            self.client.adapter_name = adapter_cfg.get("name", "")
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

    def render_world(self, session: Any, *, current_time: str, chat_log: Any, forward_content: Any = "") -> Any:
        from llm.session import sessions

        current_key = session.key if getattr(session, "key", "") else ""
        account = self.account
        return render_platform_block(
            session=session,
            sessions=sessions,
            current_key=current_key,
            current_time=current_time,
            chat_log=chat_log,
            forward_content=forward_content,
            account_id=account.account_id,
            account_name=account.account_name,
        )
