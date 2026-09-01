"""Shared platform runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .attention import AttentionEvent


@dataclass(frozen=True)
class PlatformAccount:
    platform: str
    account_id: str = ""
    account_name: str = ""


@dataclass
class PlatformToolContext:
    platform: str
    runtime: Any
    session: Any
    config: dict[str, Any]
    loop: Any = None


@dataclass
class PlatformWorldBlock:
    name: str
    attrs: dict[str, str] = field(default_factory=dict)
    content: Any = ""


class PlatformRuntime(Protocol):
    platform: str

    @property
    def enabled(self) -> bool: ...

    @property
    def connected(self) -> bool: ...

    @property
    def state(self) -> str: ...

    @property
    def account(self) -> PlatformAccount: ...

    @property
    def config(self) -> dict[str, Any]: ...

    def main_focus(self) -> Any: ...

    def tool_context(self, session: Any, app_config: dict[str, Any]) -> PlatformToolContext: ...

    async def prefetch_quoted_messages(self, session: Any) -> None: ...

    def attention_events(self, *, now: Any = None) -> list[AttentionEvent]: ...

    def world_block(
        self,
        session: Any,
        *,
        current_time: str,
        chat_log: Any,
        forward_content: Any = "",
    ) -> PlatformWorldBlock: ...
