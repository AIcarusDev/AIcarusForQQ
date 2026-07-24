"""Confirm one pending high-risk browser-image send batch."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field

from browser.image_confirmation import current_pending
from platforms.qq.session_context import ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract


class ConfirmBrowserImageSendArgs(ToolArgsModel):
    batch_id: str = Field(min_length=1, description="确认块中显示的 batch_id。")


TOOL_CONTRACT = ToolContract(
    name="confirm_browser_image_send",
    description=(
        "确认并发送本轮展示的高风险浏览器原图批次。"
        "只能确认当前会话、当前用户消息版本和当前轮可见的批次。"
    ),
    args_model=ConfirmBrowserImageSendArgs,
)

EXTERNALLY_PERCEPTIBLE = True
TOOL_EFFECT = {"surface": "qq", "kind": "session_write"}
REQUIRES_CONTEXT = ["qq_session_provider", "qq_client", "config"]


def is_available(qq_session_provider: Callable[[], Any | None], **_: Any) -> bool:
    provider = ensure_session_provider(qq_session_provider)
    session = provider()
    return session is not None and current_pending(session) is not None


def make_handler(
    qq_session_provider: Callable[[], Any | None],
    qq_client: Any,
    config: dict | None,
) -> Callable:
    from .send_message.send_message import make_handler as make_send_handler

    send = make_send_handler(qq_session_provider, qq_client, config)

    def execute(batch_id: str, **_: Any) -> dict:
        return send(_confirmed_batch_id=str(batch_id or ""))

    return execute
