"""Cancel one pending high-risk browser-image send batch."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import Field

from browser.image_confirmation import cancel_pending, current_pending
from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract


class CancelBrowserImageSendArgs(ToolArgsModel):
    batch_id: str = Field(min_length=1, description="确认块中显示的 batch_id。")


TOOL_CONTRACT = ToolContract(
    name="cancel_browser_image_send",
    description="取消本轮展示的高风险浏览器原图发送批次。",
    args_model=CancelBrowserImageSendArgs,
)

EXTERNALLY_PERCEPTIBLE = False
TOOL_EFFECT = {"surface": "qq", "kind": "session_state"}
REQUIRES_CONTEXT = ["qq_session_provider"]


def is_available(qq_session_provider: Callable[[], Any | None], **_: Any) -> bool:
    provider = ensure_session_provider(qq_session_provider)
    session = provider()
    return session is not None and current_pending(session) is not None


def make_handler(qq_session_provider: Callable[[], Any | None]) -> Callable:
    provider = ensure_session_provider(qq_session_provider)

    def execute(batch_id: str, **_: Any) -> dict:
        session = provider()
        if session is None:
            return {"error": NO_CURRENT_SESSION_ERROR}
        if not cancel_pending(
            session,
            batch_id=str(batch_id or ""),
            reason="model_cancelled",
        ):
            return {"error": "图片确认批次不存在、已过期或不属于当前会话。"}
        return {"cancelled": True, "batch_id": str(batch_id)}

    return execute
