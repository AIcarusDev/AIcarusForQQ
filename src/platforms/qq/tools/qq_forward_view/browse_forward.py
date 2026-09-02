"""browse_forward.py - open and browse merged-forward message views."""

from __future__ import annotations

from typing import Any, Callable, Literal

from pydantic import Field, RootModel

from llm.forward_browser import make_handler as make_forward_browser_handler
from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract

REQUIRES_CONTEXT: list[str] = ["qq_session_provider", "qq_client"]

class BrowseForwardOpenArgs(ToolArgsModel):
    action: Literal["open"] = Field(description="打开。")
    id: str = Field(min_length=1, description="action=open 时必填。顶层使用真实 QQ message_id，嵌套层使用 fwd: 开头的虚拟 ID。")


class BrowseForwardNavArgs(ToolArgsModel):
    action: Literal["next_page", "prev_page", "back", "close_all"] = Field(
        description="next_page/prev_page=翻页；back=回退上一层；close_all=关闭所有浏览窗口。"
    )


class BrowseForwardArgs(RootModel[BrowseForwardOpenArgs | BrowseForwardNavArgs]):
    pass


TOOL_CONTRACT = ToolContract(
    name="browse_forward",
    description=(
        "打开或浏览当前会话中的合并转发消息。"
        "看到 <content type=\"forward\" openable=\"true\"> 时用 action=open 和 id 打开；"
        "open 也可以用于打开合并转发中嵌套的合并转发。"
        "已打开后可根据需要用 next_page、prev_page、back 或 close_all 翻页、返回或关闭。"
        "若离开当前会话，合并转发浏览窗口会自动关闭。"
    ),
    args_model=BrowseForwardArgs,
)


def make_handler(qq_session_provider: Callable[[], Any | None], qq_client: Any) -> Callable:
    qq_session_provider = ensure_session_provider(qq_session_provider)

    def execute(**kwargs: Any) -> dict:
        session = qq_session_provider()
        if session is None:
            return {"ok": False, "error": NO_CURRENT_SESSION_ERROR}
        return make_forward_browser_handler(session, qq_client)(**kwargs)

    return execute


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    repaired = dict(args)
    changes: list[str] = []
    action = str(repaired.get("action") or "").strip()
    if action != repaired.get("action"):
        repaired["action"] = action
        changes.append("action: trimmed surrounding whitespace")
    if action == "open":
        target_id = str(repaired.get("id") or "").strip()
        if target_id != repaired.get("id"):
            repaired["id"] = target_id
            changes.append("id: trimmed surrounding whitespace")
        if not target_id:
            return repaired, changes, "id is required when action=open"
    return repaired, changes, None


