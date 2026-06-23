"""browse_forward.py - open and browse merged-forward message views."""

from __future__ import annotations

from typing import Any, Callable

from llm.forward_browser import make_handler as make_forward_browser_handler

SCOPE: str = "all"
REQUIRES_CONTEXT: list[str] = ["session", "qq_adapter_client"]

DECLARATION: dict = {
    "name": "browse_forward",
    "description": (
        "打开或浏览当前会话中的合并转发消息。"
        "看到 <content type=\"forward\" openable=\"true\"> 时用 action=open 和 id 打开；"
        "已打开后可用 next_page、prev_page、back 或 close_all 翻页、返回或关闭。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["open", "next_page", "prev_page", "back", "close_all"],
                "description": "open=打开；next_page/prev_page=翻页；back=回退上一层；close_all=关闭所有浏览窗口。",
            },
            "id": {
                "type": "string",
                "description": "action=open 时必填。顶层使用真实 QQ message_id，嵌套层使用 fwd: 开头的虚拟 ID。",
            },
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "open"}}, "required": ["action"]},
                "then": {"required": ["id"]},
            }
        ],
    },
}


def make_handler(session: Any, qq_adapter_client: Any) -> Callable:
    return make_forward_browser_handler(session, qq_adapter_client)


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
