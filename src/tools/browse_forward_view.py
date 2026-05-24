"""browse_forward_view.py — 浏览已打开的合并转发视图。"""

from __future__ import annotations

from typing import Any, Callable

from llm.forward_browser import make_browse_forward_view_handler

SCOPE: str = "all"
REQUIRES_CONTEXT: list[str] = ["session", "napcat_client"]

DECLARATION: dict = {
    "name": "browse_forward_view",
    "description": "在已打开的合并转发消息视图中进行翻页、返回或关闭操作。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["next_page", "prev_page", "back", "close_all"],
                "description": (
                    "next_page/prev_page=翻页；back=回退到上一层；"
                    "close_all=关闭所有浏览窗口。"
                ),
            },
        },
        "required": ["action"],
    },
}


def make_handler(session: Any, napcat_client: Any) -> Callable:
    return make_browse_forward_view_handler(session, napcat_client)
