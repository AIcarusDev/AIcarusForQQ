"""open_forward_message.py — 打开合并转发消息。"""

from __future__ import annotations

from typing import Any, Callable

from llm.forward_browser import make_open_forward_message_handler

SCOPE: str = "all"
REQUIRES_CONTEXT: list[str] = ["session", "qq_adapter_client"]

DECLARATION: dict = {
    "name": "open_forward_message",
    "description": (
        "打开会话中的合并转发消息（看到 <content type=\"forward\" openable=\"true\"> 时使用）。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "需要打开的合并转发消息 ID。顶层使用真实 QQ message_id，"
                    "嵌套层使用 fwd: 开头的虚拟 ID。"
                ),
            },
        },
        "required": ["id"],
    },
}


def make_handler(session: Any, qq_adapter_client: Any) -> Callable:
    return make_open_forward_message_handler(session, qq_adapter_client)
