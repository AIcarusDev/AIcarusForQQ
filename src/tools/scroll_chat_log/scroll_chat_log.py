"""scroll_chat_log.py — 滚动当前会话的聊天窗口

操作 <world> 中唯一的聊天记录视图。本工具不返回任何聊天内容，
只改变下一轮 <world> 中可见的聊天窗口位置。

视口生命周期：
  与会话窗口同寿。bot 离开本会话（shift 走 / 被其它会话抢焦点）后，
  下次回到本会话时由 llm_core 自动重置回 live。
  bot 在本会话内 sleep 后又在本会话被唤醒，则视口保留。
"""

import logging
from typing import Any, Callable

from llm.prompt.history_window import scroll_down, scroll_to_latest, scroll_up

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "all"

DECLARATION: dict = {
    "name": "scroll_chat_log",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["up", "down", "down_to_latest"],
                "description": (
                    "窗口操作。"
                    "up=向上查看更早的历史消息；"
                    "down=向下查看更新的历史消息；"
                    "down_to_latest=直接跳回聊天窗口的最底部(最新消息)。"
                ),
            },
            "impression": {
                "type": "string",
                "description": (
                    "你对'操作前当前可见聊天窗口'的印象。"
                    "应当写成你刚刚看完这一屏后形成的认知，"
                ),
            },
            "motivation": {"type": "string"},
        },
        "required": ["action", "impression", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        action: str = "",
        impression: str = "",
        motivation: str = "",
        **kwargs,
    ) -> dict:
        action = (action or "").strip().lower()
        if action not in ("up", "down", "down_to_latest"):
            return {
                "ok": False,
                "action": action,
                "moved": False,
                "error": f"未知 action: {action!r}，应为 up / down / down_to_latest 之一。",
            }

        if action == "up":
            result = scroll_up(session)
        elif action == "down":
            result = scroll_down(session)
        else:
            result = scroll_to_latest(session)

        # action 字段始终回显，便于回看意识流时定位
        result.setdefault("action", action)
        view = session.chat_window_view
        logger.info(
            "[tools] scroll_chat_log: action=%s moved=%s mode=%s top=%s",
            action,
            result.get("moved"),
            view.get("mode"),
            view.get("top_db_id"),
        )
        return result

    return execute