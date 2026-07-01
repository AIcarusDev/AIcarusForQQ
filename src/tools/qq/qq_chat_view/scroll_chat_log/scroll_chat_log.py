"""scroll_chat_log.py — 滚动当前会话的聊天窗口

操作 <world> 中唯一的聊天记录视图。本工具不返回任何聊天内容，
只改变下一轮 <world> 中可见的聊天窗口位置。

视口生命周期：
  与会话窗口同寿。bot 离开本会话（shift 走 / 被其它会话抢焦点）后，
  下次回到本会话时自动重置回 live。
  bot 在本会话内 sleep 后又在本会话被唤醒，则视口保留。
"""

import logging
from typing import Any, Callable

from llm.prompt.history_window import scroll_down, scroll_to_latest, scroll_up

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools")

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
        },
        "required": ["action"],
    },
}

PROMPT_SIGNATURE = """
// 滚动当前会话的聊天窗口，操作的是当前聊天窗口中唯一的聊天记录视图。
// 灵活运用此工具，可以实现自由的浏览聊天记录。
// 每次调用会根据 "action" 参数执行一次滚动操作，滚动后你会看到新的聊天记录内容。
// 使用场合示例：
// - 你需要回顾之前的聊天内容 -> 可以 up 向上滚动查看更早的消息。
// - 你已经在翻阅更早的内容，想回到当前窗口的最新消息 -> 可以 down_to_latest 直接跳转查看当前聊天窗口的最新消息。
// - 任何你觉得当前信息不足，需要/想从历史记录中找到更多上下文的情况。
scroll_chat_log(args: {
  action: "up" | "down" | "down_to_latest"; // 窗口操作。up=向上查看更早的历史消息；down=向下查看更新的历史消息；down_to_latest=直接跳回聊天窗口的最底部(最新消息)。
})
"""

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        action: str = "",
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
