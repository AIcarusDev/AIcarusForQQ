"""scroll_chat_log.py — 滚动当前会话的聊天窗口

操作 <world> 中唯一的聊天记录视图。本工具不返回任何聊天内容，
只改变下一轮 <world> 中可见的聊天窗口位置。

视口生命周期：
  与会话窗口同寿。bot 离开本会话（enter_qq_session 走 / 被其它会话抢焦点）后，
  下次回到本会话时自动重置回 live。
  bot 在本会话内 sleep 后又在本会话被唤醒，则视口保留。
"""

import logging
from typing import Annotated, Any, Callable, Literal

from pydantic import Field, RootModel

from llm.prompt.history_window import scroll_down, scroll_to_latest, scroll_to_message, scroll_up
from tools.contract import ToolArgsModel, ToolContract

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools")


class ScrollUpArgs(ToolArgsModel):
    action: Literal["up"] = Field(
        description="向上查看更早的历史消息。",
    )
    count: int = Field(
        default=10,
        ge=1,
        le=10,
        description="本次向上滚动几条消息，默认 10。",
    )


class ScrollDownArgs(ToolArgsModel):
    action: Literal["down"] = Field(
        description="向下查看更新的历史消息。",
    )
    count: int = Field(
        default=10,
        ge=1,
        le=10,
        description="本次向下滚动几条消息，默认 10。",
    )


class ScrollJumpArgs(ToolArgsModel):
    action: Literal["jump"] = Field(
        description="跳转到指定消息。",
    )
    message_id: str = Field(
        min_length=1,
        description="目标消息 id。",
    )


class ScrollLatestArgs(ToolArgsModel):
    action: Literal["down_to_latest"] = Field(
        description="直接跳回聊天窗口最底部，即最新消息。",
    )


class ScrollChatLogArgs(
    RootModel[
        Annotated[
            ScrollUpArgs | ScrollDownArgs | ScrollJumpArgs | ScrollLatestArgs,
            Field(discriminator="action"),
        ]
    ]
):
    pass


TOOL_CONTRACT = ToolContract(
    name="scroll_chat_log",
    description=DESCRIPTION,
    args_model=ScrollChatLogArgs,
)

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        action: str = "",
        count: int = 10,
        message_id: str = "",
        **kwargs,
    ) -> dict:
        action = (action or "").strip().lower()
        if action not in ("up", "down", "jump", "down_to_latest"):
            return {
                "ok": False,
                "action": action,
                "moved": False,
                "error": f"未知 action: {action!r}，应为 up / down / jump / down_to_latest 之一。",
            }

        if action == "up":
            result = scroll_up(session, count=count)
        elif action == "down":
            result = scroll_down(session, count=count)
        elif action == "jump":
            result = scroll_to_message(session, message_id=message_id)
        else:
            result = scroll_to_latest(session)

        # action 字段始终回显，便于回看意识流时定位
        result.setdefault("action", action)
        view = session.chat_window_view
        logger.info(
            "[tools] scroll_chat_log: action=%s count=%s message_id=%s moved=%s mode=%s top=%s",
            action,
            count,
            message_id,
            result.get("moved"),
            view.get("mode"),
            view.get("top_db_id"),
        )
        return result

    return execute
