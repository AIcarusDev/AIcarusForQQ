"""Scroll the local Core chat history window."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Callable, Literal

from pydantic import Field, RootModel

from platforms.chat.history_window import scroll_down, scroll_to_latest, scroll_to_message, scroll_up
from platforms.core.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools.core_chat")


class ScrollUpArgs(ToolArgsModel):
    action: Literal["up"] = Field(description="向上查看更早的聊天记录。")
    count: int = Field(default=10, ge=1, le=10, description="本次向上滚动几条消息，默认 10。")


class ScrollDownArgs(ToolArgsModel):
    action: Literal["down"] = Field(description="向下查看更新的聊天记录。")
    count: int = Field(default=10, ge=1, le=10, description="本次向下滚动几条消息，默认 10。")


class ScrollJumpArgs(ToolArgsModel):
    action: Literal["jump"] = Field(description="跳转到指定消息。")
    message_id: str = Field(min_length=1, description="目标消息 ID。")


class ScrollLatestArgs(ToolArgsModel):
    action: Literal["down_to_latest"] = Field(description="直接跳回 core 聊天窗口最底部，即最新消息。")


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
    description="翻阅当前 core 1v1 聊天记录。该工具只改变下一轮 <dialogue> 可见窗口，不直接返回聊天内容。",
    args_model=ScrollChatLogArgs,
)

REQUIRES_CONTEXT: list[str] = ["core_session_provider"]


def make_handler(core_session_provider: Callable[[], Any | None]) -> Callable:
    core_session_provider = ensure_session_provider(core_session_provider)

    def execute(
        action: str = "",
        count: int = 10,
        message_id: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        action = (action or "").strip().lower()
        if action not in ("up", "down", "jump", "down_to_latest"):
            return {
                "ok": False,
                "action": action,
                "moved": False,
                "error": f"未知 action: {action!r}，应为 up / down / jump / down_to_latest 之一。",
            }

        session = core_session_provider()
        if session is None:
            return {
                "ok": False,
                "action": action,
                "moved": False,
                "error": NO_CURRENT_SESSION_ERROR,
            }

        if action == "up":
            result = scroll_up(session, count=count)
        elif action == "down":
            result = scroll_down(session, count=count)
        elif action == "jump":
            result = scroll_to_message(session, message_id=message_id)
        else:
            result = scroll_to_latest(session)

        result.setdefault("action", action)
        view = session.chat_window_view
        logger.info(
            "[core_chat] scroll_chat_log: action=%s count=%s message_id=%s moved=%s mode=%s top=%s",
            action,
            count,
            message_id,
            result.get("moved"),
            view.get("mode"),
            view.get("top_db_id"),
        )
        return result

    return execute
