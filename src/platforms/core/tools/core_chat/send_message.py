"""Send a single message to the local Core chat."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable

from pydantic import Field

from platforms.core.session_context import CORE_MAIN_FOCUS, NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools.core_chat")


class SendMessageArgs(ToolArgsModel):
    text: str = Field(min_length=1)
    reply_to: str = Field(default="", description="要引用/回复的 core 消息 ID（可选）。")


TOOL_CONTRACT = ToolContract(
    name="send_message",
    description="向监护人发送一条文字消息。",
    args_model=SendMessageArgs,
)

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "core", "kind": "session_write"}
REQUIRES_CONTEXT: list[str] = ["core_session_provider"]


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    changes: list[str] = []
    text = str(args.get("text") or "").strip()
    if not text:
        return args, changes, "text is empty"
    if text != args.get("text"):
        args = dict(args)
        args["text"] = text
        changes.append("trimmed text")
    if "reply_to" in args and args.get("reply_to") is not None:
        reply_to = str(args.get("reply_to") or "").strip()
        if reply_to != args.get("reply_to"):
            args = dict(args)
            args["reply_to"] = reply_to
            changes.append("trimmed reply_to")
    return args, changes, None


def _schedule(loop: asyncio.AbstractEventLoop, coro: Any) -> None:
    asyncio.run_coroutine_threadsafe(coro, loop)


def make_handler(core_session_provider: Callable[[], Any | None]) -> Callable:
    core_session_provider = ensure_session_provider(core_session_provider)

    def execute(text: str = "", reply_to: str = "", **_kwargs: Any) -> dict[str, Any]:
        import app_state
        from database import save_chat_message, upsert_chat_session
        from llm.core.round_context import get_current_inner_state
        from web.debug_server import broadcast_chat_event

        session = core_session_provider()
        if session is None:
            return {
                "ok": False,
                "to": CORE_MAIN_FOCUS.key(),
                "error": NO_CURRENT_SESSION_ERROR,
                "sent_count": 0,
            }

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        if loop is None or not loop.is_running():
            return {
                "ok": False,
                "to": CORE_MAIN_FOCUS.key(),
                "error": "主事件循环不可用",
                "sent_count": 0,
            }

        snapped_to_latest = False
        if getattr(session, "is_browsing_history", lambda: False)():
            session.reset_chat_window_view()
            snapped_to_latest = True

        now_ts = datetime.now(getattr(app_state, "TIMEZONE", None)).isoformat()
        message_id = f"core_{uuid.uuid4().hex}"
        self_name = str(getattr(session, "_self_name", "") or getattr(app_state, "SELF_NAME", "") or "Core").strip()
        entry: dict[str, Any] = {
            "role": "bot",
            "message_id": message_id,
            "sender_id": "core",
            "sender_name": self_name,
            "timestamp": now_ts,
            "content": text,
            "content_type": "text",
            "content_segments": [{"type": "text", "text": text}],
        }
        if reply_to:
            entry["reply_to"] = str(reply_to)

        if not getattr(session, "conv_type", ""):
            session.set_conversation_meta(
                CORE_MAIN_FOCUS.target_type,
                CORE_MAIN_FOCUS.target_id,
                CORE_MAIN_FOCUS.target_name,
                platform=CORE_MAIN_FOCUS.platform,
            )
        session.add_to_context(entry)

        session_key = CORE_MAIN_FOCUS.key()
        _schedule(loop, save_chat_message(session_key, entry))
        _schedule(
            loop,
            upsert_chat_session(
                session_key,
                CORE_MAIN_FOCUS.target_type,
                CORE_MAIN_FOCUS.target_id,
                CORE_MAIN_FOCUS.target_name,
            ),
        )
        _schedule(
            loop,
            broadcast_chat_event({
                "type": "bot_turn",
                "conv_id": session_key,
                "conv_name": CORE_MAIN_FOCUS.target_name,
                "conv_type": CORE_MAIN_FOCUS.target_type,
                "entries": [entry],
                "inner_state": get_current_inner_state(),
            }),
        )

        result: dict[str, Any] = {
            "ok": True,
            "to": session_key,
            "message_id": message_id,
            "sent_count": 1,
            "failed_count": 0,
            "total_count": 1,
            "interrupted": False,
        }
        if snapped_to_latest:
            result["chat_window"] = {
                "snapped_to_latest": True,
                "reason": "send_message_from_history",
            }
        logger.info("[core_chat] send_message: id=%s", message_id)
        return result

    return execute
