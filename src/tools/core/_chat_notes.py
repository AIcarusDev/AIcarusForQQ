"""Persist local Core chat page note events."""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from platforms.core.session_context import CORE_MAIN_FOCUS, is_core_main_focus

logger = logging.getLogger("AICQ.tools.core.notes")

_NOTE_TEXT = {
    "enter": "进入本页",
    "leave": "离开本页",
}


def agent_self_name() -> str:
    import app_state

    return str(getattr(app_state, "SELF_NAME", "") or "Agent").strip() or "Agent"


def build_core_platform_note(action: str, *, timestamp: str | None = None) -> dict[str, Any]:
    """Build a note row for the local Core chat timeline."""

    if action not in _NOTE_TEXT:
        raise ValueError(f"unknown core platform note action: {action}")
    name = agent_self_name()
    content = f"{name} {_NOTE_TEXT[action]}"
    if timestamp is None:
        import app_state

        timestamp = datetime.now(getattr(app_state, "TIMEZONE", None)).isoformat()
    return {
        "role": "note",
        "message_id": f"core_note_{action}_{uuid.uuid4().hex}",
        "sender_id": "core",
        "sender_name": name,
        "timestamp": timestamp,
        "content": content,
        "content_type": f"core_platform_{action}",
        "content_segments": [
            {
                "type": "platform_presence",
                "platform": "core",
                "action": action,
                "actor_name": name,
                "text": content,
            }
        ],
    }


def core_focus_note_actions(prev_focus: Any, next_focus: Any) -> list[str]:
    """Return note actions needed for a focus transition relative to Core chat."""

    was_core = is_core_main_focus(prev_focus)
    is_core = is_core_main_focus(next_focus)
    if was_core == is_core:
        return []
    return ["leave"] if was_core else ["enter"]


async def persist_core_platform_note(entry: dict[str, Any]) -> None:
    from database import save_chat_message, upsert_chat_session
    from llm.session import get_or_create_session
    from web.debug_server import broadcast_chat_event

    session_key = CORE_MAIN_FOCUS.key()
    session = get_or_create_session(CORE_MAIN_FOCUS)
    if not getattr(session, "conv_type", ""):
        session.set_conversation_meta(
            CORE_MAIN_FOCUS.target_type,
            CORE_MAIN_FOCUS.target_id,
            CORE_MAIN_FOCUS.target_name,
            platform=CORE_MAIN_FOCUS.platform,
        )
    session.add_to_context(entry)
    await save_chat_message(session_key, entry)
    await upsert_chat_session(
        session_key,
        CORE_MAIN_FOCUS.target_type,
        CORE_MAIN_FOCUS.target_id,
        CORE_MAIN_FOCUS.target_name,
    )
    await broadcast_chat_event({
        "type": "system_notice",
        "conv_id": session_key,
        "conv_name": CORE_MAIN_FOCUS.target_name,
        "conv_type": CORE_MAIN_FOCUS.target_type,
        "entry": entry,
    })


def record_core_focus_transition(prev_focus: Any, next_focus: Any) -> list[dict[str, Any]]:
    """Schedule note persistence for Core chat enter/leave transitions."""

    actions = core_focus_note_actions(prev_focus, next_focus)
    if not actions:
        return []

    import app_state

    loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        logger.debug("[core_notes] skip note without running loop actions=%s", actions)
        return []

    entries = [build_core_platform_note(action) for action in actions]
    for entry in entries:
        asyncio.run_coroutine_threadsafe(persist_core_platform_note(entry), loop)
    return entries
