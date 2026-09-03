from __future__ import annotations

import asyncio

import app_state
import database
from llm.session import sessions
from platforms.core.session_context import CLOSED_PLATFORM_FOCUS, CORE_MAIN_FOCUS
from platforms.focus import FocusRef
from tools.core._chat_notes import (
    build_core_platform_note,
    core_focus_note_actions,
    persist_core_platform_note,
)


def _setup_db(monkeypatch, tmp_path):
    db_path = tmp_path / "AICQ.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    asyncio.run(database.init_db())
    return db_path


def test_core_focus_transitions_map_to_note_actions():
    qq_focus = FocusRef("qq", "private", "123", "Alice")

    assert core_focus_note_actions(CLOSED_PLATFORM_FOCUS, CORE_MAIN_FOCUS) == ["enter"]
    assert core_focus_note_actions(qq_focus, CORE_MAIN_FOCUS) == ["enter"]
    assert core_focus_note_actions(CORE_MAIN_FOCUS, CLOSED_PLATFORM_FOCUS) == ["leave"]
    assert core_focus_note_actions(CORE_MAIN_FOCUS, qq_focus) == ["leave"]
    assert core_focus_note_actions(CORE_MAIN_FOCUS, CORE_MAIN_FOCUS) == []
    assert core_focus_note_actions(qq_focus, CLOSED_PLATFORM_FOCUS) == []


def test_core_platform_note_persists_as_chat_note(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)
    monkeypatch.setattr(app_state, "SELF_NAME", "Aicarus")
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        entry = build_core_platform_note(
            "enter",
            timestamp="2026-07-08T12:00:00+08:00",
        )
        asyncio.run(persist_core_platform_note(entry))

        messages = asyncio.run(database.load_chat_messages(CORE_MAIN_FOCUS.key(), limit=10))
    finally:
        sessions.clear()
        sessions.update(original_sessions)

    assert len(messages) == 1
    assert messages[0]["role"] == "note"
    assert messages[0]["message_id"].startswith("core_note_enter_")
    assert "Aicarus" in messages[0]["content"]
    assert messages[0]["content_type"] == "core_platform_enter"
    assert messages[0]["content_segments"][0]["type"] == "platform_presence"
