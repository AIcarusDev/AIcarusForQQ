from __future__ import annotations

import app_state
from consciousness import main_loop
from llm.session import get_or_create_session, sessions
from platforms.focus import FocusRef


GROUP_ID = "1030" + "770193"
PRIVATE_ID = "2514" + "624910"
PRIVATE_KEY = f"qq:private:{PRIVATE_ID}"


def test_guard_current_world_uses_session_from_current_focus(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        group = get_or_create_session(FocusRef("qq", "group", GROUP_ID))
        get_or_create_session(FocusRef("qq", "private", PRIVATE_ID))
        monkeypatch.setattr(
            app_state,
            "current_focus",
            FocusRef("qq", "private", PRIVATE_ID),
        )
        monkeypatch.setattr(
            main_loop,
            "build_main_user_prompt",
            lambda session, *, consume_unread=True: {
                "session_key": session.key,
                "consume_unread": consume_unread,
            },
        )

        world = main_loop._build_current_focus_world(group, consume_unread=False)

        assert world == {
            "session_key": PRIVATE_KEY,
            "consume_unread": False,
        }
    finally:
        sessions.clear()
        sessions.update(original_sessions)


def test_guard_snapshot_uses_same_current_focus_session(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        group = get_or_create_session(FocusRef("qq", "group", GROUP_ID))
        get_or_create_session(FocusRef("qq", "private", PRIVATE_ID))
        monkeypatch.setattr(
            app_state,
            "current_focus",
            FocusRef("qq", "private", PRIVATE_ID),
        )
        monkeypatch.setattr(
            main_loop,
            "build_qq_guard_snapshot",
            lambda session: {"session_key": session.key},
        )

        snapshot = main_loop._build_current_focus_guard_snapshot(group)

        assert snapshot == {"session_key": PRIVATE_KEY}
    finally:
        sessions.clear()
        sessions.update(original_sessions)
