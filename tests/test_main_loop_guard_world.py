from __future__ import annotations

from types import SimpleNamespace

import app_state
from consciousness import main_loop
from llm.session import create_session, get_or_create_session, sessions
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


def test_build_tool_collection_does_not_treat_core_guardian_as_qq_user(monkeypatch):
    session = create_session(FocusRef("core", "private", "guardian"))
    captured = {}

    def fake_build_tools(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(active_specs={})

    monkeypatch.setattr(app_state, "namespace_runtime_state", None)
    monkeypatch.setattr(app_state, "GEN", {})
    monkeypatch.setattr(app_state, "consciousness_flow", SimpleNamespace(next_seq=1))
    monkeypatch.setattr(app_state, "current_focus", session.focus)
    monkeypatch.setattr(app_state, "config", {"tts": {"enabled": False}, "vision": False})
    monkeypatch.setattr(app_state, "adapter", SimpleNamespace(provider=None))
    monkeypatch.setattr(main_loop, "build_tools", fake_build_tools)

    collection = main_loop._build_tool_collection(session)

    assert collection.active_specs == {}
    assert captured["group_id"] is None
    assert captured["user_id"] is None


def test_build_tool_collection_injects_qq_private_user_id(monkeypatch):
    session = create_session(FocusRef("qq", "private", "12345"))
    captured = {}

    def fake_build_tools(config, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(active_specs={})

    monkeypatch.setattr(app_state, "namespace_runtime_state", None)
    monkeypatch.setattr(app_state, "GEN", {})
    monkeypatch.setattr(app_state, "consciousness_flow", SimpleNamespace(next_seq=1))
    monkeypatch.setattr(app_state, "current_focus", session.focus)
    monkeypatch.setattr(app_state, "config", {"tts": {"enabled": False}, "vision": False})
    monkeypatch.setattr(app_state, "adapter", SimpleNamespace(provider=None))
    monkeypatch.setattr(main_loop, "build_tools", fake_build_tools)

    main_loop._build_tool_collection(session)

    assert captured["group_id"] is None
    assert captured["user_id"] == 12345
