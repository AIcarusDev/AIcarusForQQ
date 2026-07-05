from __future__ import annotations

import asyncio
from types import SimpleNamespace

import app_state
import platforms.qq.handler as qq_handler
from llm.session import ConversationSession, sessions
from tools.core import runtime_manage


def test_runtime_manage_contract_is_single_discriminated_tool():
    declaration = runtime_manage.TOOL_CONTRACT.declaration()

    assert declaration["name"] == "runtime_manage"
    assert "wait_qq_event" not in repr(declaration)
    assert "wait_browser_event" not in repr(declaration)
    assert "oneOf" in declaration["parameters"]
    assert "action" in repr(declaration["parameters"])
    assert "minimum" in repr(declaration["parameters"])
    assert "maximum" in repr(declaration["parameters"])


def test_runtime_manage_repair_normalizes_current_contract_fields():
    repaired, changes = runtime_manage.repair_schema_args(
        {"action": "WAIT", "seconds": "3"}
    )

    assert repaired == {"action": "wait", "seconds": 3}
    assert changes == ["action: normalized", "seconds: string -> int"]

    repaired, changes = runtime_manage.repair_schema_args(
        {"action": "sleep", "minutes": "45"}
    )

    assert repaired == {"action": "sleep", "minutes": 45}
    assert changes == ["minutes: string -> int"]


def test_runtime_manage_wait_counts_from_request_start(monkeypatch):
    sleeps: list[float] = []
    times = iter([105.0, 105.0, 115.0])

    monkeypatch.setattr(runtime_manage.time, "time", lambda: next(times))
    monkeypatch.setattr(runtime_manage.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = runtime_manage.execute(
        action="wait",
        seconds=15,
        _request_started_at=100.0,
    )

    assert sleeps == [10.0]
    assert result["ok"] is True
    assert result["action"] == "wait"
    assert result["requested_seconds"] == 15
    assert result["waited_seconds"] == 10
    assert result["elapsed_seconds"] == 15


def test_runtime_manage_wait_returns_immediately_when_reasoning_already_exceeded(monkeypatch):
    sleeps: list[float] = []
    times = iter([130.0, 130.0, 130.0])

    monkeypatch.setattr(runtime_manage.time, "time", lambda: next(times))
    monkeypatch.setattr(runtime_manage.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = runtime_manage.execute(
        action="wait",
        seconds=15,
        _request_started_at=100.0,
    )

    assert sleeps == []
    assert result["ok"] is True
    assert result["elapsed_seconds"] == 30


def test_runtime_manage_idle_consumes_attention_after_request_start():
    session = SimpleNamespace(
        sleep_wake_event=None,
        sleep_arming=False,
        sleep_pending_wake=True,
        sleep_pending_wake_at=105.0,
        last_wake_reason="被私聊消息叫醒了",
        sleep_wake_from="qq:private:1",
        conv_type="private",
        conv_id="1",
        conv_name="Alice",
    )

    reason = asyncio.run(
        runtime_manage.wait_until_attention(
            session,
            0,
            pending_wake_after=100.0,
        )
    )

    assert reason == "woken"
    assert session.sleep_pending_wake is False
    result = runtime_manage.build_runtime_result(
        session,
        action="idle",
        requested_seconds=1800,
        waited_seconds=0,
        elapsed_since_request=5,
        reason=reason,
    )
    assert result["woke_up_because"] == "被私聊消息叫醒了"
    assert result["woke_from"] == "qq:private:1"


def test_runtime_manage_idle_ignores_attention_before_request_start():
    session = SimpleNamespace(
        sleep_wake_event=None,
        sleep_arming=False,
        sleep_pending_wake=True,
        sleep_pending_wake_at=90.0,
        last_wake_reason="被私聊消息叫醒了",
        sleep_wake_from="qq:private:1",
    )

    reason = asyncio.run(
        runtime_manage.wait_until_attention(
            session,
            0.001,
            pending_wake_after=100.0,
        )
    )

    assert reason == "timeout"
    assert session.sleep_pending_wake is False
    assert session.last_wake_reason == ""
    assert session.sleep_wake_from is None


def test_qq_attention_during_thinking_marks_focus_pending_wake(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        focus = ConversationSession()
        focus.set_conversation_meta("group", "focus", "Focus")
        sessions["qq:group:focus"] = focus
        monkeypatch.setattr(app_state, "current_focus", focus.focus)
        monkeypatch.setattr(qq_handler.time, "time", lambda: 105.0)

        qq_handler._dispatch_wake_signals(
            focus,
            "qq:group:focus",
            True,
            "被@叫醒了",
        )

        assert focus.sleep_pending_wake is True
        assert focus.sleep_pending_wake_at == 105.0
        assert focus.last_wake_reason == "被@叫醒了"
        assert focus.sleep_wake_from == "qq:group:focus"
    finally:
        sessions.clear()
        sessions.update(original_sessions)


def test_non_focus_mention_during_thinking_marks_focus_pending_wake(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        focus = ConversationSession()
        focus.set_conversation_meta("group", "focus", "Focus")
        incoming = ConversationSession()
        incoming.set_conversation_meta("group", "other", "Other")
        sessions["qq:group:focus"] = focus
        monkeypatch.setattr(app_state, "current_focus", focus.focus)
        monkeypatch.setattr(qq_handler.time, "time", lambda: 106.0)

        qq_handler._dispatch_wake_signals(
            incoming,
            "qq:group:other",
            True,
            "被@叫醒了",
        )

        assert focus.sleep_pending_wake is True
        assert focus.sleep_pending_wake_at == 106.0
        assert focus.last_wake_reason == "被@叫醒了"
        assert focus.sleep_wake_from == "qq:group:other"
        assert incoming.sleep_pending_wake is False
    finally:
        sessions.clear()
        sessions.update(original_sessions)

