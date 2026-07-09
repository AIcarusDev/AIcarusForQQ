from __future__ import annotations

import asyncio
import threading
import time
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


def test_runtime_manage_wait_allows_up_to_180_seconds(monkeypatch):
    sleeps: list[float] = []
    times = iter([100.0, 100.0, 280.0])

    monkeypatch.setattr(runtime_manage.time, "time", lambda: next(times))
    monkeypatch.setattr(runtime_manage.time, "sleep", lambda seconds: sleeps.append(seconds))

    result = runtime_manage.execute(
        action="wait",
        seconds=240,
        _request_started_at=100.0,
    )

    assert sleeps == [180.0]
    assert result["requested_seconds"] == 180
    assert result["elapsed_seconds"] == 180


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


def test_runtime_manage_idle_defaults_to_5_minutes(monkeypatch):
    observed: dict[str, float] = {}

    def fake_attention_sleep(session, duration_secs, *, pending_wake_after=None):
        observed["duration_secs"] = duration_secs
        return "timeout"

    monkeypatch.setattr(runtime_manage, "run_coroutine_sync", lambda coro, loop, timeout=None: coro)
    monkeypatch.setattr(runtime_manage, "wait_until_attention", fake_attention_sleep)
    monkeypatch.setattr(runtime_manage.time, "time", lambda: 100.0)

    import app_state
    from llm.session import ConversationSession, sessions
    from platforms.focus import FocusRef

    original_sessions = dict(sessions)
    sessions.clear()
    try:
        focus = ConversationSession()
        focus.set_conversation_meta("group", "focus", "Focus")
        sessions["qq:group:focus"] = focus
        monkeypatch.setattr(app_state, "main_loop", SimpleNamespace(is_running=lambda: True))
        monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", "focus"))

        result = runtime_manage.execute(action="idle", _request_started_at=100.0)

        assert observed["duration_secs"] == 5 * 60
        assert result["requested_seconds"] == 5 * 60
    finally:
        sessions.clear()
        sessions.update(original_sessions)


def test_runtime_manage_sleep_runs_memory_maintenance(monkeypatch):
    observed: dict[str, object] = {}
    order: list[str] = []

    def fake_attention_sleep(session, duration_secs, *, pending_wake_after=None):
        order.append("wait")
        observed["duration_secs"] = duration_secs
        return "timeout"

    def fake_maintenance(action, *, pause_event=None):
        order.append("maintenance")
        observed["maintenance_action"] = action
        observed["pause_event"] = pause_event
        return {"ok": True, "summary_worker": {"summaries_ready": 1}}

    monkeypatch.setattr(runtime_manage, "run_coroutine_sync", lambda coro, loop, timeout=None: coro)
    monkeypatch.setattr(runtime_manage, "wait_until_attention", fake_attention_sleep)
    monkeypatch.setattr(runtime_manage, "schedule_sleep_memory_maintenance_for_runtime", fake_maintenance)
    monkeypatch.setattr(runtime_manage.time, "time", lambda: 100.0)

    original_sessions = dict(sessions)
    sessions.clear()
    try:
        focus = ConversationSession()
        focus.set_conversation_meta("group", "focus", "Focus")
        sessions["qq:group:focus"] = focus
        monkeypatch.setattr(app_state, "main_loop", SimpleNamespace(is_running=lambda: True))
        monkeypatch.setattr(app_state, "current_focus", focus.focus)

        result = runtime_manage.execute(action="sleep", minutes=30, _request_started_at=100.0)

        assert observed["duration_secs"] == 30 * 60
        assert observed["maintenance_action"] == "sleep"
        assert order == ["maintenance", "wait"]
        assert observed["pause_event"].is_set()
        assert result["memory_maintenance"]["ok"] is True
    finally:
        sessions.clear()
        sessions.update(original_sessions)


def test_fallback_sleep_runs_memory_maintenance(monkeypatch):
    import consciousness.main_loop as main_loop

    observed: dict[str, object] = {}

    class FakeFlow:
        def __init__(self):
            self.rounds = []

        def prune(self, max_rounds):
            observed["pruned"] = max_rounds

        def append_round(self, calls, responses):
            self.rounds.append((calls, responses))

    async def fake_attention_sleep(session, duration_secs, *, pending_wake_after=None):
        observed["duration_secs"] = duration_secs
        observed["pending_wake_after"] = pending_wake_after
        observed["sleep_wake_action_during_wait"] = session.sleep_wake_action
        return "timeout"

    def fake_maintenance(action, *, pause_event=None):
        observed["maintenance_action"] = action
        observed["pause_event"] = pause_event
        return {"ok": True, "dry_run": True}

    flow = FakeFlow()
    session = ConversationSession()
    session.set_conversation_meta("group", "focus", "Focus")
    monkeypatch.setattr(app_state, "consciousness_flow", flow)
    monkeypatch.setattr(runtime_manage, "wait_until_attention", fake_attention_sleep)
    monkeypatch.setattr(runtime_manage, "schedule_sleep_memory_maintenance_for_runtime", fake_maintenance)
    monkeypatch.setattr(main_loop._time, "time", lambda: 100.0)
    monkeypatch.setattr(main_loop._time, "monotonic", lambda: 100.0)

    asyncio.run(main_loop._synthesize_fallback_sleep(session, duration=30))

    assert observed["maintenance_action"] == "sleep"
    assert observed["duration_secs"] == 30 * 60
    assert observed["sleep_wake_action_during_wait"] == "sleep"
    assert session.sleep_wake_action == ""
    assert observed["pause_event"].is_set()
    response = flow.rounds[0][1][0].response
    assert response["memory_maintenance"]["ok"] is True
    assert response["memory_maintenance"]["dry_run"] is True


def test_runtime_manage_sleep_memory_maintenance_is_scheduled_without_waiting(monkeypatch, caplog):
    import memory.sleep.sleep_maintenance as sleep_maintenance

    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()

    def slow_maintenance(**_kwargs):
        started.set()
        release.wait(timeout=1.0)
        finished.set()
        return {"ok": True, "summary_worker": {"summaries_ready": 2}}

    monkeypatch.setattr(sleep_maintenance, "run_sleep_memory_maintenance", slow_maintenance)

    with caplog.at_level("INFO", logger="AICQ.tools.runtime_manage"):
        result = runtime_manage.schedule_sleep_memory_maintenance_for_runtime("sleep")
        assert started.wait(timeout=1.0)
        assert not finished.is_set()
        release.set()
        assert finished.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while "sleep 记忆维护完成 ok=True" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)

    assert result["ok"] is True
    assert result["scheduled"] is True
    assert result["completed"] is False
    assert result["reason"] == "memory_maintenance_scheduled"
    assert "sleep 记忆维护完成 ok=True" in caplog.text
    assert "background=True" in caplog.text


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

