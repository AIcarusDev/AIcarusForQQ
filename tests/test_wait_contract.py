from __future__ import annotations

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


def test_runtime_manage_repair_maps_legacy_time_fields():
    repaired, changes = runtime_manage.repair_schema_args(
        {"action": "WAIT", "timeout": "3"}
    )

    assert repaired == {"action": "wait", "seconds": 3}
    assert changes == ["action: normalized", "timeout -> seconds", "seconds: string -> int"]

    repaired, changes = runtime_manage.repair_schema_args(
        {"action": "sleep", "duration": "45"}
    )

    assert repaired == {"action": "sleep", "minutes": 45}
    assert changes == ["duration -> minutes", "minutes: string -> int"]


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
