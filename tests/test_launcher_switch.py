import asyncio

import app_state
from web import routes_core


def test_launcher_start_switch_uses_short_force_exit_delay(monkeypatch):
    monkeypatch.delenv("AICQ_LAUNCHER_START_FORCE_EXIT_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("AICQ_LAUNCHER_STOP_FORCE_EXIT_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("AICQ_LAUNCHER_SWITCH_FORCE_EXIT_AFTER_SECONDS", raising=False)

    assert (
        routes_core._launcher_switch_force_exit_after_seconds(
            routes_core.LAUNCHER_START_CORE_EXIT_CODE
        )
        == 1.0
    )
    assert (
        routes_core._launcher_switch_force_exit_after_seconds(
            routes_core.LAUNCHER_STOP_CORE_EXIT_CODE
        )
        == 30.0
    )


def test_launcher_switch_delay_env_can_override_per_action(monkeypatch):
    monkeypatch.setenv("AICQ_LAUNCHER_START_FORCE_EXIT_AFTER_SECONDS", "2.5")
    monkeypatch.setenv("AICQ_LAUNCHER_STOP_FORCE_EXIT_AFTER_SECONDS", "12")

    assert (
        routes_core._launcher_switch_force_exit_after_seconds(
            routes_core.LAUNCHER_START_CORE_EXIT_CODE
        )
        == 2.5
    )
    assert (
        routes_core._launcher_switch_force_exit_after_seconds(
            routes_core.LAUNCHER_STOP_CORE_EXIT_CODE
        )
        == 12.0
    )


def test_launcher_switch_is_scheduled_after_response_delay(monkeypatch):
    triggered: list[int] = []

    monkeypatch.setattr(routes_core, "_LAUNCHER_SWITCH_RESPONSE_DELAY_SECONDS", 0.01)
    monkeypatch.setattr(
        routes_core,
        "_trigger_launcher_switch",
        lambda exit_code: triggered.append(exit_code),
    )

    async def _exercise():
        routes_core._schedule_launcher_switch(routes_core.LAUNCHER_START_CORE_EXIT_CODE)
        assert triggered == []
        await asyncio.sleep(0.05)

    asyncio.run(_exercise())

    assert triggered == [routes_core.LAUNCHER_START_CORE_EXIT_CODE]


def test_launcher_switch_watchdog_forces_expected_exit_code(monkeypatch):
    exit_codes: list[int] = []

    monkeypatch.setattr(routes_core.time, "sleep", lambda _delay: None)

    def _fake_hard_exit(exit_code: int):
        exit_codes.append(exit_code)

    monkeypatch.setattr(routes_core, "_hard_exit", _fake_hard_exit)
    app_state.launcher_switch_requested = True
    app_state.core_restart_exit_code = routes_core.LAUNCHER_START_CORE_EXIT_CODE

    try:
        thread = routes_core._arm_launcher_switch_force_exit(
            routes_core.LAUNCHER_START_CORE_EXIT_CODE
        )
        thread.join(timeout=1.0)
    finally:
        app_state.launcher_switch_requested = False
        app_state.core_restart_exit_code = None

    assert exit_codes == [routes_core.LAUNCHER_START_CORE_EXIT_CODE]


def test_launcher_switch_watchdog_ignores_cleared_request(monkeypatch):
    exit_codes: list[int] = []

    monkeypatch.setattr(routes_core.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        routes_core,
        "_hard_exit",
        lambda exit_code: exit_codes.append(exit_code),
    )
    app_state.launcher_switch_requested = False
    app_state.core_restart_exit_code = routes_core.LAUNCHER_START_CORE_EXIT_CODE

    try:
        thread = routes_core._arm_launcher_switch_force_exit(
            routes_core.LAUNCHER_START_CORE_EXIT_CODE
        )
        thread.join(timeout=1.0)
    finally:
        app_state.launcher_switch_requested = False
        app_state.core_restart_exit_code = None

    assert exit_codes == []
