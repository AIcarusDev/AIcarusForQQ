from __future__ import annotations

import asyncio
from types import SimpleNamespace

import app_state
from platforms.focus import FocusRef
from runtime.events import RuntimeEventHub
from tools.core import runtime_manage


def test_runtime_event_hub_delivers_global_and_targeted_events_once() -> None:
    async def scenario() -> None:
        hub = RuntimeEventHub()
        await hub.publish({"type": "workspace_command_finished", "command_id": "a" * 32})
        await hub.publish({"type": "attention", "reason": "other"}, target="qq:group:other")

        events = await hub.wait(timeout=0, target="qq:group:focus")
        assert events == [{"type": "workspace_command_finished", "command_id": "a" * 32}]
        assert await hub.wait(timeout=0, target="qq:group:focus") == []
        assert await hub.wait(timeout=0, target="qq:group:other") == [
            {"type": "attention", "reason": "other"}
        ]

    asyncio.run(scenario())


def test_runtime_event_hub_acknowledges_terminal_event_before_wait() -> None:
    async def scenario() -> None:
        hub = RuntimeEventHub()
        command_id = "b" * 32
        await hub.publish(
            {"type": "workspace_command_finished", "command_id": command_id, "status": "completed"}
        )
        assert await hub.acknowledge(
            event_type="workspace_command_finished",
            key="command_id",
            value=command_id,
        ) == 1
        assert await hub.wait(timeout=0) == []

    asyncio.run(scenario())


def test_runtime_event_hub_can_observe_without_consuming() -> None:
    async def scenario() -> None:
        hub = RuntimeEventHub()
        await hub.publish({"type": "attention"}, target="qq:private:42")
        assert await hub.wait(
            timeout=0, target="qq:private:42", event_types={"attention"}, consume=False
        ) == [{"type": "attention"}]
        assert await hub.wait(
            timeout=0, target="qq:private:42", event_types={"attention"}
        ) == [{"type": "attention"}]
        assert await hub.wait(timeout=0, target="qq:private:42") == []

    asyncio.run(scenario())


def test_runtime_wait_consumes_completion_that_arrived_before_zero_remaining(monkeypatch) -> None:
    async def scenario() -> None:
        hub = RuntimeEventHub()
        command_id = "c" * 32
        await hub.publish(
            {"type": "workspace_command_finished", "command_id": command_id, "status": "completed", "exit_code": 0}
        )
        monkeypatch.setattr(app_state, "runtime_event_hub", hub)
        monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", "focus"))
        session = SimpleNamespace(
            sleep_wake_event=None,
            sleep_arming=False,
            sleep_pending_wake=False,
            sleep_pending_wake_at=0.0,
            last_wake_reason="",
            sleep_wake_from=None,
            conv_type="group",
            conv_id="focus",
            conv_name="Focus",
        )

        reason = await runtime_manage.wait_until_attention(session, 0, pending_wake_after=100.0)
        result = runtime_manage.build_runtime_result(
            session,
            action="wait",
            requested_seconds=10,
            waited_seconds=0,
            elapsed_since_request=10,
            reason=reason,
        )
        assert reason == "woken"
        assert result["events"] == [
            {
                "type": "workspace_command_finished",
                "command_id": command_id,
                "status": "completed",
                "exit_code": 0,
            }
        ]

    asyncio.run(scenario())
