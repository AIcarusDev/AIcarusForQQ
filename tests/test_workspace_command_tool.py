from __future__ import annotations

import asyncio

import app_state
from platforms.focus import FocusRef
from runtime.events import RuntimeEventHub
from tools.results import TextPayloadResult
from workspace.models import CommandResult
from workspace.tools import command as command_tool
from workspace.tools import _common
from workspace.errors import WorkspaceError, WorkspaceErrorCode


def command_result(status: str, *, content: str = "") -> CommandResult:
    return CommandResult(
        command_id="a" * 32,
        workspace_id="default",
        status=status,
        cwd="/workspace",
        exit_code=0 if status == "completed" else None,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z" if status == "completed" else None,
        timed_out=False,
        cursor=len(content.encode("utf-8")),
        content=content,
    )


def test_command_run_returns_running_immediately_on_attention(monkeypatch) -> None:
    class Service:
        async def start_command(self, command, *, cwd, stdin):
            return command_result("running")

        async def wait_for_terminal(self, command_id, *, timeout):
            await asyncio.Event().wait()

        async def poll_command(self, command_id, *, cursor=0):
            return command_result("running", content="partial\n")

        async def mark_terminal_delivered(self, command_id):
            raise AssertionError("running result must not be marked terminal")

    async def scenario() -> None:
        hub = RuntimeEventHub()
        monkeypatch.setattr(app_state, "runtime_event_hub", hub)
        monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", "focus"))
        monkeypatch.setattr(command_tool, "run_on_main_loop", lambda coro, _loop: coro)
        await hub.publish(
            {"type": "attention", "reason": "new message", "from": "qq:group:focus"},
            target="qq:group:focus",
        )
        handler = command_tool.make_handler(Service(), hub, object())
        result = await handler(action="run", command="sleep 30", cwd="/workspace", stdin="")
        assert isinstance(result, TextPayloadResult)
        assert result.meta["status"] == "running"
        assert result.text_payload == "partial\n"

    asyncio.run(scenario())


def test_command_run_acknowledges_terminal_event_already_returned_to_model(monkeypatch) -> None:
    class Service:
        async def start_command(self, command, *, cwd, stdin):
            return command_result("running")

        async def wait_for_terminal(self, command_id, *, timeout):
            return command_result("completed", content="done\n")

        async def poll_command(self, command_id, *, cursor=0):
            return command_result("completed", content="done\n")

        async def mark_terminal_delivered(self, command_id):
            assert command_id == "a" * 32

    async def scenario() -> None:
        hub = RuntimeEventHub()
        await hub.publish(
            {
                "type": "workspace_command_finished",
                "command_id": "a" * 32,
                "status": "completed",
                "exit_code": 0,
            }
        )
        monkeypatch.setattr(app_state, "runtime_event_hub", hub)
        monkeypatch.setattr(app_state, "current_focus", FocusRef("qq", "group", "focus"))
        monkeypatch.setattr(command_tool, "run_on_main_loop", lambda coro, _loop: coro)
        handler = command_tool.make_handler(Service(), hub, object())
        result = await handler(action="run", command="true", cwd="/workspace", stdin="")
        assert isinstance(result, TextPayloadResult)
        assert result.meta["status"] == "completed"
        assert result.meta["exit_code"] == 0
        assert await hub.wait(timeout=0) == []

    asyncio.run(scenario())


def test_workspace_not_built_uses_stable_nested_tool_error(monkeypatch) -> None:
    async def operation():
        return None

    class Loop:
        def is_running(self):
            return True

    def fail(coro, _loop, timeout=None):
        coro.close()
        raise WorkspaceError(
            WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
            "工作区不存在或尚未构建，请前往 Web 配置中的“工作区”页面完成构建。",
        )

    monkeypatch.setattr(_common, "run_coroutine_sync", fail)
    assert _common.run_on_main_loop(operation(), Loop()) == {
        "ok": False,
        "error": {
            "code": "workspace_not_built",
            "message": "工作区不存在或尚未构建，请前往 Web 配置中的“工作区”页面完成构建。",
        },
    }
