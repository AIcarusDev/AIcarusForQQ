from __future__ import annotations

import asyncio

import app_state
from platforms.focus import FocusRef
from runtime.events import RuntimeEventHub
from workspace.models import CommandResult
from workspace.tools import command as command_tool
from workspace.tools import _common
from workspace.errors import WorkspaceError, WorkspaceErrorCode


def command_result(
    status: str,
    *,
    content: str = "",
    content_file: str | None = None,
    content_chars: int | None = None,
) -> CommandResult:
    return CommandResult(
        command_id="a" * 32,
        workspace_id="default",
        status=status,
        cwd="/home/agent",
        exit_code=0 if status == "completed" else None,
        started_at="2026-01-01T00:00:00Z",
        finished_at="2026-01-01T00:00:01Z" if status == "completed" else None,
        timed_out=False,
        cursor=len(content.encode("utf-8")),
        content=content,
        content_file=content_file,
        content_chars=content_chars,
    )


def test_command_result_exposes_spill_metadata_only_when_present() -> None:
    plain = _common.command_result(command_result("completed", content="done\n"))
    assert "content_file" not in plain
    assert "content_chars" not in plain

    spilled = _common.command_result(
        command_result(
            "completed",
            content="head\n[Content too long; truncated]\ntail",
            content_file="/home/agent/.aicq/command-output/abc/0-4096.log",
            content_chars=4096,
        )
    )
    assert spilled["content_file"] == "/home/agent/.aicq/command-output/abc/0-4096.log"
    assert spilled["content_chars"] == 4096


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
        result = await handler(action="run", command="sleep 30", cwd="/home/agent", stdin="")
        assert result["status"] == "running"
        assert result["content"] == "partial\n"

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
        result = await handler(action="run", command="true", cwd="/home/agent", stdin="")
        assert result["status"] == "completed"
        assert result["exit_code"] == 0
        assert result["content"] == "done\n"
        assert await hub.wait(timeout=0) == []

    asyncio.run(scenario())


def test_computer_not_built_uses_stable_nested_tool_error(monkeypatch) -> None:
    async def operation():
        return None

    class Loop:
        def is_running(self):
            return True

    def fail(coro, _loop, timeout=None):
        coro.close()
        raise WorkspaceError(
            WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
            "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
        )

    monkeypatch.setattr(_common, "run_coroutine_sync", fail)
    assert _common.run_on_main_loop(operation(), Loop()) == {
        "ok": False,
        "error": {
            "code": "computer_not_built",
            "message": "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
        },
    }
