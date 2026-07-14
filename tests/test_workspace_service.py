from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from workspace import (
    PROTOCOL_VERSION,
    WorkspaceConfig,
    WorkspaceError,
    WorkspaceErrorCode,
    WorkspaceProvisionConfig,
    WorkspaceService,
    WslWorkspaceBackend,
)
from workspace.recovery import running_command_ids_from_flow_dump


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []
        self.closed = False
        self.command_done = asyncio.Event()
        self.command_done.set()
        self.revision = "rev-1"

    def command_payload(self, status: str, *, content: str = "") -> dict[str, Any]:
        return {
            "command_id": "a" * 32,
            "workspace_id": "default",
            "status": status,
            "cwd": "/workspace",
            "exit_code": 0 if status == "completed" else None,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z" if status == "completed" else None,
            "timed_out": False,
            "cursor": len(content.encode()),
            "has_more": False,
            "truncated": False,
            "content": content,
        }

    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append((method, dict(params), timeout))
        if method == "health":
            return {
                "protocol_version": PROTOCOL_VERSION,
                "broker_version": "test",
                "distro": "AICQ-Workspace",
                "container_exists": False,
                "container_running": False,
                "image_digest": "sha256:test",
                "firewall_active": True,
            }
        if method == "ensure_default":
            return {
                "workspace_id": "default",
                "container_name": "aicq-workspace-default",
                "created": True,
                "started": True,
                "image_digest": "sha256:test",
                "limits": {},
            }
        if method == "start_command":
            return self.command_payload("running")
        if method == "wait_command":
            await self.command_done.wait()
            return self.command_payload("completed")
        if method == "poll_command":
            return self.command_payload("completed", content="ok\n")
        if method == "stop_command":
            payload = self.command_payload("completed")
            payload["status"] = "stopped"
            payload["exit_code"] = 143
            return payload
        if method == "read_file":
            return {
                "path": "/workspace/notes.txt",
                "content": "1\thello\n2\tworld",
                "revision": self.revision,
                "start_line": params["start_line"],
                "end_line": 2,
                "total_lines": 2,
                "has_more": False,
                "next_line": None,
                "truncated_lines": [],
            }
        if method == "edit_file":
            assert params["expected_revision"] == self.revision
            self.revision = "rev-2"
            return {
                "path": "/workspace/notes.txt",
                "revision": self.revision,
                "replacements": 1,
                "size_bytes": 11,
                "total_lines": 2,
            }
        if method == "write_file":
            self.revision = "rev-3"
            return {
                "path": "/workspace/notes.txt",
                "revision": self.revision,
                "created": params.get("expected_revision") is None,
                "size_bytes": len(str(params["content"]).encode()),
                "total_lines": 1,
            }
        if method in {"find_files", "search"}:
            return {
                "path": "/workspace",
                "content": "/workspace/notes.txt",
                "count": 1,
                "offset": 0,
                "next_offset": None,
                "has_more": False,
                "truncated": False,
            }
        raise AssertionError(method)

    async def close(self) -> None:
        self.closed = True


def test_service_is_lazy_and_health_does_not_ensure() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        assert backend.calls == []
        health = await service.health()
        assert health.protocol_version == 2
        assert [call[0] for call in backend.calls] == ["health"]
        await service.close()

    asyncio.run(scenario())


def test_command_job_contract_has_no_model_timeout_and_polls_content() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        started = await service.start_command("printf ok")
        assert started.status == "running"
        start_call = next(call for call in backend.calls if call[0] == "start_command")
        assert "timeout_seconds" not in start_call[1]
        assert "background" not in start_call[1]
        completed = await service.wait_for_terminal(started.command_id, timeout=1)
        assert completed is not None and completed.terminal
        page = await service.poll_command(started.command_id)
        assert page.content == "ok\n"
        assert page.cursor == 3
        await service.close()

    asyncio.run(scenario())


def test_service_validates_command_limits() -> None:
    async def scenario() -> None:
        service = WorkspaceService(FakeBackend())
        with pytest.raises(WorkspaceError) as exc_info:
            await service.start_command("x" * (64 * 1024 + 1))
        assert exc_info.value.code is WorkspaceErrorCode.INVALID_ARGUMENT
        with pytest.raises(WorkspaceError):
            await service.start_command("true", stdin="x" * (1024 * 1024 + 1))
        with pytest.raises(WorkspaceError):
            await service.poll_command("a" * 32, cursor=-1)
        with pytest.raises(WorkspaceError):
            await service.ensure_default("other")
        with pytest.raises(WorkspaceError) as windows_path:
            await service.start_command("true", cwd="C:\\host")
        assert windows_path.value.code is WorkspaceErrorCode.INVALID_ARGUMENT
        with pytest.raises(WorkspaceError) as invalid_utf8:
            await service.write_file("bad.txt", "\ud800")
        assert invalid_utf8.value.code is WorkspaceErrorCode.INVALID_ARGUMENT
        await service.close()

    asyncio.run(scenario())


def test_read_revision_guards_edit_and_full_overwrite() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        with pytest.raises(WorkspaceError) as unread:
            await service.edit_file("notes.txt", [{"old_text": "hello", "new_text": "hi"}])
        assert unread.value.code is WorkspaceErrorCode.FILE_NOT_READ

        read = await service.read_file("notes.txt")
        assert read.content.startswith("1\thello")
        edited = await service.edit_file("notes.txt", [{"old_text": "hello", "new_text": "hi"}])
        assert edited["replacements"] == 1
        written = await service.write_file("notes.txt", "complete")
        assert written["revision"] == "rev-3"
        await service.close()

    asyncio.run(scenario())


def test_find_and_search_return_paginated_text() -> None:
    async def scenario() -> None:
        service = WorkspaceService(FakeBackend())
        found = await service.find_files("**/*.txt")
        searched = await service.search("hello", literal=True)
        assert found.content == "/workspace/notes.txt"
        assert searched.count == 1
        await service.close()

    asyncio.run(scenario())


def test_health_rejects_protocol_mismatch() -> None:
    class MismatchBackend(FakeBackend):
        async def request(self, method, params, *, timeout=None):
            result = dict(await super().request(method, params, timeout=timeout))
            result["protocol_version"] = 999
            return result

    async def scenario() -> None:
        with pytest.raises(WorkspaceError) as exc_info:
            await WorkspaceService(MismatchBackend()).health()
        assert exc_info.value.code is WorkspaceErrorCode.PROTOCOL_MISMATCH

    asyncio.run(scenario())


def test_wsl_backend_uses_fixed_argv_and_json_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self, payload: bytes):
            captured["payload"] = payload
            request = json.loads(payload)
            response = {
                "version": PROTOCOL_VERSION,
                "request_id": request["request_id"],
                "ok": True,
                "result": {"seen": request["params"]},
            }
            return json.dumps(response).encode() + b"\n", b""

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def fake_create(*args, **kwargs):
        captured["argv"] = args
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    async def scenario() -> None:
        backend = WslWorkspaceBackend(WorkspaceConfig(wsl_executable="C:/Windows/System32/wsl.exe"))
        hostile = "echo $(whoami); `id`"
        result = await backend.request("start_command", {"command": hostile}, timeout=1)
        assert result["seen"]["command"] == hostile
        assert hostile not in captured["argv"]
        assert b"$(whoami)" in captured["payload"]

    asyncio.run(scenario())


def test_workspace_provision_config_resolves_user_and_default_paths() -> None:
    configured = WorkspaceProvisionConfig.from_root_config(
        {"workspace": {"provisioning": {"install_root": "E:\\Aic_forQ\\wsl"}}},
        environ={"LOCALAPPDATA": "C:\\Users\\dev\\AppData\\Local"},
    )
    assert configured.install_root == "E:\\Aic_forQ\\wsl"
    defaulted = WorkspaceProvisionConfig.from_root_config(
        {"workspace": {"provisioning": {"install_root": ""}}},
        environ={"LOCALAPPDATA": "C:\\Users\\dev\\AppData\\Local"},
    )
    assert defaulted.install_root.endswith("AICQ\\Workspace")
    with pytest.raises(ValueError, match="absolute local Windows drive"):
        WorkspaceProvisionConfig.from_root_config(
            {"workspace": {"provisioning": {"install_root": "relative\\workspace"}}},
            environ={},
        )


def test_workspace_namespace_and_protocol_v2_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    modules = (root / "src/tools/modules.yaml").read_text(encoding="utf-8")
    namespaces = (root / "src/tools/namespaces.yaml").read_text(encoding="utf-8")
    manifest = json.loads(
        (root / "scripts/workspace/appliance/opt/aicq-workspace/protocol-manifest.json").read_text(encoding="utf-8")
    )
    assert "workspace:" in modules
    assert "always_active: true" in modules
    assert "import_path: workspace.tools" in namespaces
    assert "permanent: false" in namespaces
    assert manifest["protocol_version"] == 2
    assert manifest["image_name"].endswith(":2")
    broker = (root / "scripts/workspace/appliance/opt/aicq-workspace/broker.py").read_text(encoding="utf-8")
    assert '"--workdir",\n                "/workspace"' in broker
    assert 'str(record["cwd"])' not in broker


def test_command_terminal_callback_runs_once_and_callback_failure_does_not_hide_result() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        seen: list[str] = []

        async def callback(result) -> None:
            seen.append(result.command_id)
            raise RuntimeError("event sink unavailable")

        service.set_terminal_callback(callback)
        started = await service.start_command("true")
        first = await service.wait_for_terminal(started.command_id, timeout=1)
        second = await service.wait_for_terminal(started.command_id, timeout=1)
        assert first is not None and first.status == "completed"
        assert second is first
        assert seen == [started.command_id]
        await service.close()

    asyncio.run(scenario())


def test_terminal_delivery_suppresses_late_completion_wake_callback() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        backend.command_done.clear()
        service = WorkspaceService(backend)
        seen: list[str] = []
        service.set_terminal_callback(lambda result: seen.append(result.command_id))

        started = await service.start_command("true")
        await service.mark_terminal_delivered(started.command_id)
        backend.command_done.set()
        terminal = await service.wait_for_terminal(started.command_id, timeout=1)
        assert terminal is not None and terminal.status == "completed"
        assert seen == []
        await service.close()

    asyncio.run(scenario())


def test_command_monitor_can_retry_after_transient_wait_failure() -> None:
    class RetryBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.wait_attempts = 0

        async def request(self, method, params, *, timeout=None):
            if method == "wait_command":
                self.calls.append((method, dict(params), timeout))
                self.wait_attempts += 1
                if self.wait_attempts == 1:
                    raise RuntimeError("temporary bridge failure")
                return self.command_payload("completed")
            return await super().request(method, params, timeout=timeout)

    async def scenario() -> None:
        backend = RetryBackend()
        service = WorkspaceService(backend)
        started = await service.start_command("true")
        with pytest.raises(RuntimeError, match="temporary bridge failure"):
            await service.wait_for_terminal(started.command_id, timeout=1)
        completed = await service.wait_for_terminal(started.command_id, timeout=1)
        assert completed is not None and completed.status == "completed"
        assert backend.wait_attempts == 2
        await service.close()

    asyncio.run(scenario())


def test_running_commands_are_recovered_from_latest_workspace_result() -> None:
    running = "a" * 32
    finished = "b" * 32
    entries = [
        {
            "responses": [
                {
                    "namespace": "workspace",
                    "name": "command",
                    "response": {"command_id": running, "status": "running"},
                },
                {
                    "namespace": "workspace",
                    "name": "command",
                    "response": {"command_id": finished, "status": "running"},
                },
            ]
        },
        {
            "responses": [
                {
                    "namespace": "workspace",
                    "name": "command",
                    "response": {"command_id": finished, "status": "completed"},
                },
                {"namespace": "other", "name": "command", "response": {"command_id": "ignored", "status": "running"}},
            ]
        },
    ]

    assert running_command_ids_from_flow_dump(entries) == (running,)


def test_write_requires_untruncated_full_file_read() -> None:
    class Backend(FakeBackend):
        async def request(self, method, params, *, timeout=None):
            if method == "read_file":
                return {
                    "path": "/workspace/long.txt",
                    "content": "1\tpartial… [line truncated]",
                    "revision": "rev-long",
                    "start_line": 1,
                    "end_line": 1,
                    "total_lines": 1,
                    "has_more": False,
                    "next_line": None,
                    "truncated_lines": [1],
                }
            return await super().request(method, params, timeout=timeout)

    async def scenario() -> None:
        service = WorkspaceService(Backend())
        await service.read_file("long.txt")
        with pytest.raises(WorkspaceError) as exc_info:
            await service.write_file("long.txt", "replacement")
        assert exc_info.value.code == WorkspaceErrorCode.FILE_NOT_READ
        await service.close()

    asyncio.run(scenario())
