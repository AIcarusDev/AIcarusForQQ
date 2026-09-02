from __future__ import annotations

import asyncio
import importlib.util
import io
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from workspace import (
    CommandResult,
    PROTOCOL_VERSION,
    WorkspaceConfig,
    WorkspaceError,
    WorkspaceErrorCode,
    WorkspaceProvisionConfig,
    WorkspaceService,
    WslWorkspaceBackend,
)
from workspace.recovery import running_command_ids_from_flow_dump


@pytest.fixture(autouse=True)
def isolate_service_unit_tests_from_a_live_control_job(monkeypatch) -> None:
    """Fake-backend unit tests must not depend on this machine's worker lock."""

    monkeypatch.setattr("workspace.control.workspace_control_busy", lambda: False)


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
            "cwd": "/home/agent",
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
                "path": "/home/agent/notes.txt",
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
                "path": "/home/agent/notes.txt",
                "revision": self.revision,
                "replacements": 1,
                "size_bytes": 11,
                "total_lines": 2,
            }
        if method == "write_file":
            self.revision = "rev-3"
            return {
                "path": "/home/agent/notes.txt",
                "revision": self.revision,
                "created": params.get("expected_revision") is None,
                "size_bytes": len(str(params["content"]).encode()),
                "total_lines": 1,
            }
        if method in {"find_files", "search"}:
            return {
                "path": "/home/agent",
                "content": "/home/agent/notes.txt",
                "count": 1,
                "offset": 0,
                "next_offset": None,
                "has_more": False,
                "truncated": False,
            }
        raise AssertionError(method)

    async def close(self) -> None:
        self.closed = True


def _load_appliance_broker(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    broker_path = root / "scripts/workspace/appliance/opt/aicq-workspace/broker.py"
    manifest = json.loads(
        (root / "scripts/workspace/appliance/opt/aicq-workspace/protocol-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    original_path_open = Path.open

    def open_with_manifest(path, *args, **kwargs):
        if path.as_posix() == "/opt/aicq-workspace/protocol-manifest.json":
            return io.StringIO(json.dumps(manifest))
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_manifest)
    spec = importlib.util.spec_from_file_location("aicq_workspace_broker_test", broker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_service_is_lazy_and_health_does_not_ensure() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        assert backend.calls == []
        health = await service.health()
        assert health.protocol_version == 5
        assert [call[0] for call in backend.calls] == ["health"]
        await service.close()

    asyncio.run(scenario())


def test_service_rejects_new_calls_while_workspace_control_is_busy(monkeypatch) -> None:
    monkeypatch.setattr("workspace.control.workspace_control_busy", lambda: True)

    async def scenario() -> None:
        with pytest.raises(WorkspaceError) as exc_info:
            await WorkspaceService(FakeBackend()).health()
        assert exc_info.value.code is WorkspaceErrorCode.WORKSPACE_BUSY

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
        wait_call = next(call for call in backend.calls if call[0] == "wait_command")
        assert wait_call[2] is None
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
        assert found.content == "/home/agent/notes.txt"
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


def test_wsl_backend_retries_a_cold_broker_socket_for_every_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []

    class FakeProcess:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt
            self.returncode = 69 if attempt < 3 else 0

        async def communicate(self, payload: bytes):
            if self.returncode:
                return b"", b"computer broker unavailable: [Errno 2] No such file or directory\n"
            request = json.loads(payload)
            response = {
                "version": PROTOCOL_VERSION,
                "request_id": request["request_id"],
                "ok": True,
                "result": {"status": "running"},
            }
            return json.dumps(response).encode() + b"\n", b""

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode

    async def fake_create(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return FakeProcess(attempts)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def scenario() -> None:
        backend = WslWorkspaceBackend()
        result = await backend.request("start_command", {"command": "true"}, timeout=1)
        assert result == {"status": "running"}
        await backend.close()

    asyncio.run(scenario())
    assert attempts == 3
    assert delays == [0.5, 1.0]


def test_wsl_backend_does_not_retry_an_ambiguous_broker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    class FakeProcess:
        returncode = 69

        async def communicate(self, payload: bytes):
            return b"", b"computer broker unavailable: [Errno 104] Connection reset by peer\n"

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode

    async def fake_create(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    async def scenario() -> None:
        backend = WslWorkspaceBackend()
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.request("start_command", {"command": "true"}, timeout=1)
        assert exc_info.value.code is WorkspaceErrorCode.BROKER_UNAVAILABLE
        await backend.close()

    asyncio.run(scenario())
    assert attempts == 1


def test_wsl_backend_streams_binary_export_without_putting_path_in_argv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    content = b"\x00binary\xff\ncontent"

    class FakeInput:
        def write(self, payload: bytes) -> None:
            captured["payload"] = payload

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            return None

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.stdin = FakeInput()
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(content)
            self.stdout.feed_eof()
            self.stderr = asyncio.StreamReader()
            self.stderr.feed_eof()

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    async def fake_create(*args, **kwargs):
        captured["argv"] = args
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    async def scenario() -> None:
        backend = WslWorkspaceBackend(WorkspaceConfig(wsl_executable="C:/Windows/System32/wsl.exe"))
        destination = tmp_path / "payload.bin"
        size = await backend.export_file(("reports", "测试.bin"), destination, timeout=None)

        assert size == len(content)
        assert destination.read_bytes() == content
        assert "测试.bin" not in captured["argv"]
        assert json.loads(captured["payload"]) == {"parts": ["reports", "测试.bin"]}
        await backend.close()

    asyncio.run(scenario())


def test_stage_host_file_has_no_transfer_deadline(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_request(self, method, params, *, timeout=None):
        assert method == "ensure_default"
        return {
            "workspace_id": "default",
            "container_name": "aicq-workspace-default",
            "created": False,
            "started": True,
            "image_digest": "sha256:test",
            "limits": {},
        }

    async def fake_export(self, relative_parts, destination, *, timeout=120.0):
        captured["parts"] = tuple(relative_parts)
        captured["timeout"] = timeout
        destination.write_bytes(b"large-file-placeholder")
        return destination.stat().st_size

    monkeypatch.setattr("workspace.control.require_workspace_runtime_ready", lambda: None)
    monkeypatch.setattr(WslWorkspaceBackend, "request", fake_request)
    monkeypatch.setattr(WslWorkspaceBackend, "export_file", fake_export)

    async def scenario() -> None:
        backend = WslWorkspaceBackend()
        service = WorkspaceService(backend)
        async with service.stage_host_file(
            "/home/agent/reports/large.bin",
            staging_root=tmp_path,
        ) as staged:
            assert Path(staged.host_path).read_bytes() == b"large-file-placeholder"
            assert Path(staged.host_path).parent.parent == tmp_path.resolve()
        assert list(tmp_path.iterdir()) == []
        await service.close()

    asyncio.run(scenario())
    assert captured == {"parts": ("reports", "large.bin"), "timeout": None}


def test_generic_atomic_import_and_qq_compatibility_entrypoints(monkeypatch) -> None:
    calls: list[tuple[str, tuple[str, ...], int]] = []
    generic_session = object()
    qq_session = object()

    async def fake_generic(relative_parts, expected_size):
        calls.append(("generic", tuple(relative_parts), expected_size))
        return generic_session

    async def fake_qq(relative_parts, expected_size):
        calls.append(("qq", tuple(relative_parts), expected_size))
        return qq_session

    async def fake_ensure_default(*_args, **_kwargs):
        return None

    monkeypatch.setattr("workspace.control.require_workspace_runtime_ready", lambda: None)

    async def scenario() -> None:
        backend = WslWorkspaceBackend()
        service = WorkspaceService(backend)
        monkeypatch.setattr(backend, "begin_file_import", fake_generic)
        monkeypatch.setattr(backend, "begin_qq_file_import", fake_qq)
        monkeypatch.setattr(service, "ensure_default", fake_ensure_default)

        assert await service.begin_file_import("/home/agent/media/image.png", 17) is generic_session
        assert await service.begin_qq_file_import("/home/agent/qq/file.bin", 23) is qq_session
        assert calls == [
            ("generic", ("media", "image.png"), 17),
            ("qq", ("qq", "file.bin"), 23),
        ]

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
    assert defaulted.install_root.endswith("data\\computer")
    relative = WorkspaceProvisionConfig.from_root_config(
        {"workspace": {"install_root": "relative\\computer"}},
        environ={},
    )
    assert relative.install_root.endswith("relative\\computer")
    with pytest.raises(ValueError):
        WorkspaceProvisionConfig.from_root_config(
            {"workspace": {"install_root": "E:\\"}},
            environ={},
        )


def test_broker_command_page_caps_model_content_and_spills_exact_text(monkeypatch, tmp_path) -> None:
    broker = _load_appliance_broker(monkeypatch)
    command_id = "a" * 32
    broker.COMMAND_ROOT = tmp_path / "commands"
    broker.HOME_ROOT = tmp_path / "home"
    directory = broker.COMMAND_ROOT / command_id
    directory.mkdir(parents=True)
    full_content = "开" * 1200 + "🙂" * 1200
    (directory / "merged.bin").write_bytes(full_content.encode("utf-8"))
    record = {
        "command_id": command_id,
        "workspace_id": "default",
        "status": "completed",
        "cwd": "/home/agent",
        "exit_code": 0,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:01Z",
        "truncated": False,
    }

    page = broker.command_page(record, 0)

    assert len(page["content"]) == 2000
    assert page["content"].startswith("开")
    assert page["content"].endswith("🙂")
    assert page["cursor"] == len(full_content.encode("utf-8"))
    assert page["has_more"] is False
    assert page["content_chars"] == len(full_content)
    assert page["note"]
    assert page["content_file"] == (
        f"/home/agent/.aicq/command-output/{command_id}/0-{page['cursor']}.log"
    )
    spill_path = (
        broker.HOME_ROOT
        / ".aicq"
        / "command-output"
        / command_id
        / f"0-{page['cursor']}.log"
    )
    assert spill_path.read_text(encoding="utf-8") == full_content
    assert broker.command_page(record, 0)["content_file"] == page["content_file"]
    typed_page = CommandResult.from_payload(page)
    assert typed_page.content_file == page["content_file"]
    assert typed_page.content_chars == len(full_content)
    assert typed_page.note == page["note"]

    boundary_content = "🙂" * 2000
    (directory / "merged.bin").write_bytes(boundary_content.encode("utf-8"))
    boundary_page = broker.command_page(record, 0)
    assert boundary_page["content"] == boundary_content
    assert "content_file" not in boundary_page
    assert "content_chars" not in boundary_page
    assert "note" not in boundary_page


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
        with pytest.raises(RuntimeError):
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
                    "namespace": "computer",
                    "name": "command",
                    "response": {"command_id": running, "status": "running"},
                },
                {
                    "namespace": "computer",
                    "name": "command",
                    "response": {"command_id": finished, "status": "running"},
                },
            ]
        },
        {
            "responses": [
                {
                    "namespace": "computer",
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
                    "path": "/home/agent/long.txt",
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


def test_oversized_read_does_not_authorize_overwrite() -> None:
    class Backend(FakeBackend):
        async def request(self, method, params, *, timeout=None):
            if method == "read_file":
                raise WorkspaceError(
                    WorkspaceErrorCode.CONTENT_TOO_LARGE,
                    "Content too large: retry with a smaller line range.",
                )
            return await super().request(method, params, timeout=timeout)

    async def scenario() -> None:
        service = WorkspaceService(Backend())
        with pytest.raises(WorkspaceError) as read_error:
            await service.read_file("long.txt")
        assert read_error.value.code == WorkspaceErrorCode.CONTENT_TOO_LARGE

        with pytest.raises(WorkspaceError) as edit_error:
            await service.edit_file(
                "long.txt",
                [{"old_text": "before", "new_text": "after"}],
            )
        assert edit_error.value.code == WorkspaceErrorCode.FILE_NOT_READ
        await service.close()

    asyncio.run(scenario())
