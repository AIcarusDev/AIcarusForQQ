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


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []
        self.closed = False
        self.active_execs = 0
        self.max_active_execs = 0

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
                "limits": {"cpus": 4, "memory_bytes": 8 * 1024**3, "pids": 1024},
            }
        if method == "exec":
            self.active_execs += 1
            self.max_active_execs = max(self.max_active_execs, self.active_execs)
            await asyncio.sleep(0.01)
            self.active_execs -= 1
            return {
                "command_id": "cmd-1",
                "workspace_id": "default",
                "status": "completed",
                "cwd": params["cwd"],
                "exit_code": 0,
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "timed_out": False,
                "stdout": {"text": "ok\n", "total_bytes": 3, "truncated": False},
                "stderr": {"text": "", "total_bytes": 0, "truncated": False},
            }
        if method == "get_command":
            return {
                "command_id": params["command_id"],
                "workspace_id": "default",
                "status": "completed",
                "cwd": "/workspace",
                "exit_code": 0,
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "timed_out": False,
                "stdout": {"text": "x", "total_bytes": 20 * 1024**2, "truncated": True},
                "stderr": {"text": "", "total_bytes": 0, "truncated": False},
            }
        if method == "read_text":
            return {"content": "hello", "size_bytes": 5}
        if method == "write_text":
            return {"path": params["path"], "size_bytes": len(params["content"].encode())}
        raise AssertionError(method)

    async def close(self) -> None:
        self.closed = True


def test_service_is_lazy_and_health_does_not_ensure() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        assert backend.calls == []
        health = await service.health()
        assert health.protocol_version == 1
        assert [call[0] for call in backend.calls] == ["health"]

    asyncio.run(scenario())


def test_exec_validates_limits_and_serializes_default_workspace() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        results = await asyncio.gather(service.exec("echo one"), service.exec("echo two"))
        assert [result.exit_code for result in results] == [0, 0]
        assert backend.max_active_execs == 1

        with pytest.raises(WorkspaceError) as exc_info:
            await service.exec("x" * (64 * 1024 + 1))
        assert exc_info.value.code is WorkspaceErrorCode.INVALID_ARGUMENT

        with pytest.raises(WorkspaceError):
            await service.exec("true", stdin="x" * (1024 * 1024 + 1))
        with pytest.raises(WorkspaceError):
            await service.exec("true", timeout=901)
        with pytest.raises(WorkspaceError):
            await service.write_text("large.txt", "x" * (1024 * 1024 + 1))
        with pytest.raises(WorkspaceError):
            await service.ensure_default("other")

    asyncio.run(scenario())


def test_text_limits_truncation_metadata_and_close() -> None:
    async def scenario() -> None:
        backend = FakeBackend()
        service = WorkspaceService(backend)
        assert await service.read_text("notes.txt") == "hello"
        result = await service.write_text("notes.txt", "你好")
        assert result["size_bytes"] == 6
        command = await service.get_command("cmd-1")
        assert command.stdout.truncated is True
        assert command.stdout.total_bytes == 20 * 1024**2
        await service.close()
        assert backend.closed is True
        with pytest.raises(WorkspaceError):
            await service.health()

    asyncio.run(scenario())


def test_health_rejects_protocol_mismatch() -> None:
    class MismatchBackend(FakeBackend):
        async def request(self, method, params, *, timeout=None):
            result = dict(await super().request(method, params, timeout=timeout))
            result["protocol_version"] = 2
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
            return (json.dumps(response).encode() + b"\n", b"")

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return self.returncode

    async def fake_create(*args, **kwargs):
        captured["argv"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)

    async def scenario() -> None:
        backend = WslWorkspaceBackend(WorkspaceConfig(wsl_executable="C:/Windows/System32/wsl.exe"))
        hostile = "printf '%s\\n' \"$HOME\"; echo $(whoami); `id`;\nnext"
        result = await backend.request(
            "exec", {"command": hostile, "path": "C:\\Users\\should-not-be-argv"}, timeout=1
        )
        assert result["seen"]["command"] == hostile
        assert captured["argv"] == (
            "C:/Windows/System32/wsl.exe",
            "--distribution",
            "AICQ-Workspace",
            "--user",
            "aicqws",
            "--exec",
            "/usr/local/bin/aicq-workspace-bridge",
        )
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
    assert defaulted.install_root == "C:\\Users\\dev\\AppData\\Local\\AICQ\\Workspace"

    expanded = WorkspaceProvisionConfig.from_root_config(
        {"workspace": {"provisioning": {"install_root": "%LOCALAPPDATA%\\AICQ-Dev"}}},
        environ={"LocalAppData": "D:\\Profiles\\dev\\Local"},
    )
    assert expanded.install_root == "D:\\Profiles\\dev\\Local\\AICQ-Dev"

    with pytest.raises(ValueError, match="absolute local Windows drive"):
        WorkspaceProvisionConfig.from_root_config(
            {"workspace": {"provisioning": {"install_root": "relative\\workspace"}}},
            environ={},
        )


def test_workspace_foundation_is_not_registered_or_exposed() -> None:
    root = Path(__file__).resolve().parents[1]
    modules = (root / "src/tools/modules.yaml").read_text(encoding="utf-8")
    namespaces = (root / "src/tools/namespaces.yaml").read_text(encoding="utf-8")
    prompt_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "src/llm/prompt").rglob("*.py")
    )
    web_settings_sources = "\n".join(
        [
            (root / "src/web/routes_settings.py").read_text(encoding="utf-8"),
            (root / "src/templates/settings.html").read_text(encoding="utf-8"),
        ]
    )
    config_template = (root / "templates/config.yaml.template").read_text(encoding="utf-8")
    provision_script = (root / "scripts/workspace/provision-workspace.ps1").read_text(
        encoding="utf-8"
    )
    assert "workspace" not in modules.lower()
    assert "workspace" not in namespaces.lower()
    assert "workspace" not in prompt_sources.lower()
    assert "workspace" not in web_settings_sources.lower()
    assert "workspace:" in config_template
    assert "install_root:" in config_template
    assert "Assert-NoOtherRunningDistro" not in provision_script
    assert "--shutdown" not in provision_script
    assert "$IsWindows" not in provision_script
    assert "$env:OS" in provision_script
    assert "resolve_install_root.py" in provision_script
    assert not (root / "src/tools/workspace").exists()
    assert not list((root / "src/web").glob("*workspace*"))
