from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from workspace import WorkspaceError, WorkspaceErrorCode, WorkspaceService, WslWorkspaceBackend


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("AICQ_WORKSPACE_INTEGRATION") != "1",
        reason="set AICQ_WORKSPACE_INTEGRATION=1 after provisioning the appliance",
    ),
]


def run(coro):
    return asyncio.run(coro)


def test_workspace_bash_root_and_text_io() -> None:
    async def scenario() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            health = await service.health()
            assert health.protocol_version == 1
            assert health.firewall_active is True
            ensured = await service.ensure_default()
            assert ensured.container_name == "aicq-workspace-default"
            result = await service.exec(
                "test \"$(id -u)\" = 0 && printf 'alpha\\nbeta\\n' | grep beta | tr a-z A-Z"
            )
            assert result.exit_code == 0
            assert result.stdout.text == "BETA\n"
            await service.write_text("integration/state.txt", "persistent-状态\n", create_parents=True)
            assert await service.read_text("integration/state.txt") == "persistent-状态\n"
            escaped_limit = "\x00" * (1024 * 1024)
            await service.write_text("integration/escaped-limit.txt", escaped_limit)
            assert await service.read_text("integration/escaped-limit.txt") == escaped_limit

            output = await service.exec("python -c \"print('x' * 70000)\"")
            assert output.stdout.total_bytes == 70001
            assert output.stdout.truncated is True
            assert len(output.stdout.text.encode("utf-8")) <= 64 * 1024

            with pytest.raises(WorkspaceError) as exc_info:
                await service.exec("sleep 2", timeout=0.1)
            timeout_error = exc_info.value
            assert timeout_error.code is WorkspaceErrorCode.COMMAND_TIMEOUT
            timed_out = await service.get_command(timeout_error.details["command_id"])
            assert timed_out.timed_out is True
        finally:
            await service.close()

    run(scenario())


def test_workspace_development_stack_and_isolation() -> None:
    async def scenario() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            await service.ensure_default()
            result = await service.exec(
                "set -e; "
                "python -m pip install --disable-pip-version-check --quiet packaging; "
                "python -c 'import packaging'; "
                "printf '#include <stdio.h>\\nint main(void){puts(\"ok\");}\\n' > /tmp/a.c; "
                "gcc /tmp/a.c -o /tmp/a && test \"$(/tmp/a)\" = ok; "
                "rm -rf /tmp/hello && git clone -q --depth 1 https://github.com/octocat/Hello-World.git /tmp/hello; "
                "test ! -e /mnt/c; ! command -v cmd.exe; "
                "test ! -S /run/podman/podman.sock; "
                "test ! -S /run/user/1000/podman/podman.sock; test ! -e /dev/dxg",
                timeout=600,
            )
            assert result.exit_code == 0, result.stderr.text
        finally:
            await service.close()

    run(scenario())


def test_workspace_persists_across_wsl_termination() -> None:
    command_id = ""

    async def write_marker() -> None:
        nonlocal command_id
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            await service.ensure_default()
            result = await service.exec(
                "printf persisted > /workspace/integration-restart.txt; "
                "printf '#!/bin/sh\\nprintf rootfs-persisted\\n' > /usr/local/bin/aicq-persist; "
                "chmod +x /usr/local/bin/aicq-persist"
            )
            assert result.exit_code == 0
            command_id = result.command_id
        finally:
            await service.close()

    async def read_marker() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            await service.ensure_default()
            assert await service.read_text("integration-restart.txt") == "persisted"
            result = await service.exec("aicq-persist")
            assert result.stdout.text == "rootfs-persisted"
            persisted_command = await service.get_command(command_id)
            assert persisted_command.exit_code == 0
        finally:
            await service.close()

    run(write_marker())
    subprocess.run(
        [
            "wsl.exe",
            "--distribution",
            "AICQ-Workspace",
            "--user",
            "aicqws",
            "--exec",
            "/usr/bin/env",
            "XDG_RUNTIME_DIR=/run/user/1000",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
            "/usr/bin/podman",
            "restart",
            "aicq-workspace-default",
        ],
        check=True,
        timeout=60,
    )
    subprocess.run(["wsl.exe", "--terminate", "AICQ-Workspace"], check=True, timeout=30)
    run(read_marker())


def test_workspace_resource_limits_and_public_egress_policy() -> None:
    async def scenario() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            await service.ensure_default()
            memory_limit = 8 * 1024**3
            limits = await service.exec(
                "set -e; "
                f"test \"$(cat /sys/fs/cgroup/memory.max)\" = {memory_limit}; "
                "test \"$(cat /sys/fs/cgroup/pids.max)\" = 1024; "
                "test \"$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)\" = 400000; "
                "curl -fsS --max-time 20 https://example.com/ >/dev/null"
            )
            assert limits.exit_code == 0, limits.stderr.text
            private = await service.exec(
                "! curl -fsS --connect-timeout 2 --max-time 4 http://169.254.169.254/; "
                "! curl -fsS --connect-timeout 2 --max-time 4 http://192.168.0.1/"
            )
            assert private.exit_code == 0
        finally:
            await service.close()

    run(scenario())
