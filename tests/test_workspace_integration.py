from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from workspace import CommandResult, WorkspaceService, WslWorkspaceBackend


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("AICQ_WORKSPACE_INTEGRATION") != "1",
        reason="set AICQ_WORKSPACE_INTEGRATION=1 after WebUI build/apply installs the user-managed appliance",
    ),
]


def run(coro):
    return asyncio.run(coro)


async def run_command(
    service: WorkspaceService,
    command: str,
    *,
    cwd: str = "/workspace",
) -> tuple[CommandResult, str, bool]:
    started = await service.start_command(command, cwd=cwd)
    terminal = await service.wait_for_terminal(started.command_id, timeout=905.0)
    assert terminal is not None
    cursor = 0
    chunks: list[str] = []
    truncated = False
    while True:
        page = await service.poll_command(started.command_id, cursor=cursor)
        chunks.append(page.content)
        truncated = truncated or page.truncated
        assert page.cursor >= cursor
        cursor = page.cursor
        if not page.has_more:
            return page, "".join(chunks), truncated


def test_workspace_bash_root_text_io_and_command_paging() -> None:
    async def scenario() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            health = await service.health()
            assert health.protocol_version == 2
            assert health.firewall_active is True
            ensured = await service.ensure_default()
            assert ensured.container_name == "aicq-workspace-default"

            result, output, _ = await run_command(
                service,
                "test \"$(id -u)\" = 0 && printf 'alpha\\nbeta\\n' | grep beta | tr a-z A-Z",
            )
            assert result.exit_code == 0
            assert output == "BETA\n"

            written = await service.write_file("integration/state.txt", "persistent-状态\n", create_parents=True)
            read = await service.read_file(str(written["path"]))
            assert read.content == "1\tpersistent-状态"

            result, output, truncated = await run_command(service, "python -c \"print('x' * 70000)\"")
            assert result.exit_code == 0
            assert len(output.encode("utf-8")) == 70001
            assert truncated is False

            started = await service.start_command("sleep 30")
            assert await service.wait_for_terminal(started.command_id, timeout=0.1) is None
            running = await service.poll_command(started.command_id)
            assert running.status == "running"
            stopped = await service.stop_command(started.command_id)
            assert stopped.status == "stopped"
        finally:
            await service.close()

    run(scenario())


def test_workspace_development_stack_and_isolation() -> None:
    async def scenario() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            await service.ensure_default()
            result, output, _ = await run_command(
                service,
                "set -e; "
                "python -m pip install --disable-pip-version-check --quiet packaging; "
                "python -c 'import packaging'; "
                "printf '#include <stdio.h>\\nint main(void){puts(\"ok\");}\\n' > /tmp/a.c; "
                "gcc /tmp/a.c -o /tmp/a && test \"$(/tmp/a)\" = ok; "
                "rm -rf /tmp/hello && git clone -q --depth 1 https://github.com/octocat/Hello-World.git /tmp/hello; "
                "test ! -e /mnt/c; ! command -v cmd.exe; "
                "test ! -S /run/podman/podman.sock; "
                "test ! -S /run/user/1000/podman/podman.sock; test ! -e /dev/dxg",
            )
            assert result.exit_code == 0, output
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
            result, _, _ = await run_command(
                service,
                "printf persisted > /workspace/integration-restart.txt; "
                "printf '#!/bin/sh\\nprintf rootfs-persisted\\n' > /usr/local/bin/aicq-persist; "
                "chmod +x /usr/local/bin/aicq-persist",
            )
            assert result.exit_code == 0
            command_id = result.command_id
        finally:
            await service.close()

    async def read_marker() -> None:
        service = WorkspaceService(WslWorkspaceBackend())
        try:
            await service.ensure_default()
            marker = await service.read_file("integration-restart.txt")
            assert marker.content == "1\tpersisted"
            result, output, _ = await run_command(service, "aicq-persist")
            assert result.exit_code == 0
            assert output == "rootfs-persisted"
            persisted = await service.poll_command(command_id)
            assert persisted.exit_code == 0
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
            limits, output, _ = await run_command(
                service,
                "set -e; "
                f"test \"$(cat /sys/fs/cgroup/memory.max)\" = {memory_limit}; "
                "test \"$(cat /sys/fs/cgroup/pids.max)\" = 1024; "
                "test \"$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max)\" = 400000; "
                "curl -fsS --max-time 20 https://example.com/ >/dev/null",
            )
            assert limits.exit_code == 0, output
            private, output, _ = await run_command(
                service,
                "! curl -fsS --connect-timeout 2 --max-time 4 http://169.254.169.254/; "
                "! curl -fsS --connect-timeout 2 --max-time 4 http://192.168.0.1/",
            )
            assert private.exit_code == 0, output
        finally:
            await service.close()

    run(scenario())
