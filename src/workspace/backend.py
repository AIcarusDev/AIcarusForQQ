"""Async backend protocol and the Windows-to-WSL stdio implementation."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .config import (
    DEFAULT_CONTAINER_NAME,
    PREVIEW_CONTAINER_PORT,
    PREVIEW_URL_PATH,
    PROTOCOL_VERSION,
    WorkspaceConfig,
)
from .errors import WorkspaceError, WorkspaceErrorCode


_EXPORT_SCRIPT = r"""
import json
import os
import stat
import sys

ROOT = "/var/lib/aicq-workspace/home"


def fail(message):
    print(f"computer export failed: {message}", file=sys.stderr)
    raise SystemExit(66)


try:
    request = json.loads(sys.stdin.buffer.readline().decode("utf-8"))
    parts = request.get("parts") if isinstance(request, dict) else None
    if (
        not isinstance(parts, list)
        or not parts
        or any(not isinstance(part, str) or not part or part in {".", ".."} or "/" in part for part in parts)
    ):
        fail("invalid Agent-home-relative path")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(ROOT, directory_flags)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                fail("computer path is not a regular file")
            while True:
                chunk = os.read(file_fd, 1024 * 1024)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)
except SystemExit:
    raise
except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
    fail(str(exc))
""".strip()


async def _bounded_stderr(reader: asyncio.StreamReader, limit: int = 64 * 1024) -> bytes:
    kept = bytearray()
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            return bytes(kept)
        remaining = limit - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])


class WorkspaceBackend(Protocol):
    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]: ...

    async def preview(self, *, timeout: float = 15.0) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class WslWorkspaceBackend:
    """Send one RPC through one fixed ``wsl.exe`` bridge process.

    No user value is ever part of the process argv. Commands, paths, and text
    travel only in a JSON line written to stdin.
    """

    def __init__(self, config: WorkspaceConfig | None = None) -> None:
        self.config = config or WorkspaceConfig()
        self._processes: set[asyncio.subprocess.Process] = set()
        self._closed = False
        self._state_lock = asyncio.Lock()

    def _argv(self) -> tuple[str, ...]:
        cfg = self.config
        return (
            cfg.wsl_executable,
            "--distribution",
            cfg.distro_name,
            "--user",
            cfg.appliance_user,
            "--exec",
            cfg.bridge_path,
        )

    def _export_argv(self) -> tuple[str, ...]:
        cfg = self.config
        return (
            cfg.wsl_executable,
            "--distribution",
            cfg.distro_name,
            "--user",
            cfg.appliance_user,
            "--exec",
            "/usr/bin/python3",
            "-I",
            "-c",
            _EXPORT_SCRIPT,
        )

    def _preview_argv(self) -> tuple[str, ...]:
        cfg = self.config
        return (
            cfg.wsl_executable,
            "--distribution",
            cfg.distro_name,
            "--user",
            cfg.appliance_user,
            "--exec",
            "/usr/bin/env",
            "XDG_RUNTIME_DIR=/run/user/1000",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus",
            "/usr/bin/podman",
            "port",
            DEFAULT_CONTAINER_NAME,
            f"{PREVIEW_CONTAINER_PORT}/tcp",
        )

    async def preview(self, *, timeout: float = 15.0) -> Mapping[str, Any]:
        """Return the one provisioned loopback preview endpoint.

        The argv is fully fixed: callers cannot select a container, container
        port, host address, or host port.
        """

        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")
        request_id = uuid.uuid4().hex
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._preview_argv(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
                "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
                details={"transport_error": str(exc)},
                request_id=request_id,
            ) from exc

        async with self._state_lock:
            if self._closed:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    "computer backend closed while reading preview endpoint",
                    request_id=request_id,
                )
            self._processes.add(proc)

        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    max(0.1, float(timeout)) + self.config.bridge_grace_seconds,
                )
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.PREVIEW_UNAVAILABLE,
                    "Agent 电脑浏览器投射端口查询超时。",
                    request_id=request_id,
                ) from exc
            except asyncio.CancelledError:
                proc.kill()
                await proc.wait()
                raise
        finally:
            async with self._state_lock:
                self._processes.discard(proc)

        if proc.returncode != 0:
            diagnostic = stderr.decode("utf-8", errors="replace").strip()[-4096:]
            raise WorkspaceError(
                WorkspaceErrorCode.PREVIEW_UNAVAILABLE,
                "Agent 电脑浏览器投射尚未就绪，请前往 Web 配置中的“Agent 电脑”页面原地应用设置。",
                details={"returncode": proc.returncode, "diagnostic": diagnostic},
                request_id=request_id,
            )

        endpoint = stdout.decode("utf-8", errors="replace").strip()
        match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", endpoint)
        port = int(match.group(1)) if match else 0
        if not match or not 1 <= port <= 65535:
            raise WorkspaceError(
                WorkspaceErrorCode.PREVIEW_UNAVAILABLE,
                "Agent 电脑返回了无效或非本机回环的浏览器投射地址。",
                details={"endpoint": endpoint[:256]},
                request_id=request_id,
            )
        return {
            "url": f"http://127.0.0.1:{port}{PREVIEW_URL_PATH}",
            "host": "127.0.0.1",
            "port": port,
            "container_port": PREVIEW_CONTAINER_PORT,
        }

    async def export_file(
        self,
        relative_parts: Sequence[str],
        destination: Path,
        *,
        timeout: float = 120.0,
    ) -> int:
        """Stream one Agent-home file into a Windows-local staging path.

        The Linux path is sent over stdin, never interpolated into a shell or
        process argv.  The fixed exporter opens every component with
        ``O_NOFOLLOW`` so home symlinks cannot escape into the appliance.
        """

        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")
        parts = [str(part) for part in relative_parts]
        payload = (json.dumps({"parts": parts}, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        request_id = uuid.uuid4().hex
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._export_argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
                "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
                details={"transport_error": str(exc)},
                request_id=request_id,
            ) from exc

        async with self._state_lock:
            if self._closed:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    "computer backend closed while opening file exporter",
                    request_id=request_id,
                )
            self._processes.add(proc)

        async def transfer() -> tuple[int, bytes, int]:
            assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
            proc.stdin.write(payload)
            await proc.stdin.drain()
            proc.stdin.close()
            stderr_task = asyncio.create_task(_bounded_stderr(proc.stderr))
            size = 0
            try:
                with destination.open("wb") as output:
                    while True:
                        chunk = await proc.stdout.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        size += len(chunk)
                    output.flush()
                return await proc.wait(), await stderr_task, size
            finally:
                if not stderr_task.done():
                    stderr_task.cancel()
                    await asyncio.gather(stderr_task, return_exceptions=True)

        try:
            try:
                returncode, stderr, size = await asyncio.wait_for(
                    transfer(),
                    max(0.1, float(timeout)) + self.config.bridge_grace_seconds,
                )
            except asyncio.TimeoutError as exc:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    "computer file export did not finish before its deadline",
                    request_id=request_id,
                ) from exc
            except asyncio.CancelledError:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
                raise
            except OSError as exc:
                if proc.returncode is None:
                    proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    f"Windows staging file could not be written: {exc}",
                    request_id=request_id,
                ) from exc
        finally:
            async with self._state_lock:
                self._processes.discard(proc)

        if returncode != 0:
            destination.unlink(missing_ok=True)
            diagnostic = stderr.decode("utf-8", errors="replace").strip()[-4096:]
            lowered = diagnostic.casefold()
            if any(
                marker in lowered
                for marker in (
                    "there is no distribution",
                    "distribution was not found",
                    "找不到具有所提供名称的分发",
                    "wsl_e_distro_not_found",
                )
            ):
                code = WorkspaceErrorCode.WORKSPACE_NOT_BUILT
                message = "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。"
            elif returncode == 66:
                code = WorkspaceErrorCode.PATH_ERROR
                message = diagnostic or "computer file could not be exported"
            else:
                code = WorkspaceErrorCode.BROKER_UNAVAILABLE
                message = diagnostic or f"WSL file exporter exited with code {returncode}"
            raise WorkspaceError(
                code,
                message,
                details={"returncode": returncode},
                request_id=request_id,
            )
        return size

    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")

        request_id = uuid.uuid4().hex
        envelope = {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "method": str(method),
            "params": dict(params),
        }
        payload = (json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *self._argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
                "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
                details={"transport_error": str(exc)},
                request_id=request_id,
            ) from exc

        async with self._state_lock:
            if self._closed:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    "computer backend closed while opening bridge",
                    request_id=request_id,
                )
            self._processes.add(proc)

        transport_timeout = None
        if timeout is not None:
            transport_timeout = max(0.1, float(timeout)) + self.config.bridge_grace_seconds

        try:
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(payload), transport_timeout)
            except asyncio.TimeoutError as exc:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    "WSL bridge did not return before its transport deadline",
                    details={"method": method},
                    request_id=request_id,
                ) from exc
            except asyncio.CancelledError:
                proc.kill()
                await proc.wait()
                raise
        finally:
            async with self._state_lock:
                self._processes.discard(proc)

        if proc.returncode != 0:
            diagnostic = stderr.decode("utf-8", errors="replace").strip()[-4096:]
            lowered = diagnostic.casefold()
            if any(
                marker in lowered
                for marker in (
                    "there is no distribution",
                    "distribution was not found",
                    "找不到具有所提供名称的分发",
                    "wsl_e_distro_not_found",
                )
            ):
                code = WorkspaceErrorCode.WORKSPACE_NOT_BUILT
                message = "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。"
            elif "protocol" in lowered or "bridge" in lowered and "not found" in lowered:
                code = WorkspaceErrorCode.WORKSPACE_NEEDS_UPGRADE
                message = "Agent 电脑系统与当前程序不兼容，请前往 Web 配置中的“Agent 电脑”页面更新系统。"
            else:
                code = WorkspaceErrorCode.BROKER_UNAVAILABLE
                message = diagnostic or f"WSL bridge exited with code {proc.returncode}"
            raise WorkspaceError(
                code,
                message,
                details={"returncode": proc.returncode},
                request_id=request_id,
            )

        lines = stdout.splitlines()
        if len(lines) != 1:
            raise WorkspaceError(
                WorkspaceErrorCode.PROTOCOL_MISMATCH,
                "bridge response must contain exactly one NDJSON record",
                details={"line_count": len(lines)},
                request_id=request_id,
            )
        try:
            response = json.loads(lines[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.PROTOCOL_MISMATCH,
                "bridge returned invalid UTF-8 JSON",
                request_id=request_id,
            ) from exc

        if not isinstance(response, dict):
            raise WorkspaceError(
                WorkspaceErrorCode.PROTOCOL_MISMATCH,
                "bridge response is not an object",
                request_id=request_id,
            )
        if response.get("version") != PROTOCOL_VERSION or response.get("request_id") != request_id:
            raise WorkspaceError(
                WorkspaceErrorCode.WORKSPACE_NEEDS_UPGRADE,
                "Agent 电脑系统与当前程序不兼容，请前往 Web 配置中的“Agent 电脑”页面更新系统。",
                details={
                    "expected_version": PROTOCOL_VERSION,
                    "received_version": response.get("version"),
                },
                request_id=request_id,
            )
        if response.get("ok") is not True:
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise WorkspaceError(
                str(error.get("code", WorkspaceErrorCode.INTERNAL_ERROR.value)),
                str(error.get("message", "computer broker request failed")),
                details=error.get("details") if isinstance(error.get("details"), dict) else {},
                request_id=request_id,
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise WorkspaceError(
                WorkspaceErrorCode.PROTOCOL_MISMATCH,
                "successful bridge response has no object result",
                request_id=request_id,
            )
        return result

    async def close(self) -> None:
        async with self._state_lock:
            if self._closed:
                return
            self._closed = True
            processes = list(self._processes)
        for proc in processes:
            if proc.returncode is None:
                proc.kill()
        if processes:
            await asyncio.gather(*(proc.wait() for proc in processes), return_exceptions=True)
        async with self._state_lock:
            self._processes.clear()
