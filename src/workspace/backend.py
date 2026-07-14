"""Async backend protocol and the Windows-to-WSL stdio implementation."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Mapping, Protocol

from .config import PROTOCOL_VERSION, WorkspaceConfig
from .errors import WorkspaceError, WorkspaceErrorCode


class WorkspaceBackend(Protocol):
    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]: ...

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

    async def request(
        self,
        method: str,
        params: Mapping[str, Any],
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "workspace backend is closed")

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
                WorkspaceErrorCode.DISTRO_UNAVAILABLE,
                f"could not start WSL bridge: {exc}",
                request_id=request_id,
            ) from exc

        async with self._state_lock:
            if self._closed:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(
                    WorkspaceErrorCode.BROKER_UNAVAILABLE,
                    "workspace backend closed while opening bridge",
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
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                diagnostic or f"WSL bridge exited with code {proc.returncode}",
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
                WorkspaceErrorCode.PROTOCOL_MISMATCH,
                "bridge protocol version or request id did not match",
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
                str(error.get("message", "workspace broker request failed")),
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
