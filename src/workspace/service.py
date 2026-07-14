"""Async-first, internal-only workspace service."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from .backend import WorkspaceBackend
from .config import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_WORKSPACE_ID,
    MAX_COMMAND_BYTES,
    MAX_COMMAND_TIMEOUT_SECONDS,
    MAX_STDIN_BYTES,
    MAX_TEXT_BYTES,
    PROTOCOL_VERSION,
)
from .errors import WorkspaceError, WorkspaceErrorCode
from .models import CommandResult, EnsureResult, HealthResult


def _utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def _require_default(workspace_id: str) -> str:
    value = str(workspace_id or "").strip()
    if value != DEFAULT_WORKSPACE_ID:
        raise WorkspaceError(
            WorkspaceErrorCode.INVALID_ARGUMENT,
            "only workspace_id='default' is supported in phase one",
        )
    return value


class WorkspaceService:
    """Validated application boundary around a workspace backend.

    The service owns no automatic startup task. ``health`` is parallel-safe;
    stateful default-workspace operations are serialized.
    """

    def __init__(self, backend: WorkspaceBackend) -> None:
        self._backend = backend
        self._workspace_lock = asyncio.Lock()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "workspace service is closed")

    async def health(self) -> HealthResult:
        self._require_open()
        result = await self._backend.request("health", {}, timeout=15.0)
        health = HealthResult.from_payload(result)
        if health.protocol_version != PROTOCOL_VERSION:
            raise WorkspaceError(
                WorkspaceErrorCode.PROTOCOL_MISMATCH,
                "broker health response uses an unsupported protocol",
                details={"received_version": health.protocol_version},
            )
        return health

    async def ensure_default(self, workspace_id: str = DEFAULT_WORKSPACE_ID) -> EnsureResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        async with self._workspace_lock:
            result = await self._backend.request(
                "ensure_default",
                {"workspace_id": workspace_id},
                timeout=MAX_COMMAND_TIMEOUT_SECONDS,
            )
        return EnsureResult.from_payload(result)

    async def exec(
        self,
        command: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        cwd: str = "/workspace",
        stdin: str = "",
        timeout: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> CommandResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        if not isinstance(command, str) or not command:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "command must be a non-empty string")
        if _utf8_size(command) > MAX_COMMAND_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "command exceeds the 64 KiB limit")
        if not isinstance(stdin, str) or _utf8_size(stdin) > MAX_STDIN_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "stdin exceeds the 1 MiB limit")
        if not isinstance(cwd, str) or not cwd:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "cwd must be a non-empty Linux path")
        timeout = float(timeout)
        if timeout <= 0 or timeout > MAX_COMMAND_TIMEOUT_SECONDS:
            raise WorkspaceError(
                WorkspaceErrorCode.INVALID_ARGUMENT,
                "timeout must be greater than zero and at most 900 seconds",
            )
        params = {
            "workspace_id": workspace_id,
            "command": command,
            "cwd": cwd,
            "stdin": stdin,
            "timeout_seconds": timeout,
        }
        async with self._workspace_lock:
            result = await self._backend.request("exec", params, timeout=timeout)
        return CommandResult.from_payload(result)

    async def get_command(
        self,
        command_id: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> CommandResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        if not isinstance(command_id, str) or not command_id.strip():
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "command_id is required")
        result = await self._backend.request(
            "get_command",
            {"workspace_id": workspace_id, "command_id": command_id.strip()},
            timeout=15.0,
        )
        return CommandResult.from_payload(result)

    async def read_text(
        self,
        path: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> str:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        if not isinstance(path, str) or not path:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "path is required")
        async with self._workspace_lock:
            result = await self._backend.request(
                "read_text", {"workspace_id": workspace_id, "path": path}, timeout=30.0
            )
        content = result.get("content")
        if not isinstance(content, str) or _utf8_size(content) > MAX_TEXT_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.PROTOCOL_MISMATCH, "invalid read_text result")
        return content

    async def write_text(
        self,
        path: str,
        content: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        create_parents: bool = False,
    ) -> Mapping[str, Any]:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        if not isinstance(path, str) or not path:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "path is required")
        if not isinstance(content, str) or _utf8_size(content) > MAX_TEXT_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "content exceeds the 1 MiB limit")
        async with self._workspace_lock:
            return await self._backend.request(
                "write_text",
                {
                    "workspace_id": workspace_id,
                    "path": path,
                    "content": content,
                    "create_parents": bool(create_parents),
                },
                timeout=30.0,
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._backend.close()
