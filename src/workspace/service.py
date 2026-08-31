"""Async-first application boundary for the Agent's isolated Linux computer."""

from __future__ import annotations

import asyncio
import inspect
import logging
import posixpath
import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator

from .backend import WorkspaceBackend
from .config import (
    COMMAND_OBSERVATION_SECONDS,
    DEFAULT_AGENT_HOME,
    DEFAULT_WORKSPACE_ID,
    ENSURE_TIMEOUT_SECONDS,
    MAX_COMMAND_BYTES,
    MAX_STDIN_BYTES,
    MAX_TEXT_BYTES,
    PROTOCOL_VERSION,
)
from .errors import WorkspaceError, WorkspaceErrorCode
from .models import CommandResult, EnsureResult, FileReadResult, HealthResult, TextListResult


TerminalCallback = Callable[[CommandResult], Awaitable[None] | None]
logger = logging.getLogger("AICQ.workspace")


def _utf8_size(value: str, name: str = "text") -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise WorkspaceError(
            WorkspaceErrorCode.INVALID_ARGUMENT,
            f"{name} must be valid UTF-8 text",
        ) from exc


def _require_default(workspace_id: str) -> str:
    value = str(workspace_id or "").strip()
    if value != DEFAULT_WORKSPACE_ID:
        raise WorkspaceError(
            WorkspaceErrorCode.INVALID_ARGUMENT,
            "only the default Agent computer is supported",
        )
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, f"{name} is required")
    _utf8_size(value, name)
    return value


def _linux_path(value: Any, name: str) -> str:
    raw = _require_text(value, name)
    if "\\" in raw or re.match(r"^[A-Za-z]:", raw):
        raise WorkspaceError(
            WorkspaceErrorCode.INVALID_ARGUMENT,
            f"{name} must be a Linux path; Windows and host paths are not accepted",
        )
    return posixpath.normpath(raw if raw.startswith("/") else posixpath.join(DEFAULT_AGENT_HOME, raw))


def _agent_home_file_parts(path: str) -> tuple[str, tuple[str, ...]]:
    normalized = _linux_path(path, "path")
    try:
        relative = PurePosixPath(normalized).relative_to(DEFAULT_AGENT_HOME)
    except ValueError as exc:
        raise WorkspaceError(
            WorkspaceErrorCode.INVALID_ARGUMENT,
            f"path must be inside {DEFAULT_AGENT_HOME}",
        ) from exc
    if not relative.parts:
        raise WorkspaceError(
            WorkspaceErrorCode.INVALID_ARGUMENT,
            f"path must identify a file inside {DEFAULT_AGENT_HOME}",
        )
    return normalized, relative.parts


@dataclass(frozen=True, slots=True)
class WorkspaceHostFile:
    workspace_path: str
    host_path: str
    name: str
    size: int


@dataclass(slots=True)
class _ReadState:
    revision: str
    total_lines: int
    intervals: list[tuple[int, int]] = field(default_factory=list)
    truncated_lines: set[int] = field(default_factory=set)

    def add(self, start: int, end: int) -> None:
        if end < start:
            if self.total_lines == 0:
                self.intervals = [(1, 0)]
            return
        merged: list[tuple[int, int]] = []
        for left, right in sorted([*self.intervals, (start, end)]):
            if not merged or left > merged[-1][1] + 1:
                merged.append((left, right))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        self.intervals = merged

    @property
    def fully_read(self) -> bool:
        if self.total_lines == 0:
            return True
        return bool(
            not self.truncated_lines
            and self.intervals
            and self.intervals[0][0] <= 1
            and self.intervals[0][1] >= self.total_lines
        )


class WorkspaceService:
    """Validated service around the persistent Agent-computer broker."""

    def __init__(self, backend: WorkspaceBackend) -> None:
        self._backend = backend
        self._ensure_lock = asyncio.Lock()
        self._read_state_lock = asyncio.Lock()
        self._read_states: dict[str, _ReadState] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._terminal_futures: dict[str, asyncio.Future[CommandResult]] = {}
        self._terminal_callback: TerminalCallback | None = None
        self._terminal_delivery_lock = asyncio.Lock()
        self._terminal_delivered: set[str] = set()
        self._staging_lock = asyncio.Lock()
        self._staged_directories: set[Path] = set()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer service is closed")
        from .control import workspace_control_busy

        if workspace_control_busy():
            raise WorkspaceError(
                WorkspaceErrorCode.WORKSPACE_BUSY,
                "Agent 电脑正在执行安装、更新或维护操作，请稍后重试。",
            )
        # Only the real WSL backend needs the host-side ownership/version gate.
        # Test/in-process backends remain side-effect free and independently usable.
        from .backend import WslWorkspaceBackend

        if isinstance(self._backend, WslWorkspaceBackend):
            from .control import require_workspace_runtime_ready

            require_workspace_runtime_ready()

    def set_terminal_callback(self, callback: TerminalCallback | None) -> None:
        self._terminal_callback = callback

    async def mark_terminal_delivered(self, command_id: str) -> None:
        """Suppress a future wake event after a terminal result reached the model."""

        command_id = _require_text(command_id, "command_id").strip()
        async with self._terminal_delivery_lock:
            self._terminal_delivered.add(command_id)

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
        async with self._ensure_lock:
            result = await self._backend.request(
                "ensure_default",
                {"workspace_id": workspace_id},
                timeout=ENSURE_TIMEOUT_SECONDS,
            )
        return EnsureResult.from_payload(result)

    @asynccontextmanager
    async def stage_host_file(
        self,
        path: str,
        *,
        staging_root: str | Path | None = None,
    ) -> AsyncIterator[WorkspaceHostFile]:
        """Export one Agent-home file to Windows for the lifetime of the context."""

        self._require_open()
        normalized, relative_parts = _agent_home_file_parts(path)
        backend = self._backend
        from .backend import WslWorkspaceBackend

        if not isinstance(backend, WslWorkspaceBackend):
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "computer backend cannot expose files to the host",
            )
        await self.ensure_default()
        if staging_root is None:
            staging_directory = Path(tempfile.mkdtemp(prefix="aicq-workspace-send-"))
        else:
            root = Path(staging_root).expanduser().resolve()
            await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
            staging_directory = Path(
                tempfile.mkdtemp(prefix="aicq-workspace-send-", dir=str(root))
            )
        host_path = staging_directory / "payload.bin"
        try:
            size = await backend.export_file(relative_parts, host_path, timeout=None)
            prepared = WorkspaceHostFile(
                workspace_path=normalized,
                host_path=str(host_path),
                name=PurePosixPath(normalized).name,
                size=size,
            )
            async with self._staging_lock:
                if self._closed:
                    raise WorkspaceError(
                        WorkspaceErrorCode.BROKER_UNAVAILABLE,
                        "computer service closed while staging a file",
                    )
                self._staged_directories.add(staging_directory)
            try:
                yield prepared
            finally:
                async with self._staging_lock:
                    self._staged_directories.discard(staging_directory)
        finally:
            await asyncio.to_thread(shutil.rmtree, staging_directory, True)

    async def import_host_file(self, path: str, host_path: str | Path) -> Mapping[str, Any]:
        """Atomically place one host-local file at an Agent-home path."""

        self._require_open()
        _normalized, relative_parts = _agent_home_file_parts(path)
        from .backend import WslWorkspaceBackend

        if not isinstance(self._backend, WslWorkspaceBackend):
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "computer backend cannot import host files",
            )
        await self.ensure_default()
        return await self._backend.import_file(relative_parts, Path(host_path), timeout=None)

    async def begin_qq_file_import(self, path: str, expected_size: int):
        """Open a direct binary stream into one Agent-home QQ file."""

        self._require_open()
        _normalized, relative_parts = _agent_home_file_parts(path)
        from .backend import WslWorkspaceBackend

        if not isinstance(self._backend, WslWorkspaceBackend):
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "computer backend cannot import QQ file streams",
            )
        await self.ensure_default()
        return await self._backend.begin_qq_file_import(relative_parts, expected_size)

    async def qq_file_operation(self, action: str, path: str) -> Mapping[str, Any]:
        """Run a fixed metadata/list/delete operation for one Agent-home path."""

        self._require_open()
        normalized = _linux_path(path, "path")
        pure = PurePosixPath(normalized)
        if pure == PurePosixPath(DEFAULT_AGENT_HOME):
            relative_parts: tuple[str, ...] = ()
        else:
            try:
                relative = pure.relative_to(PurePosixPath(DEFAULT_AGENT_HOME))
            except ValueError as exc:
                raise WorkspaceError(
                    WorkspaceErrorCode.INVALID_ARGUMENT,
                    "path must be inside /home/agent",
                ) from exc
            relative_parts = tuple(relative.parts)
        from .backend import WslWorkspaceBackend

        if not isinstance(self._backend, WslWorkspaceBackend):
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "computer backend cannot manage host files",
            )
        await self.ensure_default()
        return await self._backend.qq_file_operation(action, relative_parts)

    async def start_command(
        self,
        command: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
        cwd: str = DEFAULT_AGENT_HOME,
        stdin: str = "",
    ) -> CommandResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        if not isinstance(command, str) or not command or "\x00" in command:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "command must be a non-empty string")
        if _utf8_size(command, "command") > MAX_COMMAND_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "command exceeds the 64 KiB limit")
        if not isinstance(stdin, str) or _utf8_size(stdin, "stdin") > MAX_STDIN_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "stdin exceeds the 1 MiB limit")
        cwd = _linux_path(cwd, "cwd")
        result = CommandResult.from_payload(
            await self._backend.request(
                "start_command",
                {
                    "workspace_id": workspace_id,
                    "command": command,
                    "cwd": cwd,
                    "stdin": stdin,
                },
                timeout=60.0,
            )
        )
        self._ensure_monitor(result.command_id)
        return result

    def _ensure_monitor(self, command_id: str) -> asyncio.Future[CommandResult]:
        loop = asyncio.get_running_loop()
        future = self._terminal_futures.get(command_id)
        failed = bool(
            future is not None
            and not future.cancelled()
            and future.done()
            and future.exception() is not None
        )
        if future is None or future.cancelled() or failed:
            future = loop.create_future()
            self._terminal_futures[command_id] = future
        task = self._monitor_tasks.get(command_id)
        if task is None or task.done():
            self._monitor_tasks[command_id] = loop.create_task(
                self._monitor_command(command_id),
                name=f"workspace-command-{command_id[:8]}",
            )
        return future

    async def _monitor_command(self, command_id: str) -> None:
        future = self._terminal_futures[command_id]
        try:
            payload = await self._backend.request(
                "wait_command",
                {"workspace_id": DEFAULT_WORKSPACE_ID, "command_id": command_id},
                timeout=None,
            )
            result = CommandResult.from_payload(payload)
            async with self._terminal_delivery_lock:
                callback = self._terminal_callback
                if callback is not None and command_id not in self._terminal_delivered:
                    try:
                        callback_result = callback(result)
                        if inspect.isawaitable(callback_result):
                            await callback_result
                    except Exception:
                        logger.warning(
                            "[computer] command terminal callback failed: %s",
                            command_id,
                            exc_info=True,
                        )
            if not future.done():
                future.set_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
                # Background monitors may have no current waiter (for example
                # after run returned early on attention). Mark the exception as
                # observed while preserving normal await semantics.
                future.exception()
        finally:
            self._monitor_tasks.pop(command_id, None)

    async def wait_for_terminal(
        self,
        command_id: str,
        *,
        timeout: float = COMMAND_OBSERVATION_SECONDS,
    ) -> CommandResult | None:
        self._require_open()
        command_id = _require_text(command_id, "command_id").strip()
        future = self._ensure_monitor(command_id)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=max(0.0, float(timeout)))
        except asyncio.TimeoutError:
            return None

    async def resume_command_monitor(self, command_id: str) -> None:
        self._require_open()
        self._ensure_monitor(_require_text(command_id, "command_id").strip())

    async def poll_command(
        self,
        command_id: str,
        *,
        cursor: int = 0,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> CommandResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        command_id = _require_text(command_id, "command_id").strip()
        if int(cursor) < 0:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "cursor must be >= 0")
        result = CommandResult.from_payload(
            await self._backend.request(
                "poll_command",
                {"workspace_id": workspace_id, "command_id": command_id, "cursor": int(cursor)},
                timeout=15.0,
            )
        )
        if not result.terminal:
            self._ensure_monitor(command_id)
        return result

    async def stop_command(
        self,
        command_id: str,
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> CommandResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        command_id = _require_text(command_id, "command_id").strip()
        result = CommandResult.from_payload(
            await self._backend.request(
                "stop_command",
                {"workspace_id": workspace_id, "command_id": command_id},
                timeout=30.0,
            )
        )
        return result

    async def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        line_count: int = 2000,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> FileReadResult:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        path = _linux_path(path, "path")
        result = FileReadResult.from_payload(
            await self._backend.request(
                "read_file",
                {
                    "workspace_id": workspace_id,
                    "path": path,
                    "start_line": int(start_line),
                    "line_count": int(line_count),
                },
                timeout=30.0,
            )
        )
        async with self._read_state_lock:
            state = self._read_states.get(result.path)
            if state is None or state.revision != result.revision:
                state = _ReadState(result.revision, result.total_lines)
                self._read_states[result.path] = state
            state.total_lines = result.total_lines
            state.add(result.start_line, result.end_line)
            state.truncated_lines.update(result.truncated_lines)
        return result

    async def edit_file(
        self,
        path: str,
        edits: Sequence[Mapping[str, Any]],
        *,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> Mapping[str, Any]:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        path = _linux_path(path, "path")
        normalized_edits = [dict(edit) for edit in edits]
        if not normalized_edits:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "edits must not be empty")
        if sum(
            _utf8_size(str(edit.get("old_text", "")), "old_text")
            + _utf8_size(str(edit.get("new_text", "")), "new_text")
            for edit in normalized_edits
        ) > MAX_TEXT_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "edit payload exceeds the 1 MiB limit")
        async with self._read_state_lock:
            candidates = [(known_path, state) for known_path, state in self._read_states.items() if known_path == path]
            if not candidates:
                raise WorkspaceError(WorkspaceErrorCode.FILE_NOT_READ, "read the file before editing it")
            known_path, state = candidates[0]
            expected_revision = state.revision
        result = dict(
            await self._backend.request(
                "edit_file",
                {
                    "workspace_id": workspace_id,
                    "path": known_path,
                    "expected_revision": expected_revision,
                    "edits": normalized_edits,
                },
                timeout=30.0,
            )
        )
        async with self._read_state_lock:
            refreshed = self._read_states.pop(known_path, state)
            refreshed.revision = str(result.get("revision", ""))
            refreshed.total_lines = int(result.get("total_lines", refreshed.total_lines) or 0)
            self._read_states[str(result.get("path") or known_path)] = refreshed
        return result

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        create_parents: bool = False,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> Mapping[str, Any]:
        self._require_open()
        workspace_id = _require_default(workspace_id)
        path = _linux_path(path, "path")
        if not isinstance(content, str) or "\x00" in content:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "content must be UTF-8 text without NUL")
        if _utf8_size(content, "content") > MAX_TEXT_BYTES:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "content exceeds the 1 MiB limit")
        async with self._read_state_lock:
            known_path = path
            state = self._read_states.get(known_path)
            if state is not None and not state.fully_read:
                raise WorkspaceError(WorkspaceErrorCode.FILE_NOT_READ, "read the complete file before overwriting it")
            expected_revision = state.revision if state is not None else None
        result = dict(
            await self._backend.request(
                "write_file",
                {
                    "workspace_id": workspace_id,
                    "path": path,
                    "content": content,
                    "create_parents": bool(create_parents),
                    "expected_revision": expected_revision,
                },
                timeout=30.0,
            )
        )
        resolved_path = str(result.get("path") or known_path)
        total_lines = int(result.get("total_lines", 0) or 0)
        async with self._read_state_lock:
            new_state = _ReadState(str(result.get("revision", "")), total_lines)
            new_state.add(1, total_lines)
            self._read_states[resolved_path] = new_state
        return result

    async def find_files(
        self,
        pattern: str,
        *,
        path: str = DEFAULT_AGENT_HOME,
        offset: int = 0,
        limit: int = 100,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> TextListResult:
        return await self._text_list_request(
            "find_files",
            {
                "workspace_id": _require_default(workspace_id),
                "pattern": _require_text(pattern, "pattern"),
                "path": _linux_path(path, "path"),
                "offset": int(offset),
                "limit": int(limit),
            },
        )

    async def search(
        self,
        pattern: str,
        *,
        path: str = DEFAULT_AGENT_HOME,
        glob: str | None = None,
        mode: str = "content",
        literal: bool = False,
        case_sensitive: bool = False,
        context_before: int = 0,
        context_after: int = 0,
        multiline: bool = False,
        offset: int = 0,
        limit: int = 250,
        workspace_id: str = DEFAULT_WORKSPACE_ID,
    ) -> TextListResult:
        params: dict[str, Any] = {
            "workspace_id": _require_default(workspace_id),
            "pattern": _require_text(pattern, "pattern"),
            "path": _linux_path(path, "path"),
            "mode": mode,
            "literal": bool(literal),
            "case_sensitive": bool(case_sensitive),
            "context_before": int(context_before),
            "context_after": int(context_after),
            "multiline": bool(multiline),
            "offset": int(offset),
            "limit": int(limit),
        }
        if glob:
            params["glob"] = _require_text(glob, "glob")
        return await self._text_list_request("search", params)

    async def _text_list_request(self, method: str, params: Mapping[str, Any]) -> TextListResult:
        self._require_open()
        return TextListResult.from_payload(await self._backend.request(method, params, timeout=30.0))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self._monitor_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._monitor_tasks.clear()
        await self._backend.close()
        async with self._staging_lock:
            staged_directories = list(self._staged_directories)
            self._staged_directories.clear()
        if staged_directories:
            await asyncio.gather(
                *(asyncio.to_thread(shutil.rmtree, path, True) for path in staged_directories),
                return_exceptions=True,
            )


__all__ = ["WorkspaceHostFile", "WorkspaceService"]
