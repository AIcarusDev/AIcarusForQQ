"""Async backend protocol and the Windows-to-WSL stdio implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .config import PROTOCOL_VERSION, WorkspaceConfig
from .errors import WorkspaceError, WorkspaceErrorCode


logger = logging.getLogger("AICQ.workspace")


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


_QQ_FILE_OP_SCRIPT = r"""
import json
import os
import stat
import sys

ROOT = "/var/lib/aicq-workspace/home"


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()


def fail(code, message):
    emit({"ok": False, "code": code, "message": message})
    raise SystemExit(0)


def parts(value):
    if not isinstance(value, list) or any(
        not isinstance(part, str) or not part or part in {".", ".."} or "/" in part or "\x00" in part
        for part in value
    ):
        fail("invalid_path", "invalid Agent-home-relative path")
    return value


def open_dir(relative):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(ROOT, flags)
    try:
        for part in relative:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


try:
    request = json.loads(sys.stdin.buffer.readline().decode("utf-8"))
    action = request.get("action") if isinstance(request, dict) else None
    relative = parts(request.get("parts"))
    if action == "free":
        info = os.statvfs(ROOT)
        emit({"ok": True, "free_bytes": info.f_bavail * info.f_frsize})
    elif action == "stat":
        if not relative:
            fail("not_regular", "path is not a regular file")
        parent = open_dir(relative[:-1])
        try:
            info = os.stat(relative[-1], dir_fd=parent, follow_symlinks=False)
            kind = "regular" if stat.S_ISREG(info.st_mode) else (
                "symlink" if stat.S_ISLNK(info.st_mode) else (
                    "directory" if stat.S_ISDIR(info.st_mode) else "other"
                )
            )
            emit({"ok": True, "kind": kind, "size_bytes": info.st_size, "modified_ns": info.st_mtime_ns})
        finally:
            os.close(parent)
    elif action == "delete":
        if not relative:
            fail("not_regular", "path is not a regular file")
        parent = open_dir(relative[:-1])
        try:
            info = os.stat(relative[-1], dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                fail("symlink", "path is a symbolic link")
            if stat.S_ISDIR(info.st_mode):
                fail("directory", "path is a directory")
            if not stat.S_ISREG(info.st_mode):
                fail("not_regular", "path is not a regular file")
            os.unlink(relative[-1], dir_fd=parent)
            os.fsync(parent)
            emit({"ok": True, "size_bytes": info.st_size})
        finally:
            os.close(parent)
    elif action in {"list", "cleanup_temps"}:
        base = open_dir(relative)
        rows = []
        removed = 0
        stack = [(base, [])]
        while stack:
            directory, prefix = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        name = entry.name
                        if name.startswith(".aicq-qq-file-"):
                            if action == "cleanup_temps" and entry.is_file(follow_symlinks=False):
                                try:
                                    os.unlink(name, dir_fd=directory)
                                    removed += 1
                                except OSError:
                                    pass
                            continue
                        child = prefix + [name]
                        if entry.is_symlink():
                            continue
                        if entry.is_file(follow_symlinks=False):
                            info = entry.stat(follow_symlinks=False)
                            rows.append({"parts": child, "size_bytes": info.st_size, "modified_ns": info.st_mtime_ns})
                        elif entry.is_dir(follow_symlinks=False):
                            try:
                                child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                            except OSError:
                                continue
                            stack.append((child_fd, child))
            finally:
                os.close(directory)
        if action == "cleanup_temps":
            emit({"ok": True, "removed": removed})
        else:
            emit({"ok": True, "files": rows})
    else:
        fail("invalid_action", "unsupported file operation")
except FileNotFoundError:
    fail("not_found", "file or parent directory does not exist")
except PermissionError:
    fail("permission_denied", "permission denied")
except OSError as exc:
    fail("filesystem_error", str(exc))
except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
    fail("invalid_path", str(exc))
""".strip()


_QQ_FILE_IMPORT_SCRIPT = r"""
import json
import os
import sys

ROOT = "/var/lib/aicq-workspace/home"


def emit(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.flush()


def fail(code, message):
    emit({"ok": False, "code": code, "message": message})
    raise SystemExit(0)


def valid_parts(value):
    return isinstance(value, list) and value and all(
        isinstance(part, str) and part and part not in {".", ".."} and "/" not in part and "\x00" not in part
        for part in value
    )


temp_name = ""
directory_fd = None
try:
    request = json.loads(sys.stdin.buffer.readline().decode("utf-8"))
    relative = request.get("parts") if isinstance(request, dict) else None
    expected_size = request.get("expected_size") if isinstance(request, dict) else None
    token = request.get("token") if isinstance(request, dict) else None
    if not valid_parts(relative) or not isinstance(expected_size, int) or expected_size < 0:
        fail("invalid_path", "invalid import request")
    if not isinstance(token, str) or not token.isalnum() or len(token) > 64:
        fail("invalid_path", "invalid import token")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(ROOT, flags)
    for part in relative[:-1]:
        try:
            next_fd = os.open(part, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            os.mkdir(part, 0o700, dir_fd=directory_fd)
            next_fd = os.open(part, flags, dir_fd=directory_fd)
        os.close(directory_fd)
        directory_fd = next_fd
    temp_name = ".aicq-qq-file-" + token
    output_fd = os.open(temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    size = 0
    try:
        while True:
            chunk = sys.stdin.buffer.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > expected_size:
                fail("size_mismatch", "import data exceeds expected size")
            view = memoryview(chunk)
            while view:
                written = os.write(output_fd, view)
                view = view[written:]
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    if size != expected_size:
        fail("size_mismatch", "import data size does not match expected size")
    try:
        os.link(temp_name, relative[-1], src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
    except FileExistsError:
        fail("already_exists", "target file already exists")
    os.unlink(temp_name, dir_fd=directory_fd)
    temp_name = ""
    os.fsync(directory_fd)
    emit({"ok": True, "size_bytes": size})
except FileNotFoundError:
    fail("not_found", "file or parent directory does not exist")
except PermissionError:
    fail("permission_denied", "permission denied")
except OSError as exc:
    fail("filesystem_error", str(exc))
except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
    fail("invalid_path", str(exc))
finally:
    if directory_fd is not None:
        if temp_name:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except OSError:
                pass
        os.close(directory_fd)
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

    async def close(self) -> None: ...


class WslQQFileImportSession:
    """One direct host-to-WSL file stream with an atomic final link."""

    def __init__(
        self,
        backend: "WslWorkspaceBackend",
        process: asyncio.subprocess.Process,
        relative_parts: tuple[str, ...],
        token: str,
    ) -> None:
        self.backend = backend
        self.process = process
        self.relative_parts = relative_parts
        self.token = token
        self._finished = False

    async def write(self, chunk: bytes) -> None:
        if self._finished or self.process.stdin is None:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "QQ 文件写入流已关闭")
        self.process.stdin.write(chunk)
        await self.process.stdin.drain()

    async def finish(self) -> Mapping[str, Any]:
        if self._finished:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "QQ 文件写入流已关闭")
        self._finished = True
        proc = self.process
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        proc.stdin.close()
        try:
            stdout, stderr = await asyncio.gather(proc.stdout.read(), proc.stderr.read())
            await proc.wait()
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            await asyncio.shield(proc.wait())
            raise
        finally:
            async with self.backend._state_lock:
                self.backend._processes.discard(proc)
        if proc.returncode != 0:
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件写入失败",
                details={"stderr": stderr.decode("utf-8", errors="replace")[-4096:]},
            )
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "QQ 文件写入返回了无效响应") from exc
        return result if isinstance(result, Mapping) else {"ok": False, "code": "filesystem_error"}

    async def abort(self) -> None:
        if self._finished:
            return
        self._finished = True
        proc = self.process
        if proc.returncode is None:
            proc.kill()
        await proc.wait()
        async with self.backend._state_lock:
            self.backend._processes.discard(proc)
        temporary_parts = (*self.relative_parts[:-1], f".aicq-qq-file-{self.token}")
        try:
            await self.backend.qq_file_operation("delete", temporary_parts, timeout=30.0)
        except Exception:
            logger.warning("Linux QQ 文件临时项清理失败", exc_info=True)


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

    def _python_argv(self, script: str) -> tuple[str, ...]:
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
            script,
        )

    async def qq_file_operation(
        self,
        action: str,
        relative_parts: Sequence[str],
        *,
        timeout: float = 120.0,
    ) -> Mapping[str, Any]:
        """Run one fixed, symlink-safe file metadata operation in Agent home."""

        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")
        payload = (
            json.dumps(
                {"action": str(action), "parts": [str(part) for part in relative_parts]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._python_argv(_QQ_FILE_OP_SCRIPT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=max(0.1, timeout))
        except (FileNotFoundError, OSError, asyncio.TimeoutError) as exc:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await asyncio.shield(proc.wait())
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件存储不可用",
                details={"transport_error": str(exc)},
            ) from exc
        except BaseException:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await asyncio.shield(proc.wait())
            raise
        assert proc is not None
        if proc.returncode != 0:
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件存储操作失败",
                details={"stderr": stderr.decode("utf-8", errors="replace")[-4096:]},
            )
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件存储返回了无效响应",
            ) from exc
        return result if isinstance(result, Mapping) else {"ok": False, "code": "filesystem_error"}

    async def import_file(
        self,
        relative_parts: Sequence[str],
        source: Path,
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        """Stream a host-local file into Agent home without shell interpolation."""

        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")
        expected_size = source.stat().st_size
        payload = (
            json.dumps(
                {
                    "parts": [str(part) for part in relative_parts],
                    "expected_size": expected_size,
                    "token": uuid.uuid4().hex,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._python_argv(_QQ_FILE_IMPORT_SCRIPT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件存储不可用",
                details={"transport_error": str(exc)},
            ) from exc

        async def transfer() -> tuple[bytes, bytes]:
            assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
            proc.stdin.write(payload)
            await proc.stdin.drain()
            with source.open("rb") as handle:
                while True:
                    chunk = await asyncio.to_thread(handle.read, 1024 * 1024)
                    if not chunk:
                        break
                    proc.stdin.write(chunk)
                    await proc.stdin.drain()
            proc.stdin.close()
            stdout, stderr = await asyncio.gather(proc.stdout.read(), proc.stderr.read())
            await proc.wait()
            return stdout, stderr

        try:
            if timeout is None:
                stdout, stderr = await transfer()
            else:
                stdout, stderr = await asyncio.wait_for(transfer(), max(0.1, timeout))
        except (OSError, asyncio.TimeoutError) as exc:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件写入未完成",
                details={"transport_error": str(exc)},
            ) from exc
        if proc.returncode != 0:
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件写入失败",
                details={"stderr": stderr.decode("utf-8", errors="replace")[-4096:]},
            )
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "QQ 文件写入返回了无效响应") from exc
        return result if isinstance(result, Mapping) else {"ok": False, "code": "filesystem_error"}

    async def begin_qq_file_import(
        self,
        relative_parts: Sequence[str],
        expected_size: int,
    ) -> WslQQFileImportSession:
        """Open a direct stream whose temporary and final bytes both stay in WSL."""

        if self._closed:
            raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")
        size = int(expected_size)
        if size < 0:
            raise WorkspaceError(WorkspaceErrorCode.INVALID_ARGUMENT, "QQ 文件大小无效")
        parts = tuple(str(part) for part in relative_parts)
        token = uuid.uuid4().hex
        payload = (
            json.dumps(
                {"parts": list(parts), "expected_size": size, "token": token},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._python_argv(_QQ_FILE_IMPORT_SCRIPT),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, OSError) as exc:
            raise WorkspaceError(
                WorkspaceErrorCode.BROKER_UNAVAILABLE,
                "QQ 文件存储不可用",
                details={"transport_error": str(exc)},
            ) from exc
        async with self._state_lock:
            if self._closed:
                proc.kill()
                await proc.wait()
                raise WorkspaceError(WorkspaceErrorCode.BROKER_UNAVAILABLE, "computer backend is closed")
            self._processes.add(proc)
        assert proc.stdin is not None
        try:
            proc.stdin.write(payload)
            await proc.stdin.drain()
        except BaseException:
            if proc.returncode is None:
                proc.kill()
            await asyncio.shield(proc.wait())
            async with self._state_lock:
                self._processes.discard(proc)
            raise
        return WslQQFileImportSession(self, proc, parts, token)

    async def export_file(
        self,
        relative_parts: Sequence[str],
        destination: Path,
        *,
        timeout: float | None = 120.0,
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
                if timeout is None:
                    returncode, stderr, size = await transfer()
                else:
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
