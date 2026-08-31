"""Physical storage implementations behind Linux-shaped QQ file paths."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from workspace.control import detect_workspace_presence

from .logical import AGENT_HOME, agent_home_parts


class StorageError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class StoredFile:
    path: str
    size_bytes: int
    modified_ns: int
    kind: str = "regular"


def _is_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return path.is_symlink() or bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _encode_component(value: str) -> str:
    unsafe = set('<>:"/\\|?*%')
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    force_escape_first = value.startswith("~q") or value.split(".", 1)[0].upper() in reserved
    pieces: list[str] = []
    encoded_any = False
    for index, char in enumerate(value):
        must_encode = (
            (index == 0 and force_escape_first)
            or char in unsafe
            or ord(char) < 32
            or (index == len(value) - 1 and char in {" ", "."})
        )
        if must_encode:
            encoded_any = True
            pieces.extend(f"%{byte:02X}" for byte in char.encode("utf-8"))
        else:
            pieces.append(char)
    encoded = "".join(pieces)
    return ("~q" + encoded) if encoded_any else encoded


def _decode_component(value: str) -> str:
    if not value.startswith("~q"):
        return value
    value = value[2:]
    try:
        output = bytearray()
        result: list[str] = []
        index = 0
        while index < len(value):
            if value[index] == "%" and index + 2 < len(value):
                try:
                    output.append(int(value[index + 1 : index + 3], 16))
                    index += 3
                    continue
                except ValueError:
                    pass
            if output:
                result.append(output.decode("utf-8", errors="strict"))
                output.clear()
            result.append(value[index])
            index += 1
        if output:
            result.append(output.decode("utf-8", errors="strict"))
        return "".join(result)
    except UnicodeDecodeError:
        return value


class HostFallbackStorage:
    backend_name = "host_fallback"

    def __init__(self, project_root: Path) -> None:
        self.root = project_root / "cache" / "qq_file_fallback" / "home" / "agent"

    def _physical(self, logical: PurePosixPath) -> Path:
        parts = agent_home_parts(logical)
        return self.root.joinpath(*(_encode_component(part) for part in parts))

    def storage_relpath(self, logical: PurePosixPath) -> str:
        return "/".join(_encode_component(part) for part in agent_home_parts(logical))

    def _ensure_safe_parents(self, target: Path, *, create: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        current = self.root
        if _is_reparse(current):
            raise StorageError("filesystem_unavailable", "宿主机后备目录不可安全访问")
        for part in target.relative_to(self.root).parts[:-1]:
            current = current / part
            if current.exists():
                if _is_reparse(current) or not current.is_dir():
                    raise StorageError("symlink_not_allowed", "文件路径包含不允许的重解析目录")
            elif create:
                current.mkdir()
            else:
                raise FileNotFoundError(str(current))

    async def stat(self, logical: PurePosixPath) -> StoredFile | None:
        def work() -> StoredFile | None:
            target = self._physical(logical)
            try:
                self._ensure_safe_parents(target, create=False)
                info = target.lstat()
            except FileNotFoundError:
                return None
            kind = "symlink" if _is_reparse(target) else (
                "regular" if stat.S_ISREG(info.st_mode) else (
                    "directory" if stat.S_ISDIR(info.st_mode) else "other"
                )
            )
            return StoredFile(str(logical), int(info.st_size), int(info.st_mtime_ns), kind)

        return await asyncio.to_thread(work)

    async def list(self, logical_root: PurePosixPath) -> list[StoredFile]:
        def work() -> list[StoredFile]:
            base = self._physical(logical_root)
            if not base.exists():
                return []
            self._ensure_safe_parents(base / "_", create=False)
            if _is_reparse(base) or not base.is_dir():
                raise StorageError("filesystem_unavailable", "宿主机后备目录不可安全访问")
            rows: list[StoredFile] = []
            for directory, dirs, files in os.walk(base, followlinks=False):
                directory_path = Path(directory)
                dirs[:] = [name for name in dirs if not _is_reparse(directory_path / name)]
                for name in files:
                    if name.startswith(".aicq-qq-file-"):
                        continue
                    target = directory_path / name
                    try:
                        info = target.lstat()
                    except FileNotFoundError:
                        continue
                    if _is_reparse(target) or not stat.S_ISREG(info.st_mode):
                        continue
                    rel_physical = target.relative_to(base).parts
                    rel_logical = tuple(_decode_component(part) for part in rel_physical)
                    logical = logical_root.joinpath(*rel_logical)
                    rows.append(StoredFile(str(logical), int(info.st_size), int(info.st_mtime_ns)))
            return rows

        return await asyncio.to_thread(work)

    async def commit(self, source: Path, logical: PurePosixPath) -> int:
        def work() -> int:
            target = self._physical(logical)
            self._ensure_safe_parents(target, create=True)
            temporary = target.parent / f".aicq-qq-file-{uuid.uuid4().hex}"
            try:
                with source.open("rb") as input_file, temporary.open("xb") as output_file:
                    shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.link(temporary, target)
                return int(target.stat().st_size)
            except FileExistsError as exc:
                raise StorageError("already_exists", "目标文件已经存在") from exc
            except OSError as exc:
                raise StorageError("write_failed", "宿主机后备文件写入失败", retryable=True) from exc
            finally:
                temporary.unlink(missing_ok=True)

        return await asyncio.to_thread(work)

    async def delete(self, logical: PurePosixPath) -> int:
        def work() -> int:
            target = self._physical(logical)
            try:
                self._ensure_safe_parents(target, create=False)
                info = target.lstat()
            except FileNotFoundError as exc:
                raise StorageError("file_not_found", "指定的文件路径不存在") from exc
            if _is_reparse(target):
                raise StorageError("symlink_not_allowed", "不允许删除符号链接或重解析点")
            if stat.S_ISDIR(info.st_mode):
                raise StorageError("directory_not_allowed", "不允许删除目录")
            if not stat.S_ISREG(info.st_mode):
                raise StorageError("not_a_regular_file", "指定路径不是普通文件")
            try:
                target.unlink()
            except PermissionError as exc:
                raise StorageError("permission_denied", "没有权限删除指定文件", retryable=True) from exc
            return int(info.st_size)

        return await asyncio.to_thread(work)

    async def free_bytes(self) -> int:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
        return int((await asyncio.to_thread(shutil.disk_usage, self.root)).free)

    def host_path(self, logical: PurePosixPath) -> Path:
        return self._physical(logical)

    async def cleanup_temps(self, _logical_root: PurePosixPath) -> None:
        return None


class LinuxWorkspaceStorage:
    backend_name = "linux"

    def __init__(self, workspace_service: Any) -> None:
        if workspace_service is None:
            raise StorageError("runtime_unavailable", "Linux 文件空间当前不可用", retryable=True)
        self.workspace_service = workspace_service

    def storage_relpath(self, logical: PurePosixPath) -> str:
        return "/".join(agent_home_parts(logical))

    async def stat(self, logical: PurePosixPath) -> StoredFile | None:
        try:
            result = await self.workspace_service.qq_file_operation("stat", str(logical))
        except Exception as exc:
            raise StorageError("filesystem_unavailable", "Linux 文件空间当前不可用", retryable=True) from exc
        if result.get("ok"):
            return StoredFile(
                str(logical),
                int(result.get("size_bytes") or 0),
                int(result.get("modified_ns") or 0),
                str(result.get("kind") or "regular"),
            )
        if result.get("code") == "not_found":
            return None
        raise StorageError("filesystem_unavailable", str(result.get("message") or "Linux 文件空间不可用"), retryable=True)

    async def list(self, logical_root: PurePosixPath) -> list[StoredFile]:
        try:
            result = await self.workspace_service.qq_file_operation("list", str(logical_root))
        except Exception as exc:
            raise StorageError("filesystem_unavailable", "Linux 文件空间当前不可用", retryable=True) from exc
        if result.get("code") == "not_found":
            return []
        if not result.get("ok"):
            raise StorageError("filesystem_unavailable", str(result.get("message") or "Linux 文件空间不可用"), retryable=True)
        rows: list[StoredFile] = []
        for item in result.get("files") or []:
            parts = item.get("parts") if isinstance(item, dict) else None
            if not isinstance(parts, list):
                continue
            logical = logical_root.joinpath(*(str(part) for part in parts))
            rows.append(StoredFile(str(logical), int(item.get("size_bytes") or 0), int(item.get("modified_ns") or 0)))
        return rows

    async def commit(self, source: Path, logical: PurePosixPath) -> int:
        try:
            result = await self.workspace_service.import_host_file(str(logical), source)
        except Exception as exc:
            raise StorageError("write_failed", "Linux 文件写入失败", retryable=True) from exc
        if result.get("ok"):
            return int(result.get("size_bytes") or 0)
        code = str(result.get("code") or "write_failed")
        raise StorageError(code, str(result.get("message") or "Linux 文件写入失败"), retryable=code != "already_exists")

    async def delete(self, logical: PurePosixPath) -> int:
        try:
            result = await self.workspace_service.qq_file_operation("delete", str(logical))
        except Exception as exc:
            raise StorageError("filesystem_unavailable", "Linux 文件空间当前不可用", retryable=True) from exc
        if result.get("ok"):
            return int(result.get("size_bytes") or 0)
        code = str(result.get("code") or "filesystem_unavailable")
        mapped = {
            "not_found": "file_not_found",
            "not_regular": "not_a_regular_file",
            "directory": "directory_not_allowed",
            "symlink": "symlink_not_allowed",
        }.get(code, code)
        raise StorageError(mapped, str(result.get("message") or "Linux 文件删除失败"), retryable=mapped == "filesystem_unavailable")

    async def free_bytes(self) -> int:
        try:
            result = await self.workspace_service.qq_file_operation("free", str(AGENT_HOME))
        except Exception as exc:
            raise StorageError("filesystem_unavailable", "无法读取 Linux 文件空间容量", retryable=True) from exc
        if result.get("ok"):
            return int(result.get("free_bytes") or 0)
        raise StorageError(
            "filesystem_unavailable",
            str(result.get("message") or "无法读取 Linux 文件空间容量"),
            retryable=True,
        )

    def download_sink(self, logical: PurePosixPath):
        return LinuxDownloadSink(self.workspace_service, logical)

    async def cleanup_temps(self, logical_root: PurePosixPath) -> None:
        try:
            result = await self.workspace_service.qq_file_operation("cleanup_temps", str(logical_root))
        except Exception as exc:
            raise StorageError("filesystem_unavailable", "无法清理中断的 Linux QQ 文件", retryable=True) from exc
        if not result.get("ok") and result.get("code") != "not_found":
            raise StorageError("filesystem_unavailable", "无法清理中断的 Linux QQ 文件", retryable=True)


class LinuxDownloadSink:
    """Adapter-facing sink that keeps partial and final bytes inside Linux."""

    def __init__(self, workspace_service: Any, logical: PurePosixPath) -> None:
        self.workspace_service = workspace_service
        self.logical = logical
        self.session: Any = None
        self.size = 0

    async def begin(self, expected_size: int) -> None:
        if self.session is not None:
            raise StorageError("write_failed", "Linux QQ 文件写入流重复打开")
        try:
            self.session = await self.workspace_service.begin_qq_file_import(
                str(self.logical), int(expected_size)
            )
        except Exception as exc:
            raise StorageError("write_failed", "Linux QQ 文件写入流无法打开", retryable=True) from exc

    async def write(self, chunk: bytes) -> None:
        if self.session is None:
            raise StorageError("write_failed", "Linux QQ 文件写入流尚未打开")
        try:
            await self.session.write(chunk)
        except Exception as exc:
            raise StorageError("write_failed", "Linux QQ 文件写入失败", retryable=True) from exc
        self.size += len(chunk)

    async def finish(self) -> int:
        if self.session is None:
            raise StorageError("write_failed", "Linux QQ 文件写入流尚未打开")
        try:
            result = await self.session.finish()
        except Exception as exc:
            raise StorageError("write_failed", "Linux QQ 文件提交失败", retryable=True) from exc
        if not result.get("ok"):
            code = str(result.get("code") or "write_failed")
            raise StorageError(code, str(result.get("message") or "Linux QQ 文件写入失败"), retryable=code != "already_exists")
        committed = int(result.get("size_bytes") or 0)
        if committed != self.size:
            raise StorageError("size_mismatch", "Linux QQ 文件写入大小不一致", retryable=True)
        return committed

    async def abort(self) -> None:
        if self.session is not None:
            await self.session.abort()

    async def rollback(self) -> None:
        try:
            result = await self.workspace_service.qq_file_operation("delete", str(self.logical))
        except Exception as exc:
            raise StorageError("write_failed", "无法回滚未登记的 Linux QQ 文件", retryable=True) from exc
        if not result.get("ok") and result.get("code") != "not_found":
            raise StorageError("write_failed", "无法回滚未登记的 Linux QQ 文件", retryable=True)


class StorageRouter:
    def __init__(self, project_root: Path, workspace_service: Any) -> None:
        self.project_root = project_root
        self.workspace_service = workspace_service

    def active(self):
        presence = detect_workspace_presence()
        if presence == "present":
            return LinuxWorkspaceStorage(self.workspace_service)
        if presence == "absent":
            return HostFallbackStorage(self.project_root)
        raise StorageError("runtime_unavailable", "无法确认 Linux 文件空间是否存在", retryable=True)

    def frozen(self, backend: str):
        if backend == "linux":
            return LinuxWorkspaceStorage(self.workspace_service)
        if backend == "host_fallback":
            return HostFallbackStorage(self.project_root)
        raise StorageError("runtime_unavailable", "下载任务的存储后端不可用", retryable=True)
