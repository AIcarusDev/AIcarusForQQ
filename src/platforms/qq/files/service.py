"""High-level QQ file operations shared by the model-facing tools."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator, Iterable

from .cursor import CursorCodec
from .logical import (
    LogicalPathError,
    account_file_root,
    collision_name,
    conversation_root,
    extension_for,
    normalized_filename,
    require_agent_qq,
    sanitize_filename,
    validate_logical_path,
)
from .repository import ACTIVE_STATUSES, QQFileRepository, now_iso
from .storage import HostFallbackStorage, LinuxWorkspaceStorage, StorageError, StorageRouter
from .parsers import parse_document_safe


logger = logging.getLogger("AICQ.qq_file")
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024 * 1024
DOWNLOAD_OBSERVATION_SECONDS = 15.0
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_PARSER_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=2)


class QQFileError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


def _source_file_segment(message: dict[str, Any]) -> tuple[str, str, int | None]:
    raw_segments = message.get("message")
    if not isinstance(raw_segments, list):
        raise QQFileError("not_file_message", "指定消息不是 QQ 文件消息")
    files = [segment for segment in raw_segments if isinstance(segment, dict) and segment.get("type") == "file"]
    if len(files) != 1:
        raise QQFileError("not_file_message", "指定消息不是单文件 QQ 文件消息")
    data = files[0].get("data") if isinstance(files[0].get("data"), dict) else {}
    filename = ""
    for key in ("name", "file_name", "filename", "file"):
        filename = str(data.get(key) or "").strip()
        if filename:
            break
    file_id = str(data.get("file_id") or data.get("id") or "").strip()
    size: int | None = None
    for key in ("file_size", "size"):
        if data.get(key) is None or data.get(key) == "":
            continue
        try:
            candidate = int(data[key])
        except (TypeError, ValueError):
            continue
        if candidate >= 0:
            size = candidate
            break
    return sanitize_filename(filename), file_id, size


class QQFileService:
    def __init__(
        self,
        qq_client: Any,
        workspace_service: Any,
        *,
        project_root: Path = _PROJECT_ROOT,
        repository: QQFileRepository | None = None,
        storage_router: StorageRouter | None = None,
    ) -> None:
        self.qq_client = qq_client
        self.workspace_service = workspace_service
        self.project_root = project_root
        self.repository = repository or QQFileRepository()
        self.storage_router = storage_router or StorageRouter(project_root, workspace_service)
        self.cursor = CursorCodec(project_root)
        self.temp_root = project_root / "cache" / "qq_file_downloads"
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._ready_lock = asyncio.Lock()
        self._ready = False
        self._path_lock = asyncio.Lock()

    def agent_qq(self) -> str:
        return require_agent_qq(
            getattr(self.qq_client, "bot_id", None) or getattr(self.qq_client, "last_bot_id", None)
        )

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        async with self._ready_lock:
            if self._ready:
                return
            await self.repository.recover_interrupted(self.agent_qq())
            await asyncio.to_thread(self.temp_root.mkdir, parents=True, exist_ok=True)
            for temporary in await asyncio.to_thread(lambda: list(self.temp_root.glob("*.part"))):
                await asyncio.to_thread(temporary.unlink, missing_ok=True)
            try:
                storage = self.storage_router.active()
                await storage.cleanup_temps(account_file_root(self.agent_qq()))
            except Exception:
                logger.warning("中断的 QQ 文件临时项暂时无法清理", exc_info=True)
            self._ready = True

    @staticmethod
    def _session_identity(session: Any) -> tuple[str, str, str]:
        if session is None:
            raise QQFileError("no_current_qq_session", "当前不在具体 QQ 会话中", retryable=False)
        conv_type = str(getattr(session, "conv_type", "") or "")
        conv_id = str(getattr(session, "conv_id", "") or "")
        session_key = str(getattr(session, "key", "") or "")
        if conv_type not in {"private", "group"} or not conv_id or not session_key:
            raise QQFileError("no_current_qq_session", "当前 QQ 会话不支持文件下载", retryable=False)
        return session_key, conv_type, conv_id

    async def _resolve_message(self, message_id: str, conv_type: str, conv_id: str) -> tuple[str, str, int | None]:
        if not getattr(self.qq_client, "connected", False):
            raise QQFileError("qq_adapter_unavailable", "QQ adapter 当前不可用", retryable=True)
        response = await self.qq_client.send_api("get_msg", {"message_id": str(message_id)}, timeout=15.0)
        if not isinstance(response, dict):
            raise QQFileError("message_not_found", "当前 QQ 会话中未找到指定消息", retryable=False)
        message_type = str(response.get("message_type") or "")
        if conv_type == "group":
            if message_type and message_type != "group":
                raise QQFileError("message_not_found", "指定消息不属于当前 QQ 会话")
            if str(response.get("group_id") or "") != conv_id:
                raise QQFileError("message_not_found", "指定消息不属于当前 QQ 会话")
        elif conv_type == "private":
            if message_type and message_type != "private":
                raise QQFileError("message_not_found", "指定消息不属于当前 QQ 会话")
            peer_ids = {
                str(response.get("user_id") or ""),
                str((response.get("sender") or {}).get("user_id") or ""),
                str(response.get("target_id") or ""),
            }
            peer_ids.discard("")
            peer_ids.discard(self.agent_qq())
            if peer_ids and conv_id not in peer_ids:
                raise QQFileError("message_not_found", "指定消息不属于当前 QQ 会话")
        return _source_file_segment(response)

    async def _choose_target(self, storage: Any, root: PurePosixPath, filename: str, agent_qq: str) -> PurePosixPath:
        for index in range(0, 10000):
            candidate = root / collision_name(filename, index)
            if await storage.stat(candidate) is not None:
                continue
            if await self.repository.active_for_path(agent_qq, storage.backend_name, str(candidate)) is not None:
                continue
            return candidate
        raise QQFileError("internal_error", "无法为下载文件分配目标名称")

    async def start(self, message_id: object, session: Any) -> dict[str, Any]:
        await self.ensure_ready()
        normalized_message_id = str(message_id or "").strip()
        if not normalized_message_id:
            raise QQFileError("message_not_found", "message_id 不能为空")
        agent_qq = self.agent_qq()
        session_key, conv_type, conv_id = self._session_identity(session)
        try:
            storage = self.storage_router.active()
        except StorageError as exc:
            raise QQFileError(exc.code, str(exc), retryable=exc.retryable) from exc

        active = await self.repository.find_active(agent_qq, session_key, normalized_message_id)
        if active is not None:
            return {
                "ok": True,
                "action": "start",
                "outcome": "already_downloading",
                "observation_timeout": False,
                "job": self.repository.job(active),
            }

        record = await self.repository.latest_record(
            agent_qq, session_key, normalized_message_id, storage.backend_name
        )
        if record is not None:
            existing = await storage.stat(PurePosixPath(record["local_path"]))
            if existing is not None and existing.kind == "regular":
                return {
                    "ok": True,
                    "action": "start",
                    "outcome": "already_exists",
                    "file": {
                        "message_id": normalized_message_id,
                        "conversation": {"type": conv_type, "id": conv_id},
                        "original_filename": record["original_filename"],
                        "local_path": record["local_path"],
                        "size_bytes": existing.size_bytes,
                        "downloaded_at": record["downloaded_at"],
                    },
                }

        filename, source_file_id, declared_size = await self._resolve_message(
            normalized_message_id, conv_type, conv_id
        )
        if declared_size is not None and declared_size > MAX_DOWNLOAD_BYTES:
            raise QQFileError(
                "file_too_large",
                "QQ 文件超过单文件下载上限",
                details={"size_bytes": declared_size, "limit_bytes": MAX_DOWNLOAD_BYTES},
            )
        if declared_size is not None:
            free = await storage.free_bytes()
            if free < declared_size:
                raise QQFileError(
                    "insufficient_disk_space",
                    "当前文件空间不足以保存该文件",
                    details={"required_bytes": declared_size, "available_bytes": free},
                )
        root = conversation_root(agent_qq, conv_type, conv_id)
        async with self._path_lock:
            active = await self.repository.find_active(agent_qq, session_key, normalized_message_id)
            if active is not None:
                return {
                    "ok": True,
                    "action": "start",
                    "outcome": "already_downloading",
                    "observation_timeout": False,
                    "job": self.repository.job(active),
                }
            target = await self._choose_target(storage, root, filename, agent_qq)
            row = await self.repository.create_job(
                {
                    "agent_qq": agent_qq,
                    "session_key": session_key,
                    "message_id": normalized_message_id,
                    "conversation_type": conv_type,
                    "conversation_id": conv_id,
                    "original_filename": filename,
                    "source_file_id": source_file_id,
                    "total_bytes": declared_size,
                    "target_path": str(target),
                    "storage_backend": storage.backend_name,
                    "storage_relpath": storage.storage_relpath(target),
                }
            )
        download_id = row["download_id"]
        task = asyncio.create_task(self._run_download(download_id), name=f"qq-file-{download_id}")
        self._tasks[download_id] = task
        observation_timeout = False
        try:
            await asyncio.wait_for(asyncio.shield(task), DOWNLOAD_OBSERVATION_SECONDS)
        except asyncio.TimeoutError:
            observation_timeout = True
        current = await self.repository.get_job_row(download_id, agent_qq=agent_qq)
        assert current is not None
        return {
            "ok": True,
            "action": "start",
            "outcome": "started",
            "observation_timeout": observation_timeout,
            "job": self.repository.job(current),
        }

    async def _run_download(self, download_id: str) -> None:
        row = await self.repository.get_job_row(download_id)
        if row is None:
            return
        temporary = self.temp_root / f"{download_id}.part"
        last_update = 0.0
        last_bytes = 0
        committed = False
        storage: Any = None

        async def progress(downloaded: int, total: int | None) -> None:
            nonlocal last_update, last_bytes
            now = time.monotonic()
            if downloaded != total and downloaded - last_bytes < 1024 * 1024 and now - last_update < 0.5:
                return
            last_update = now
            last_bytes = downloaded
            await self.repository.update_job(
                download_id,
                bytes_downloaded=downloaded,
                total_bytes=total if total is not None else row.get("total_bytes"),
            )

        try:
            await self.repository.update_job(download_id, status="resolving")
            storage = self.storage_router.frozen(row["storage_backend"])
            await self.repository.update_job(download_id, status="downloading")
            if not str(row.get("source_file_id") or ""):
                raise QQFileError("source_unavailable", "该文件消息当前没有可用的下载入口", retryable=True)
            destination: Any = temporary
            if isinstance(storage, LinuxWorkspaceStorage):
                destination = storage.download_sink(PurePosixPath(row["target_path"]))
            result = await self.qq_client.download_file_stream(
                row["source_file_id"],
                destination,
                max_bytes=MAX_DOWNLOAD_BYTES,
                on_progress=progress,
            )
            size = int(result.get("size_bytes") or 0)
            committed = isinstance(storage, LinuxWorkspaceStorage)
            await self.repository.update_job(
                download_id,
                status="verifying",
                bytes_downloaded=size,
                total_bytes=size if row.get("total_bytes") is None else row.get("total_bytes"),
            )
            if row.get("total_bytes") is not None and int(row["total_bytes"]) != size:
                raise QQFileError("size_mismatch", "QQ 文件实际大小与声明不一致", retryable=True)
            async with self._path_lock:
                committed_size = size
                if not committed:
                    commit_task = asyncio.create_task(
                        storage.commit(temporary, PurePosixPath(row["target_path"]))
                    )
                    try:
                        committed_size = await asyncio.shield(commit_task)
                        committed = True
                    except asyncio.CancelledError:
                        try:
                            committed_size = await commit_task
                            committed = True
                        finally:
                            if committed:
                                await storage.delete(PurePosixPath(row["target_path"]))
                                committed = False
                        raise
                if committed_size != size:
                    try:
                        await storage.delete(PurePosixPath(row["target_path"]))
                    except Exception:
                        logger.warning("QQ 文件提交大小异常后的清理失败: %s", row["target_path"], exc_info=True)
                    raise QQFileError("verification_failed", "保存后的 QQ 文件大小校验失败", retryable=True)
                await self.repository.add_record(row, committed_size)
                await self.repository.update_job(
                    download_id,
                    status="completed",
                    local_path=row["target_path"],
                    bytes_downloaded=committed_size,
                    total_bytes=committed_size,
                    finished_at=now_iso(),
                )
        except asyncio.CancelledError:
            if committed and storage is not None:
                try:
                    await storage.delete(PurePosixPath(row["target_path"]))
                    committed = False
                except Exception:
                    logger.warning("停止下载时无法清理已提交文件: %s", row["target_path"], exc_info=True)
            await self.repository.update_job(
                download_id,
                status="stopped",
                finished_at=now_iso(),
                failure_code=None,
                failure_message=None,
                failure_retryable=None,
            )
            raise
        except Exception as exc:
            if committed and storage is not None:
                try:
                    await storage.delete(PurePosixPath(row["target_path"]))
                    committed = False
                except Exception:
                    logger.warning("下载失败后无法清理已提交文件: %s", row["target_path"], exc_info=True)
            code = "internal_error"
            retryable = True
            message = "QQ 文件下载失败"
            if isinstance(exc, QQFileError):
                code, retryable = exc.code, exc.retryable
                message = str(exc)
            elif isinstance(exc, StorageError):
                code, retryable = exc.code, exc.retryable
                message = str(exc)
            elif isinstance(exc, OverflowError):
                code, retryable = "file_too_large", False
                message = "QQ 文件超过下载大小限制"
            elif isinstance(exc, (ConnectionError, TimeoutError)):
                code, retryable = "transport_error", True
                message = "QQ 文件传输中断"
            elif isinstance(exc, ValueError):
                code, retryable = "verification_failed", True
                message = "QQ 文件下载内容校验失败"
            logger.warning("QQ 文件下载失败 download_id=%s: %s", download_id, exc)
            await self.repository.update_job(
                download_id,
                status="failed",
                failure_code=code,
                failure_message=message,
                failure_retryable=1 if retryable else 0,
                finished_at=now_iso(),
            )
        finally:
            await asyncio.to_thread(temporary.unlink, missing_ok=True)
            self._tasks.pop(download_id, None)

    async def poll(self, download_id: object) -> dict[str, Any]:
        await self.ensure_ready()
        row = await self.repository.get_job_row(str(download_id or "").strip(), agent_qq=self.agent_qq())
        if row is None:
            raise QQFileError("download_not_found", "未找到指定下载任务")
        return {"ok": True, "action": "poll", "job": self.repository.job(row)}

    async def _observe_existing_job(self, download_id: str, seconds: float) -> dict[str, Any] | None:
        deadline = asyncio.get_running_loop().time() + seconds
        while True:
            row = await self.repository.get_job_row(download_id, agent_qq=self.agent_qq())
            if row is None or row["status"] not in ACTIVE_STATUSES:
                return row
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return row
            task = self._tasks.get(download_id)
            if task is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(task), remaining)
                except asyncio.TimeoutError:
                    return await self.repository.get_job_row(download_id, agent_qq=self.agent_qq())
            else:
                await asyncio.sleep(min(0.25, remaining))

    async def list_downloads(
        self,
        statuses: Iterable[str] | None,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        await self.ensure_ready()
        active, terminal, has_more = await self.repository.list_jobs(
            self.agent_qq(), statuses, offset, limit
        )
        return {
            "ok": True,
            "action": "list",
            "active": [self.repository.job(row) for row in active],
            "terminal": [self.repository.job(row) for row in terminal],
            "offset": offset,
            "limit": limit,
            "terminal_has_more": has_more,
            "next_offset": offset + limit if has_more else None,
        }

    async def stop(self, download_id: object) -> dict[str, Any]:
        await self.ensure_ready()
        normalized = str(download_id or "").strip()
        row = await self.repository.get_job_row(normalized, agent_qq=self.agent_qq())
        if row is None:
            raise QQFileError("download_not_found", "未找到指定下载任务")
        if row["status"] not in ACTIVE_STATUSES:
            return {"ok": True, "action": "stop", "outcome": "already_terminal", "job": self.repository.job(row)}
        task = self._tasks.get(normalized)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            await self.repository.update_job(normalized, status="stopped", finished_at=now_iso())
        current = await self.repository.get_job_row(normalized, agent_qq=self.agent_qq())
        assert current is not None
        return {"ok": True, "action": "stop", "outcome": "stopped", "job": self.repository.job(current)}

    def _resolve_scope(self, scope: dict[str, Any] | None, session: Any) -> tuple[dict[str, Any], PurePosixPath]:
        agent_qq = self.agent_qq()
        scope = scope or {"type": "current"}
        scope_type = str(scope.get("type") or "current")
        if scope_type == "current":
            _key, conv_type, conv_id = self._session_identity(session)
            return (
                {"type": "current", "conversation": {"type": conv_type, "id": conv_id}},
                conversation_root(agent_qq, conv_type, conv_id),
            )
        if scope_type == "conversation":
            conv_type = str(scope.get("conversation_type") or "")
            conv_id = str(scope.get("conversation_id") or "")
            return (
                {"type": "conversation", "conversation": {"type": conv_type, "id": conv_id}},
                conversation_root(agent_qq, conv_type, conv_id),
            )
        if scope_type == "all":
            return {"type": "all"}, account_file_root(agent_qq)
        raise QQFileError("invalid_scope", "QQ 文件搜索范围无效")

    @staticmethod
    def _conversation_from_path(path: PurePosixPath, agent_qq: str) -> dict[str, str] | None:
        try:
            relative = path.relative_to(account_file_root(agent_qq))
        except ValueError:
            return None
        if not relative.parts:
            return None
        directory = relative.parts[0]
        for conv_type in ("private", "group"):
            prefix = conv_type + "_"
            if directory.startswith(prefix) and directory[len(prefix) :]:
                return {"type": conv_type, "id": directory[len(prefix) :]}
        return None

    async def _local_rows(self, storage: Any, root: PurePosixPath) -> list[dict[str, Any]]:
        agent_qq = self.agent_qq()
        files = await storage.list(root)
        records = await self.repository.records_for_paths(
            agent_qq, storage.backend_name, (item.path for item in files)
        )
        rows: list[dict[str, Any]] = []
        account_root = account_file_root(agent_qq)
        for item in files:
            logical = PurePosixPath(item.path)
            conversation = self._conversation_from_path(logical, agent_qq)
            if conversation is None:
                continue
            record = records.get(item.path)
            source = None
            if record is not None:
                source = {
                    "message_id": record["message_id"],
                    "original_filename": record["original_filename"],
                    "recorded_size_bytes": int(record["size_bytes"]),
                    "downloaded_at": record["downloaded_at"],
                }
            rows.append(
                {
                    "name": logical.name,
                    "path": item.path,
                    "relative_path": logical.relative_to(account_root).as_posix(),
                    "extension": extension_for(logical.name),
                    "size_bytes": item.size_bytes,
                    "modified_at": datetime.fromtimestamp(
                        item.modified_ns / 1_000_000_000, timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    "conversation": conversation,
                    "managed": record is not None,
                    "source": source,
                }
            )
        return rows

    async def list_files(
        self,
        *,
        scope: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
        session: Any,
    ) -> dict[str, Any]:
        await self.ensure_ready()
        agent_qq = self.agent_qq()
        try:
            storage = self.storage_router.active()
        except StorageError as exc:
            raise QQFileError(exc.code, str(exc), retryable=exc.retryable) from exc
        offset = 0
        if cursor:
            try:
                state = self.cursor.loads(cursor, "list_files")
            except Exception as exc:
                raise QQFileError("invalid_cursor", str(exc)) from exc
            if state.get("agent_qq") != agent_qq or state.get("backend") != storage.backend_name:
                raise QQFileError("invalid_cursor", "游标对应的文件空间已经变化")
            resolved_scope = state.get("scope")
            root = PurePosixPath(str(state.get("root") or ""))
            limit = int(state.get("limit") or limit)
            offset = int(state.get("offset") or 0)
        else:
            resolved_scope, root = self._resolve_scope(scope, session)
        rows = await self._local_rows(storage, root)
        rows.sort(key=lambda row: row["relative_path"])
        page = rows[offset : offset + limit]
        has_more = offset + limit < len(rows)
        next_cursor = None
        if has_more:
            next_cursor = self.cursor.dumps(
                "list_files",
                {
                    "agent_qq": agent_qq,
                    "backend": storage.backend_name,
                    "scope": resolved_scope,
                    "root": str(root),
                    "limit": limit,
                    "offset": offset + limit,
                },
            )
        return {
            "ok": True,
            "scope": resolved_scope,
            "files": page,
            "count": len(page),
            "has_more": has_more,
            "next_cursor": next_cursor,
            "warnings": [],
        }

    @staticmethod
    def _match(filename: str, query: str) -> tuple[bool, str]:
        if not query:
            return True, "substring"
        haystack = normalized_filename(filename)
        needle = normalized_filename(query)
        if haystack == needle:
            return True, "exact"
        if haystack.startswith(needle):
            return True, "prefix"
        if needle in haystack:
            return True, "substring"
        return False, "substring"

    async def search(
        self,
        *,
        source: str | None,
        query: str | None,
        file_types: Iterable[str] | None,
        scope: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
        session: Any,
    ) -> dict[str, Any]:
        await self.ensure_ready()
        agent_qq = self.agent_qq()
        normalized_types = sorted({str(value).strip().lstrip(".").casefold() for value in (file_types or []) if str(value).strip().lstrip(".")})
        normalized_query = str(query or "")
        offset = 0
        try:
            storage = self.storage_router.active()
        except StorageError as exc:
            raise QQFileError(exc.code, str(exc), retryable=exc.retryable) from exc
        if cursor:
            try:
                state = self.cursor.loads(cursor, "search")
            except Exception as exc:
                raise QQFileError("invalid_cursor", str(exc)) from exc
            if state.get("agent_qq") != agent_qq or state.get("backend") != storage.backend_name:
                raise QQFileError("invalid_cursor", "游标对应的文件空间已经变化")
            source = str(state.get("source") or "")
            normalized_query = str(state.get("query") or "")
            normalized_types = [str(item) for item in state.get("file_types") or []]
            resolved_scope = state.get("scope")
            root = PurePosixPath(str(state.get("root") or account_file_root(agent_qq)))
            limit = int(state.get("limit") or limit)
            offset = int(state.get("offset") or 0)
        else:
            resolved_scope, root = self._resolve_scope(scope, session)
        if not normalized_query and not normalized_types:
            raise QQFileError("invalid_filters", "query 与 file_types 至少需要提供一个")
        filters = {"query": normalized_query, "file_types": normalized_types}

        if source == "local":
            rows = await self._local_rows(storage, root)
            matched: list[dict[str, Any]] = []
            for row in rows:
                if normalized_types and row["extension"] not in normalized_types:
                    continue
                accepted, match_type = self._match(row["name"], normalized_query)
                if accepted:
                    matched.append({**row, "match_type": match_type})
            rank = {"exact": 0, "prefix": 1, "substring": 2}
            matched.sort(key=lambda row: (rank[row["match_type"]], row["relative_path"]))
            key = "files"
            history_coverage = None
        elif source == "history":
            history = await self.repository.history_rows(agent_qq)
            matched = []
            current_key = str(getattr(session, "key", "") or "") if session is not None else ""
            scope_type = str(resolved_scope.get("type") or "")
            wanted_conversation = resolved_scope.get("conversation") or {}
            for row in history:
                if scope_type in {"current", "conversation"} and (
                    row["conversation_type"] != wanted_conversation.get("type")
                    or row["conversation_id"] != wanted_conversation.get("id")
                ):
                    continue
                if normalized_types and row.get("extension") not in normalized_types:
                    continue
                accepted, match_type = self._match(row["filename"], normalized_query)
                if not accepted:
                    continue
                record = await self.repository.latest_record(
                    agent_qq, row["session_key"], row["message_id"], storage.backend_name
                )
                local_file = None
                if record is not None:
                    existing = await storage.stat(PurePosixPath(record["local_path"]))
                    if existing is not None and existing.kind == "regular":
                        local_file = {
                            "path": record["local_path"],
                            "size_bytes": existing.size_bytes,
                            "downloaded_at": record["downloaded_at"],
                        }
                matched.append(
                    {
                        "message_id": row["message_id"],
                        "filename": row["filename"],
                        "extension": row.get("extension"),
                        "match_type": match_type,
                        "declared_size_bytes": row.get("size_bytes"),
                        "conversation": {
                            "type": row["conversation_type"],
                            "id": row["conversation_id"],
                            "name": row.get("conversation_name") or "",
                        },
                        "sender": {"id": row.get("sender_id") or "", "display_name": row.get("sender_name") or ""},
                        "sent_at": row.get("sent_at") or "",
                        "in_current_session": bool(current_key and current_key == row["session_key"]),
                        "local_file": local_file,
                    }
                )
            rank = {"exact": 0, "prefix": 1, "substring": 2}
            matched.sort(key=lambda row: row["sent_at"], reverse=True)
            matched.sort(key=lambda row: rank[row["match_type"]])
            key = "messages"
            history_coverage = "aicq_synced_only"
        else:
            raise QQFileError("invalid_source", "search.source 必须是 local 或 history")

        page = matched[offset : offset + limit]
        has_more = offset + limit < len(matched)
        next_cursor = None
        if has_more:
            next_cursor = self.cursor.dumps(
                "search",
                {
                    "agent_qq": agent_qq,
                    "backend": storage.backend_name,
                    "source": source,
                    "query": normalized_query,
                    "file_types": normalized_types,
                    "scope": resolved_scope,
                    "root": str(root),
                    "limit": limit,
                    "offset": offset + limit,
                },
            )
        result: dict[str, Any] = {
            "ok": True,
            "source": source,
            "scope": resolved_scope,
            "filters": filters,
            key: page,
            "count": len(page),
            "has_more": has_more,
            "next_cursor": next_cursor,
            "warnings": [],
        }
        if history_coverage:
            result["history_coverage"] = history_coverage
        return result

    async def delete(self, path: object) -> dict[str, Any]:
        await self.ensure_ready()
        agent_qq = self.agent_qq()
        try:
            logical = validate_logical_path(path, agent_qq)
        except LogicalPathError as exc:
            code = "path_outside_qq_file_root" if "根目录" in str(exc) else "invalid_path"
            raise QQFileError(code, str(exc)) from exc
        if any(part.startswith(".aicq-qq-file-") for part in logical.parts):
            raise QQFileError("protected_internal_path", "不能删除内部临时文件")
        try:
            storage = self.storage_router.active()
        except StorageError as exc:
            raise QQFileError(exc.code, str(exc), retryable=exc.retryable) from exc
        active = await self.repository.active_for_path(agent_qq, storage.backend_name, str(logical))
        if active is not None:
            raise QQFileError(
                "file_busy",
                "该路径当前由下载任务占用。请先停止对应下载任务后重试。",
                retryable=True,
                details={"blocking_download_id": active["download_id"]},
            )
        records = await self.repository.records_for_paths(agent_qq, storage.backend_name, [str(logical)])
        record = records.get(str(logical))
        async with self._path_lock:
            try:
                size = await storage.delete(logical)
            except StorageError as exc:
                raise QQFileError(exc.code, str(exc), retryable=exc.retryable) from exc
        warnings: list[dict[str, str]] = []
        was_managed: bool | None = record is not None
        try:
            await self.repository.mark_path_deleted(agent_qq, storage.backend_name, str(logical))
        except Exception:
            logger.warning("QQ 文件删除后记录同步失败: %s", logical, exc_info=True)
            was_managed = None
            record = None
            warnings.append(
                {
                    "code": "record_state_unsynchronized",
                    "message": "文件已永久删除，但本次无法读取或更新下载记录状态。",
                }
            )
        source = None
        if record is not None:
            source = {
                "message_id": record["message_id"],
                "conversation": {"type": record["conversation_type"], "id": record["conversation_id"]},
                "original_filename": record["original_filename"],
                "recorded_size_bytes": int(record["size_bytes"]),
                "downloaded_at": record["downloaded_at"],
            }
        return {
            "ok": True,
            "deleted": True,
            "path": str(logical),
            "name": logical.name,
            "size_bytes": size,
            "was_managed": was_managed,
            "source": source,
            "deleted_at": now_iso(),
            "warnings": warnings,
        }

    async def read(
        self,
        *,
        source: dict[str, Any] | None,
        selection: dict[str, Any] | None,
        cursor: str | None,
        session: Any,
    ) -> dict[str, Any]:
        await self.ensure_ready()
        agent_qq = self.agent_qq()
        try:
            storage = self.storage_router.active()
        except StorageError as exc:
            raise QQFileError(exc.code, str(exc), retryable=exc.retryable) from exc

        character_offset = 0
        if cursor:
            try:
                state = self.cursor.loads(cursor, "read")
            except Exception as exc:
                raise QQFileError("invalid_cursor", str(exc)) from exc
            if state.get("agent_qq") != agent_qq or state.get("backend") != storage.backend_name:
                raise QQFileError("cursor_scope_mismatch", "游标对应的 QQ 文件空间已经变化")
            logical = validate_logical_path(state.get("path"), agent_qq)
            selection = state.get("selection") if isinstance(state.get("selection"), dict) else None
            character_offset = int(state.get("offset") or 0)
            expected_size = int(state.get("size_bytes") or 0)
            expected_modified = int(state.get("modified_ns") or 0)
        else:
            if not isinstance(source, dict):
                raise QQFileError("internal_error", "read.source 不能为空")
            if "message_id" in source:
                started = await self.start(source.get("message_id"), session)
                if started.get("outcome") == "already_exists":
                    logical = validate_logical_path(started["file"]["local_path"], agent_qq)
                else:
                    job = started.get("job") or {}
                    if started.get("outcome") == "already_downloading":
                        observed = await self._observe_existing_job(
                            str(job.get("download_id") or ""), DOWNLOAD_OBSERVATION_SECONDS
                        )
                        if observed is not None:
                            job = self.repository.job(observed)
                    status = job.get("status")
                    if status in ACTIVE_STATUSES:
                        return {
                            "ok": True,
                            "outcome": "download_pending",
                            "download": {
                                key: job.get(key)
                                for key in (
                                    "download_id", "message_id", "status", "bytes_downloaded",
                                    "total_bytes", "progress_percent", "target_path", "updated_at",
                                )
                            },
                        }
                    if status == "failed":
                        failure = job.get("failure") or {}
                        raise QQFileError(
                            str(failure.get("code") or "internal_error"),
                            str(failure.get("message") or "QQ 文件下载失败"),
                            retryable=bool(failure.get("retryable")),
                        )
                    if status == "stopped":
                        raise QQFileError("download_stopped", "QQ 文件下载已停止")
                    logical = validate_logical_path(job.get("local_path"), agent_qq)
            elif "path" in source:
                try:
                    logical = validate_logical_path(source.get("path"), agent_qq)
                except Exception as exc:
                    raise QQFileError("path_outside_qq_file_root", str(exc)) from exc
            else:
                raise QQFileError("internal_error", "read.source 无效")
            expected_size = -1
            expected_modified = -1

        existing = await storage.stat(logical)
        if existing is None:
            raise QQFileError("not_found", "指定的 QQ 文件不存在", details={"path": str(logical)})
        if existing.kind == "symlink":
            raise QQFileError("symlink_not_allowed", "不允许读取符号链接", details={"path": str(logical)})
        if existing.kind != "regular":
            raise QQFileError("not_regular_file", "指定路径不是普通文件", details={"path": str(logical)})
        if expected_size >= 0 and (
            expected_size != existing.size_bytes or expected_modified != existing.modified_ns
        ):
            raise QQFileError("file_changed", "游标对应的文件已经发生变化", details={"path": str(logical)})

        try:
            async with self.host_file(storage, logical) as host_path:
                loop = asyncio.get_running_loop()
                parsed_envelope = await asyncio.wait_for(
                    loop.run_in_executor(
                        _PARSER_POOL,
                        parse_document_safe,
                        str(host_path),
                        selection,
                    ),
                    timeout=90.0,
                )
        except asyncio.TimeoutError as exc:
            raise QQFileError("read_limit_exceeded", "文档解析超过时间限制", retryable=True, details={"path": str(logical)}) from exc
        if not parsed_envelope.get("ok"):
            error = parsed_envelope.get("error") or {}
            details = dict(error.get("details") or {})
            details.setdefault("size_bytes", existing.size_bytes)
            raise QQFileError(
                str(error.get("code") or "parse_failed"),
                str(error.get("message") or "文档解析失败"),
                details={"path": str(logical), **details},
            )
        parsed = parsed_envelope["result"]
        full_text = str(parsed.get("text") or "")
        if character_offset < 0 or character_offset > len(full_text):
            raise QQFileError("invalid_cursor", "读取游标位置无效", details={"path": str(logical)})
        content = full_text[character_offset : character_offset + 8000]
        next_offset = character_offset + len(content)
        has_more = next_offset < len(full_text)
        base_locations = parsed.get("locations")
        if not isinstance(base_locations, list):
            base_locations = [parsed.get("location") or {}]
        locations = []
        for raw_location in base_locations:
            location = dict(raw_location or {})
            location["starts_mid_unit"] = character_offset > 0
            location["ends_mid_unit"] = has_more
            locations.append(location)
        next_cursor = None
        if has_more:
            next_cursor = self.cursor.dumps(
                "read",
                {
                    "agent_qq": agent_qq,
                    "backend": storage.backend_name,
                    "path": str(logical),
                    "selection": selection,
                    "offset": next_offset,
                    "size_bytes": existing.size_bytes,
                    "modified_ns": existing.modified_ns,
                },
            )
        return {
            "ok": True,
            "outcome": "content",
            "path": str(logical),
            "file_type": parsed["file_type"],
            "size_bytes": existing.size_bytes,
            "document": parsed["document"],
            "content": content,
            "locations": locations,
            "has_more": has_more,
            "next_cursor": next_cursor,
            "warnings": parsed.get("warnings") or [],
        }

    @asynccontextmanager
    async def host_file(self, storage: Any, logical: PurePosixPath) -> AsyncIterator[Path]:
        if isinstance(storage, HostFallbackStorage):
            yield storage.host_path(logical)
            return
        async with self.workspace_service.stage_host_file(str(logical)) as staged:
            yield Path(staged.host_path)


_SERVICES: dict[int, QQFileService] = {}


def get_qq_file_service(qq_client: Any, workspace_service: Any) -> QQFileService:
    key = id(qq_client)
    service = _SERVICES.get(key)
    if service is None or service.workspace_service is not workspace_service:
        service = QQFileService(qq_client, workspace_service)
        _SERVICES[key] = service
    return service
