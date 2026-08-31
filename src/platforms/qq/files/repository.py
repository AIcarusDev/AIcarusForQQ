"""SQLite persistence for QQ file messages, downloads and committed files."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import aiosqlite

from database import DB_PATH


ACTIVE_STATUSES = ("queued", "resolving", "downloading", "verifying")
TERMINAL_STATUSES = ("completed", "failed", "stopped")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class QQFileRepository:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path

    async def _connect(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path, timeout=30.0)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA busy_timeout=30000")
        return db

    @staticmethod
    def job(row: aiosqlite.Row | dict[str, Any]) -> dict[str, Any]:
        value = dict(row)
        total = value.get("total_bytes")
        downloaded = int(value.get("bytes_downloaded") or 0)
        progress = None
        if total is not None:
            total = int(total)
            progress = 100.0 if total == 0 else round(min(100.0, downloaded * 100.0 / total), 2)
        failure = None
        if value.get("status") == "failed":
            failure = {
                "code": value.get("failure_code") or "internal_error",
                "message": value.get("failure_message") or "下载失败",
                "retryable": bool(value.get("failure_retryable")),
            }
        return {
            "download_id": value["download_id"],
            "message_id": value["message_id"],
            "conversation": {
                "type": value["conversation_type"],
                "id": value["conversation_id"],
            },
            "original_filename": value["original_filename"],
            "status": value["status"],
            "bytes_downloaded": downloaded,
            "total_bytes": total,
            "progress_percent": progress,
            "target_path": value["target_path"],
            "local_path": value.get("local_path") if value.get("status") == "completed" else None,
            "created_at": value["created_at"],
            "updated_at": value["updated_at"],
            "finished_at": value.get("finished_at"),
            "failure": failure,
        }

    async def recover_interrupted(self, agent_qq: str) -> None:
        timestamp = now_iso()
        db = await self._connect()
        try:
            await db.execute(
                """UPDATE qq_file_downloads
                   SET status='failed', failure_code='download_interrupted',
                       failure_message='下载因服务重启而中断。', failure_retryable=1,
                       updated_at=?, finished_at=?
                   WHERE agent_qq=?
                     AND status IN ('queued','resolving','downloading','verifying')""",
                (timestamp, timestamp, agent_qq),
            )
            await db.commit()
        finally:
            await db.close()

    async def create_job(self, values: dict[str, Any]) -> dict[str, Any]:
        timestamp = now_iso()
        row = {
            "download_id": values.get("download_id") or f"qfd_{uuid.uuid4().hex}",
            "status": "queued",
            "bytes_downloaded": 0,
            "local_path": None,
            "failure_code": None,
            "failure_message": None,
            "failure_retryable": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "finished_at": None,
            **values,
        }
        columns = (
            "download_id", "agent_qq", "session_key", "message_id", "conversation_type",
            "conversation_id", "original_filename", "source_file_id", "status",
            "bytes_downloaded", "total_bytes", "target_path", "local_path",
            "storage_backend", "storage_relpath", "failure_code", "failure_message",
            "failure_retryable", "created_at", "updated_at", "finished_at",
        )
        db = await self._connect()
        try:
            await db.execute(
                f"INSERT INTO qq_file_downloads ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(row.get(column) for column in columns),
            )
            await db.commit()
        finally:
            await db.close()
        return row

    async def update_job(self, download_id: str, **changes: Any) -> dict[str, Any] | None:
        allowed = {
            "status", "bytes_downloaded", "total_bytes", "local_path", "failure_code",
            "failure_message", "failure_retryable", "finished_at", "updated_at",
        }
        values = {key: value for key, value in changes.items() if key in allowed}
        values["updated_at"] = values.get("updated_at") or now_iso()
        db = await self._connect()
        try:
            assignments = ", ".join(f"{key}=?" for key in values)
            await db.execute(
                f"UPDATE qq_file_downloads SET {assignments} WHERE download_id=?",
                (*values.values(), download_id),
            )
            await db.commit()
            async with db.execute("SELECT * FROM qq_file_downloads WHERE download_id=?", (download_id,)) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def get_job_row(self, download_id: str, *, agent_qq: str | None = None) -> dict[str, Any] | None:
        db = await self._connect()
        try:
            if agent_qq:
                query = "SELECT * FROM qq_file_downloads WHERE download_id=? AND agent_qq=?"
                params = (download_id, agent_qq)
            else:
                query = "SELECT * FROM qq_file_downloads WHERE download_id=?"
                params = (download_id,)
            async with db.execute(query, params) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def find_active(self, agent_qq: str, session_key: str, message_id: str) -> dict[str, Any] | None:
        db = await self._connect()
        try:
            async with db.execute(
                """SELECT * FROM qq_file_downloads
                   WHERE agent_qq=? AND session_key=? AND message_id=?
                     AND status IN ('queued','resolving','downloading','verifying')
                   ORDER BY created_at DESC LIMIT 1""",
                (agent_qq, session_key, message_id),
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def list_jobs(
        self,
        agent_qq: str,
        statuses: Iterable[str] | None,
        offset: int,
        limit: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        selected = tuple(dict.fromkeys(statuses or ()))
        active_filter = [status for status in selected if status in ACTIVE_STATUSES] if selected else list(ACTIVE_STATUSES)
        terminal_filter = [status for status in selected if status in TERMINAL_STATUSES] if selected else list(TERMINAL_STATUSES)
        db = await self._connect()
        try:
            active: list[dict[str, Any]] = []
            if active_filter:
                marks = ",".join("?" for _ in active_filter)
                async with db.execute(
                    f"SELECT * FROM qq_file_downloads WHERE agent_qq=? AND status IN ({marks}) ORDER BY created_at ASC",
                    (agent_qq, *active_filter),
                ) as cur:
                    active = [dict(row) for row in await cur.fetchall()]
            terminal: list[dict[str, Any]] = []
            has_more = False
            if terminal_filter:
                marks = ",".join("?" for _ in terminal_filter)
                async with db.execute(
                    f"""SELECT * FROM qq_file_downloads WHERE agent_qq=? AND status IN ({marks})
                        ORDER BY finished_at DESC, created_at DESC LIMIT ? OFFSET ?""",
                    (agent_qq, *terminal_filter, limit + 1, offset),
                ) as cur:
                    rows = [dict(row) for row in await cur.fetchall()]
                has_more = len(rows) > limit
                terminal = rows[:limit]
            return active, terminal, has_more
        finally:
            await db.close()

    async def latest_record(self, agent_qq: str, session_key: str, message_id: str, backend: str) -> dict[str, Any] | None:
        db = await self._connect()
        try:
            async with db.execute(
                """SELECT * FROM qq_file_records
                   WHERE agent_qq=? AND session_key=? AND message_id=?
                     AND storage_backend=? AND deleted_at IS NULL
                   ORDER BY downloaded_at DESC LIMIT 1""",
                (agent_qq, session_key, message_id, backend),
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def add_record(self, job_row: dict[str, Any], size_bytes: int) -> dict[str, Any]:
        timestamp = now_iso()
        record = {
            "record_id": f"qfr_{uuid.uuid4().hex}",
            "agent_qq": job_row["agent_qq"],
            "session_key": job_row["session_key"],
            "message_id": job_row["message_id"],
            "conversation_type": job_row["conversation_type"],
            "conversation_id": job_row["conversation_id"],
            "original_filename": job_row["original_filename"],
            "local_path": job_row["target_path"],
            "storage_backend": job_row["storage_backend"],
            "storage_relpath": job_row["storage_relpath"],
            "size_bytes": int(size_bytes),
            "downloaded_at": timestamp,
            "deleted_at": None,
        }
        db = await self._connect()
        try:
            await db.execute(
                """INSERT INTO qq_file_records
                   (record_id, agent_qq, session_key, message_id, conversation_type,
                    conversation_id, original_filename, local_path, storage_backend,
                    storage_relpath, size_bytes, downloaded_at, deleted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(record.values()),
            )
            await db.commit()
        finally:
            await db.close()
        return record

    async def records_for_paths(self, agent_qq: str, backend: str, paths: Iterable[str]) -> dict[str, dict[str, Any]]:
        normalized = tuple(dict.fromkeys(str(path) for path in paths))
        if not normalized:
            return {}
        marks = ",".join("?" for _ in normalized)
        db = await self._connect()
        try:
            async with db.execute(
                f"""SELECT * FROM qq_file_records
                    WHERE agent_qq=? AND storage_backend=? AND deleted_at IS NULL
                      AND local_path IN ({marks}) ORDER BY downloaded_at DESC""",
                (agent_qq, backend, *normalized),
            ) as cur:
                rows = await cur.fetchall()
        finally:
            await db.close()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = dict(row)
            result.setdefault(value["local_path"], value)
        return result

    async def mark_path_deleted(self, agent_qq: str, backend: str, path: str) -> None:
        db = await self._connect()
        try:
            await db.execute(
                """UPDATE qq_file_records SET deleted_at=?
                   WHERE agent_qq=? AND storage_backend=? AND local_path=? AND deleted_at IS NULL""",
                (now_iso(), agent_qq, backend, path),
            )
            await db.commit()
        finally:
            await db.close()

    async def active_for_path(self, agent_qq: str, backend: str, path: str) -> dict[str, Any] | None:
        db = await self._connect()
        try:
            async with db.execute(
                """SELECT * FROM qq_file_downloads
                   WHERE agent_qq=? AND storage_backend=? AND target_path=?
                     AND status IN ('queued','resolving','downloading','verifying') LIMIT 1""",
                (agent_qq, backend, path),
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None
        finally:
            await db.close()

    async def history_rows(self, agent_qq: str) -> list[dict[str, Any]]:
        db = await self._connect()
        try:
            async with db.execute(
                """SELECT m.*, COALESCE(NULLIF(s.focus_name, ''), NULLIF(s.conv_name, ''), '') AS conversation_name
                   FROM qq_file_messages AS m
                   LEFT JOIN chat_sessions AS s ON s.session_key=m.session_key
                   WHERE m.agent_qq=?""",
                (agent_qq,),
            ) as cur:
                return [dict(row) for row in await cur.fetchall()]
        finally:
            await db.close()
