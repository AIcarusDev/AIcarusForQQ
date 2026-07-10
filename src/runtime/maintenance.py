"""Runtime and data maintenance actions for the WebUI."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

import app_state
from database import DB_PATH, init_db, save_adapter_contents
from llm.session import sessions

logger = logging.getLogger("AICQ.runtime.maintenance")


RESET_COGNITION = "reset_cognition"
DELETE_LONG_TERM_MEMORY = "delete_long_term_memory"
CLEAR_ALL_DATA = "clear_all_data"


class MaintenanceError(RuntimeError):
    """User-facing maintenance action failure."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class EmergencyResetResult:
    reset_id: str
    epoch: int
    previous_focus: str | None
    cleared_flow_rounds: int
    cleared_compression_pending_jobs: int
    cleared_compression_inflight_job: bool
    woken_waits: int
    woken_sleeps: int
    main_loop_restarted: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MaintenanceActionResult:
    action: str
    maintenance_id: str
    epoch: int
    message: str
    backup_path: str | None = None
    deleted_rows: dict[str, int] = field(default_factory=dict)
    cancelled_archive_tasks: int = 0
    reset: EmergencyResetResult | None = None
    db_reinitialized: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data["ok"] = True
        return data


class MaintenanceService:
    """Central owner for dangerous runtime/data reset operations."""

    _LONG_TERM_DELETE_ORDER: tuple[str, ...] = (
        "pending_archive_jobs",
        "MemoryClusterSummaryTaskRelations",
        "MemoryClusterSummaryTaskEvents",
        "MemorySummaryCache",
        "MemoryClusterSummaryTasks",
        "MemoryClusterRevisions",
        "MemoryClusterRelations",
        "MemoryThreadStateRevisions",
        "MemoryThreadStates",
        "MemoryMounts",
        "MemoryClusterMemberRevisions",
        "MemoryClusterMembers",
        "MemoryClusters",
        "MemoryClusterRuns",
        "MemoryRelationRevisions",
        "MemoryEpisodeMembers",
        "MemoryEpisodes",
        "MemoryEventRelations",
        "MemoryEventRelationRuns",
        "MemoryEntityMergeSuspicions",
        "MemoryEntityMentions",
        "MemoryEntityAliases",
        "MemoryCanonicalEntities",
        "MemoryPreprocessRuns",
        "MemoryRelations",
        "MemoryEventSources",
        "MemoryParticipants",
        "MemoryVectors",
        "MemoryEmbeddingJobs",
        "MemoryPredicates",
        "MemoryEvents",
        "bot_memories",
        "merge_suggestions",
    )
    _LONG_TERM_SEQUENCE_NAMES: tuple[str, ...] = (
        "pending_archive_jobs",
        "MemoryPreprocessRuns",
        "MemoryEventRelationRuns",
        "MemoryClusterRuns",
        "MemoryMounts",
        "MemoryClusterSummaryTasks",
        "MemorySummaryCache",
        "MemoryRelations",
        "MemoryEventSources",
        "MemoryParticipants",
        "MemoryVectors",
        "MemoryEmbeddingJobs",
        "MemoryPredicates",
        "MemoryEvents",
    )
    _OVERVIEW_TABLES: tuple[str, ...] = (
        "MemoryEvents",
        "MemoryEventSources",
        "MemoryParticipants",
        "MemoryPredicates",
        "MemoryRelations",
        "MemoryMounts",
        "MemoryThreadStates",
        "MemoryClusterRelations",
        "MemoryClusterSummaryTasks",
        "MemorySummaryCache",
        "MemoryCanonicalEntities",
        "CognitionSources",
        "chat_sessions",
        "chat_messages",
        "bot_turns",
        "llm_usage_events",
        "adapter_state",
        "pending_archive_jobs",
    )

    def expected_confirmation(self, action: str = RESET_COGNITION) -> str:
        self_name = str(getattr(app_state, "SELF_NAME", "") or "").strip() or "AIcarus"
        if action == RESET_COGNITION:
            return f"RESET {self_name}"
        if action == DELETE_LONG_TERM_MEMORY:
            return f"DELETE MEMORY {self_name}"
        if action == CLEAR_ALL_DATA:
            return f"CLEAR DB {self_name}"
        raise MaintenanceError(f"未知维护动作: {action}")

    def describe_actions(self) -> list[dict[str, Any]]:
        core_running = not getattr(app_state, "webui_only", False) and app_state.consciousness_flow is not None
        return [
            {
                "id": RESET_COGNITION,
                "label": "重置认知",
                "danger": "medium",
                "available": core_running,
                "disabled_reason": "" if core_running else "核心未运行，无法重置认知运行时",
                "confirmation": self.expected_confirmation(RESET_COGNITION),
                "summary": "清空当前意识流、取消压缩队列，并让机器人回到无焦点等待态。",
                "keeps": "保留聊天记录、长期记忆、实体资料和统计数据。",
            },
            {
                "id": DELETE_LONG_TERM_MEMORY,
                "label": "删除全部长期记忆",
                "danger": "high",
                "available": True,
                "disabled_reason": "",
                "confirmation": self.expected_confirmation(DELETE_LONG_TERM_MEMORY),
                "summary": "删除旧事件记忆、记忆图谱、向量/检索辅助数据和待归档任务。",
                "keeps": "保留聊天记录、会话、bot_turns、实体/群资料、目标和用量统计。",
            },
            {
                "id": CLEAR_ALL_DATA,
                "label": "清空所有数据",
                "danger": "critical",
                "available": True,
                "disabled_reason": "",
                "confirmation": self.expected_confirmation(CLEAR_ALL_DATA),
                "summary": "备份后清空整个 SQLite 数据库，并重建为空库 schema。",
                "keeps": "保留配置文件、日志、模型文件和数据库备份。",
            },
        ]

    async def overview(self) -> dict[str, int]:
        result: dict[str, int] = {}
        async with aiosqlite.connect(DB_PATH) as db:
            for table in self._OVERVIEW_TABLES:
                result[table] = await self._count_table(db, table)
        return result

    async def perform(self, action: str) -> MaintenanceActionResult:
        if action == RESET_COGNITION:
            reset = await self.perform_emergency_reset()
            return MaintenanceActionResult(
                action=action,
                maintenance_id=reset.reset_id,
                epoch=reset.epoch,
                message="认知运行时已重置",
                reset=reset,
            )
        if action == DELETE_LONG_TERM_MEMORY:
            return await self.delete_long_term_memory()
        if action == CLEAR_ALL_DATA:
            return await self.clear_all_data()
        raise MaintenanceError(f"未知维护动作: {action}")

    def is_runtime_epoch_stale(self, epoch: int) -> bool:
        return int(getattr(app_state, "runtime_reset_epoch", 0)) != int(epoch)

    def make_runtime_epoch_checker(self, epoch: int):
        return lambda: self.is_runtime_epoch_stale(epoch)

    def mark_result_aborted_by_reset(self, result, epoch: int):
        result.failed = True
        result.aborted_by_runtime_reset = True
        result.runtime_reset_epoch = epoch
        return result

    async def perform_emergency_reset(self) -> EmergencyResetResult:
        if getattr(app_state, "webui_only", False) or app_state.consciousness_flow is None:
            raise MaintenanceError("核心未运行，无法执行紧急恢复")
        async with app_state.runtime_reset_lock:
            return await self._reset_cognition_locked(restart_main_loop=True, persist_flow=True)

    async def delete_long_term_memory(self) -> MaintenanceActionResult:
        async with app_state.runtime_reset_lock:
            reset: EmergencyResetResult | None = None
            core_running = not getattr(app_state, "webui_only", False) and app_state.consciousness_flow is not None
            if core_running:
                reset = await self._reset_cognition_locked(restart_main_loop=False, persist_flow=True)
            else:
                app_state.runtime_reset_epoch = int(getattr(app_state, "runtime_reset_epoch", 0)) + 1

            cancelled_archive_tasks = await self._cancel_archive_tasks()
            backup_path = await asyncio.to_thread(self._backup_database, DELETE_LONG_TERM_MEMORY)
            deleted_rows = await self._delete_long_term_memory_rows()
            self._clear_recalled_memory_cache()
            if core_running:
                self._start_main_loop()
                if reset is not None:
                    reset.main_loop_restarted = True

            return MaintenanceActionResult(
                action=DELETE_LONG_TERM_MEMORY,
                maintenance_id=uuid.uuid4().hex,
                epoch=int(getattr(app_state, "runtime_reset_epoch", 0)),
                message="长期记忆已删除",
                backup_path=backup_path,
                deleted_rows=deleted_rows,
                cancelled_archive_tasks=cancelled_archive_tasks,
                reset=reset,
            )

    async def clear_all_data(self) -> MaintenanceActionResult:
        async with app_state.runtime_reset_lock:
            maintenance_id = uuid.uuid4().hex
            reset: EmergencyResetResult | None = None
            core_running = not getattr(app_state, "webui_only", False) and app_state.consciousness_flow is not None
            if core_running:
                reset = await self._reset_cognition_locked(
                    restart_main_loop=False,
                    persist_flow=False,
                )
            else:
                app_state.runtime_reset_epoch = int(getattr(app_state, "runtime_reset_epoch", 0)) + 1

            cancelled_archive_tasks = await self._cancel_archive_tasks()
            backup_path = await asyncio.to_thread(self._backup_database, CLEAR_ALL_DATA)
            await self._drop_and_reinitialize_db()
            self._clear_process_data_caches()

            if core_running:
                from consciousness import ConsciousnessFlow

                app_state.consciousness_flow = ConsciousnessFlow()
                self._start_main_loop()
                if reset is not None:
                    reset.main_loop_restarted = True

            return MaintenanceActionResult(
                action=CLEAR_ALL_DATA,
                maintenance_id=maintenance_id,
                epoch=int(getattr(app_state, "runtime_reset_epoch", 0)),
                message="数据库已清空并重建",
                backup_path=backup_path,
                deleted_rows={},
                cancelled_archive_tasks=cancelled_archive_tasks,
                reset=reset,
                db_reinitialized=True,
            )

    async def _reset_cognition_locked(
        self,
        *,
        restart_main_loop: bool,
        persist_flow: bool,
    ) -> EmergencyResetResult:
        from consciousness import ConsciousnessFlow
        from platforms.focus import current_focus_key

        reset_id = uuid.uuid4().hex
        previous_focus = current_focus_key(app_state.current_focus)
        old_flow = app_state.consciousness_flow
        cleared_flow_rounds = old_flow.round_count if old_flow is not None else 0

        app_state.runtime_reset_epoch = int(getattr(app_state, "runtime_reset_epoch", 0)) + 1
        epoch = app_state.runtime_reset_epoch
        logger.warning(
            "[maintenance] reset cognition reset_id=%s epoch=%d previous_focus=%s",
            reset_id,
            epoch,
            previous_focus,
        )

        old_main_task = app_state.consciousness_main_task
        app_state.consciousness_main_task = None

        compression_task = app_state.cognition_compression_task
        app_state.cognition_compression_task = None
        cleared_compression_pending_jobs = len(
            getattr(app_state, "cognition_compression_pending_jobs", None) or []
        )
        cleared_compression_inflight_job = (
            getattr(app_state, "cognition_compression_inflight_job", None) is not None
        )
        app_state.cognition_compression_pending_jobs = []
        app_state.cognition_compression_inflight_job = None

        app_state.current_focus = None
        app_state.last_active_session = None
        app_state.first_input_event.clear()

        woken_waits, woken_sleeps = self._wake_and_clear_session_waits()

        await self._cancel_task(old_main_task, timeout=0.5, label="consciousness_main_loop")
        await self._cancel_task(compression_task, timeout=0.5, label="cognition_compression")

        app_state.consciousness_flow = ConsciousnessFlow()
        if persist_flow:
            contents, timestamps = app_state.consciousness_flow.dump()
            await save_adapter_contents("flow", contents, timestamps)

        app_state.shutdown_event.clear()
        app_state.first_input_event.clear()
        if restart_main_loop:
            self._start_main_loop()

        logger.warning(
            "[maintenance] reset cognition completed reset_id=%s epoch=%d waits=%d sleeps=%d",
            reset_id,
            epoch,
            woken_waits,
            woken_sleeps,
        )
        return EmergencyResetResult(
            reset_id=reset_id,
            epoch=epoch,
            previous_focus=previous_focus,
            cleared_flow_rounds=cleared_flow_rounds,
            cleared_compression_pending_jobs=cleared_compression_pending_jobs,
            cleared_compression_inflight_job=cleared_compression_inflight_job,
            woken_waits=woken_waits,
            woken_sleeps=woken_sleeps,
            main_loop_restarted=restart_main_loop,
        )

    def _start_main_loop(self) -> None:
        from consciousness import consciousness_main_loop

        app_state.shutdown_event.clear()
        app_state.first_input_event.clear()
        app_state.consciousness_main_task = asyncio.create_task(
            consciousness_main_loop(),
            name="consciousness_main_loop",
        )

    async def _cancel_task(self, task: asyncio.Task | None, *, timeout: float, label: str) -> bool:
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning("[maintenance] %s cancel timed out after %.1fs", label, timeout)
        except Exception:
            logger.debug("[maintenance] %s cancel raised", label, exc_info=True)
        return True

    async def _cancel_archive_tasks(self) -> int:
        tasks = [task for task in list(getattr(app_state, "archive_tasks", set())) if not task.done()]
        if not tasks:
            return 0
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[maintenance] archive task cancel timed out after 2.0s")
        except Exception:
            logger.debug("[maintenance] archive task cancel raised", exc_info=True)
        return len(tasks)

    def _wake_and_clear_session_waits(self) -> tuple[int, int]:
        woken_waits = 0
        woken_sleeps = 0
        for session in list(sessions.values()):
            sleep_event = getattr(session, "sleep_wake_event", None)
            if sleep_event is not None and not sleep_event.is_set():
                sleep_event.set()
                woken_sleeps += 1

            for attr, value in (
                ("sleep_pending_wake", False),
                ("sleep_pending_wake_at", 0.0),
                ("sleep_arming", False),
                ("sleep_wake_action", ""),
                ("sleep_wake_from", None),
                ("last_wake_reason", ""),
            ):
                if hasattr(session, attr):
                    setattr(session, attr, value)
            reset_transient_views = getattr(session, "reset_transient_views", None)
            if callable(reset_transient_views):
                try:
                    reset_transient_views()
                except Exception:
                    logger.debug("[maintenance] reset_transient_views failed", exc_info=True)
        return woken_waits, woken_sleeps

    def _clear_recalled_memory_cache(self) -> None:
        for session in list(sessions.values()):
            if hasattr(session, "recalled_events"):
                session.recalled_events = []
            if hasattr(session, "_nick_cache"):
                session._nick_cache = {}

    def _clear_process_data_caches(self) -> None:
        sessions.clear()
        try:
            from llm.prompt import goals as _goals

            _goals.restore([])
        except Exception:
            logger.debug("[maintenance] failed to clear in-memory goals", exc_info=True)

        app_state.current_focus = None
        app_state.last_active_session = None
        app_state.first_input_event.clear()
        self._clear_recalled_memory_cache()
        self._reset_memory_repo_flags()
        self._reset_archiver_cache(clear_signatures=True)

    def _backup_database(self, action: str) -> str | None:
        db_path = Path(DB_PATH)
        if not db_path.exists():
            return None
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = backup_dir / f"AICQ.{stamp}.{action}.db"
        with sqlite3.connect(str(db_path)) as src:
            try:
                src.execute("PRAGMA wal_checkpoint(FULL)")
            except sqlite3.DatabaseError:
                logger.debug("[maintenance] wal checkpoint before backup failed", exc_info=True)
            with sqlite3.connect(str(backup_path)) as dst:
                src.backup(dst)
        return str(backup_path)

    async def _delete_long_term_memory_rows(self) -> dict[str, int]:
        deleted: dict[str, int] = {}
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("PRAGMA foreign_keys=ON")
            for table in self._LONG_TERM_DELETE_ORDER:
                deleted[table] = await self._delete_table(db, table)
            await self._reset_sequences(db, self._LONG_TERM_SEQUENCE_NAMES)
            await self._rebuild_fts(db, "MemorySearch")
            await self._rebuild_fts(db, "MemorySearch")
            await db.commit()
        self._reset_memory_repo_flags()
        self._reset_archiver_cache(clear_signatures=False)
        return {table: count for table, count in deleted.items() if count}

    async def _drop_and_reinitialize_db(self) -> None:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("PRAGMA foreign_keys=OFF")
            await self._drop_views(db)
            await self._drop_virtual_tables(db)
            await self._drop_regular_tables(db)
            await db.commit()
            try:
                await db.execute("VACUUM")
                await db.commit()
            except Exception:
                logger.debug("[maintenance] VACUUM after DB clear failed", exc_info=True)
        self._reset_memory_repo_flags()
        await init_db()

    async def _drop_views(self, db: aiosqlite.Connection) -> None:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='view'") as cur:
            rows = [str(row[0]) for row in await cur.fetchall()]
        for name in rows:
            await db.execute(f'DROP VIEW IF EXISTS {self._quote_ident(name)}')

    async def _drop_virtual_tables(self, db: aiosqlite.Connection) -> None:
        async with db.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"
        ) as cur:
            rows = await cur.fetchall()
        names = [
            str(name)
            for name, sql in rows
            if "virtual table" in str(sql or "").lower()
        ]
        for name in names:
            if name.startswith("sqlite_"):
                continue
            await db.execute(f'DROP TABLE IF EXISTS {self._quote_ident(name)}')
        await db.commit()

    async def _drop_regular_tables(self, db: aiosqlite.Connection) -> None:
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name") as cur:
            rows = [str(row[0]) for row in await cur.fetchall()]
        for name in rows:
            if name.startswith("sqlite_"):
                continue
            await db.execute(f'DROP TABLE IF EXISTS {self._quote_ident(name)}')

    async def _delete_table(self, db: aiosqlite.Connection, table: str) -> int:
        if not await self._table_exists(db, table):
            return 0
        count = await self._count_table(db, table)
        if count:
            await db.execute(f'DELETE FROM {self._quote_ident(table)}')
        return count

    async def _count_table(self, db: aiosqlite.Connection, table: str) -> int:
        if not await self._table_exists(db, table):
            return 0
        try:
            async with db.execute(f"SELECT COUNT(*) FROM {self._quote_ident(table)}") as cur:
                row = await cur.fetchone()
            return int(row[0] or 0) if row else 0
        except Exception:
            return 0

    async def _table_exists(self, db: aiosqlite.Connection, table: str) -> bool:
        async with db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ) as cur:
            return await cur.fetchone() is not None

    async def _reset_sequences(self, db: aiosqlite.Connection, names: tuple[str, ...]) -> None:
        if not await self._table_exists(db, "sqlite_sequence"):
            return
        placeholders = ",".join("?" * len(names))
        await db.execute(f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})", list(names))

    async def _rebuild_fts(self, db: aiosqlite.Connection, table: str) -> None:
        if not await self._table_exists(db, table):
            return
        try:
            await db.execute(
                f"INSERT INTO {self._quote_ident(table)}({self._quote_ident(table)}) VALUES('rebuild')"
            )
        except Exception:
            logger.debug("[maintenance] FTS rebuild failed for %s", table, exc_info=True)

    def _reset_memory_repo_flags(self) -> None:
        try:
            import memory.repo.events as events

            events._SCHEMA_READY = False
        except Exception:
            logger.debug("[maintenance] failed to reset memory repo flags", exc_info=True)

    def _reset_archiver_cache(self, *, clear_signatures: bool) -> None:
        try:
            import memory.archive.archiver as archiver

            if clear_signatures:
                archiver._LAST_ARCHIVED_SIG.clear()
                archiver._sig_loaded = False
            else:
                archiver._sig_loaded = False
        except Exception:
            logger.debug("[maintenance] failed to reset archiver cache", exc_info=True)

    def _quote_ident(self, name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'


maintenance_service = MaintenanceService()


__all__ = [
    "CLEAR_ALL_DATA",
    "DELETE_LONG_TERM_MEMORY",
    "EmergencyResetResult",
    "MaintenanceActionResult",
    "MaintenanceError",
    "MaintenanceService",
    "RESET_COGNITION",
    "maintenance_service",
]
