"""Memory repository and recall implementation."""

from __future__ import annotations

import heapq
import html
import json
import math
import re
from collections import defaultdict
from typing import Any

from cognition_sources_schema import COGNITION_SOURCES_SCHEMA_SQL
from memory.embedding import (
    HashEmbeddingClient,
    build_embedding_client,
    dot,
    pack_vector,
    source_hash,
    unpack_vector,
)
from memory.sleep.consolidation import ensure_preprocessing_schema_async
from memory.tokenizer import build_fts_query, tokenize

from ._common import _connect, _ms, aiosqlite, logger

__all__ = [
    "ensure_schema",
    "load_events_for_recall",
    "merge_event_occurrence",
    "prefetch_candidates_for_archiver",
    "rebuild_embeddings",
    "run_embedding_backfill",
    "soft_delete_event",
    "write_event",
    "write_prompt_event",
]


_PREDICATE_THRESHOLD_DEFAULT = 0.8
_MAX_RESULTS_DEFAULT = 8
_RECENT_FALLBACK_DEFAULT = True

_BFS_MAX_ENERGY = 5.0
_BFS_MAX_DEPTH = 3
_BFS_MAX_NODES = 256
_HUB_PENALTY_WEIGHT = 0.3
_TIME_DECAY_WEIGHT = 0.15
_SUMMARY_VECTOR_WEIGHT = 0.45
_PREDICATE_VECTOR_WEIGHT = 0.25
_ENTITY_EDGE_BASE_COST = 1.0
_PREDICATE_EDGE_BASE_COST = 1.0
_SESSION_EDGE_BASE_COST = 1.4
_LEGACY_MEMORY_CONFIG_KEY = "v" + "2"

_SCHEMA_READY = False
_EMBED_CLIENT = HashEmbeddingClient()
_EMBED_CLIENT_KEY = ""


async def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    async with _connect() as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(COGNITION_SOURCES_SCHEMA_SQL)
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS MemoryEvents (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                summary_tok TEXT NOT NULL DEFAULT '',
                event_type TEXT NOT NULL,
                event_type_norm TEXT NOT NULL,
                is_negated INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'actual',
                confidence REAL NOT NULL DEFAULT 0.5,
                occurred_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL DEFAULT 0,
                last_accessed INTEGER NOT NULL DEFAULT 0,
                access_count INTEGER NOT NULL DEFAULT 0,
                occurrences INTEGER NOT NULL DEFAULT 1,
                source TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                conv_type TEXT NOT NULL DEFAULT '',
                conv_id TEXT NOT NULL DEFAULT '',
                conv_name TEXT NOT NULL DEFAULT '',
                raw_event_json TEXT NOT NULL,
                dedupe_signature TEXT NOT NULL DEFAULT '',
                is_deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memory_events_time
                ON MemoryEvents(is_deleted, occurred_at);
            CREATE INDEX IF NOT EXISTS idx_memory_events_conv
                ON MemoryEvents(conv_type, conv_id, is_deleted, occurred_at);
            CREATE INDEX IF NOT EXISTS idx_memory_events_pred
                ON MemoryEvents(event_type_norm, is_deleted);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_events_dedupe
                ON MemoryEvents(conv_type, conv_id, dedupe_signature)
                WHERE is_deleted=0 AND dedupe_signature<>'';

            CREATE TABLE IF NOT EXISTS MemoryParticipants (
                participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                entity TEXT,
                value_text TEXT,
                value_tok TEXT NOT NULL DEFAULT '',
                raw_participant_json TEXT NOT NULL,
                CHECK (entity IS NOT NULL OR value_text IS NOT NULL)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_part_event
                ON MemoryParticipants(event_id);
            CREATE INDEX IF NOT EXISTS idx_memory_part_entity
                ON MemoryParticipants(entity, role);

            CREATE TABLE IF NOT EXISTS MemoryPredicates (
                predicate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type_norm TEXT NOT NULL UNIQUE,
                display_event_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                occurrences INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS MemoryRelations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE,
                dst_event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                reason TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_memory_rel_src ON MemoryRelations(src_event_id);
            CREATE INDEX IF NOT EXISTS idx_memory_rel_dst ON MemoryRelations(dst_event_id);

            CREATE TABLE IF NOT EXISTS MemoryEventSources (
                event_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES MemoryEvents(event_id) ON DELETE CASCADE,
                source_kind TEXT NOT NULL DEFAULT 'cognition',
                source_uid TEXT NOT NULL DEFAULT '' REFERENCES CognitionSources(source_uid),
                source_id TEXT NOT NULL,
                prompt_source_id TEXT NOT NULL DEFAULT '',
                source_seq INTEGER,
                source_timestamp TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                UNIQUE(event_id, source_kind, source_id)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_sources_event
                ON MemoryEventSources(event_id);
            CREATE INDEX IF NOT EXISTS idx_memory_sources_source
                ON MemoryEventSources(source_kind, source_id);

            CREATE TABLE IF NOT EXISTS MemoryVectors (
                vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_type TEXT NOT NULL,
                owner_id INTEGER NOT NULL,
                embedding_kind TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                model TEXT NOT NULL,
                model_version TEXT NOT NULL DEFAULT '',
                normalized INTEGER NOT NULL DEFAULT 1,
                source_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(owner_type, owner_id, embedding_kind, model, model_version)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_vec_owner
                ON MemoryVectors(owner_type, owner_id, embedding_kind);

            CREATE TABLE IF NOT EXISTS MemoryEmbeddingJobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_type TEXT NOT NULL,
                owner_id INTEGER NOT NULL,
                embedding_kind TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_embed_jobs
                ON MemoryEmbeddingJobs(status, owner_type, owner_id, embedding_kind);
            """
        )
        await _ensure_schema_migrations(db)
        await ensure_preprocessing_schema_async(db)
        await _ensure_fts(db)
        await db.commit()
    _SCHEMA_READY = True


async def _ensure_schema_migrations(db: aiosqlite.Connection) -> None:
    async with db.execute("PRAGMA table_info(MemoryEventSources)") as cur:
        columns = {str(row[1]) for row in await cur.fetchall()}
    for column, ddl in (
        ("source_uid", "ALTER TABLE MemoryEventSources ADD COLUMN source_uid TEXT NOT NULL DEFAULT ''"),
        ("prompt_source_id", "ALTER TABLE MemoryEventSources ADD COLUMN prompt_source_id TEXT NOT NULL DEFAULT ''"),
    ):
        if column not in columns:
            await db.execute(ddl)
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_sources_uid ON MemoryEventSources(source_uid)"
    )
    async with db.execute("PRAGMA table_info(MemoryEvents)") as cur:
        event_columns = {str(row[1]) for row in await cur.fetchall()}
    if "access_count" not in event_columns:
        await db.execute(
            "ALTER TABLE MemoryEvents ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0"
        )


async def _ensure_fts(db: aiosqlite.Connection) -> None:
    try:
        for trigger in ("mv2_fts_insert", "mv2_fts_delete", "mv2_fts_update"):
            await db.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        await db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS MemorySearch USING fts5(
                summary_tok,
                content='MemoryEvents',
                content_rowid='event_id',
                tokenize='unicode61'
            )
            """
        )
        await db.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS memory_fts_insert
            AFTER INSERT ON MemoryEvents BEGIN
                INSERT INTO MemorySearch(rowid, summary_tok)
                VALUES (new.event_id, new.summary_tok);
            END;
            CREATE TRIGGER IF NOT EXISTS memory_fts_delete
            AFTER DELETE ON MemoryEvents BEGIN
                INSERT INTO MemorySearch(MemorySearch, rowid, summary_tok)
                VALUES ('delete', old.event_id, old.summary_tok);
            END;
            CREATE TRIGGER IF NOT EXISTS memory_fts_update
            AFTER UPDATE OF summary_tok ON MemoryEvents BEGIN
                INSERT INTO MemorySearch(MemorySearch, rowid, summary_tok)
                VALUES ('delete', old.event_id, old.summary_tok);
                INSERT INTO MemorySearch(rowid, summary_tok)
                VALUES (new.event_id, new.summary_tok);
            END;
            """
        )
        await db.execute("INSERT INTO MemorySearch(MemorySearch) VALUES ('rebuild')")
    except Exception:
        logger.warning("[memory] FTS5 unavailable; keyword recall will be degraded", exc_info=True)


async def write_prompt_event(
    event: dict[str, Any],
    *,
    raw_event_json: str = "",
    source: str = "",
    reason: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    occurred_at: int | None = None,
    supersedes: int | None = None,
    source_ids: list[str] | None = None,
    source_meta: dict[str, dict[str, str]] | None = None,
) -> int:
    return await write_event(
        event_type=str(event.get("event_type") or ""),
        summary=str(event.get("summary") or ""),
        confidence=_float(event.get("confidence"), 0.5),
        source=str(event.get("source") or source or ""),
        reason=str(event.get("reason") or reason or ""),
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
        roles=event.get("roles") if isinstance(event.get("roles"), list) else [],
        is_negated=bool(event.get("is_negated", False)),
        status=str(event.get("status") or "actual"),
        raw_event_json=raw_event_json or json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        occurred_at=occurred_at,
        supersedes=supersedes,
        source_ids=source_ids if source_ids is not None else _normalize_source_ids(event.get("source_id")),
        source_meta=source_meta,
    )


async def write_event(
    event_type: str,
    summary: str,
    summary_tok: str = "",
    modality: str = "actual",
    confidence: float = 0.5,
    context_type: str = "episodic",
    recall_scope: str = "global",
    source: str = "",
    reason: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    roles: list[dict] | None = None,
    supersedes: int | None = None,
    *,
    is_negated: bool = False,
    status: str | None = None,
    raw_event_json: str = "",
    occurred_at: int | None = None,
    source_ids: list[str] | None = None,
    source_meta: dict[str, dict[str, str]] | None = None,
) -> int:
    del modality, context_type, recall_scope
    await ensure_schema()
    now = _ms()
    summary = str(summary or "").strip()
    event_type = str(event_type or "").strip()
    if not summary or not event_type:
        raise ValueError("summary and event_type are required")
    event_type_norm = normalize_event_type(event_type)
    status_norm = normalize_status(status)
    summary_tok = summary_tok or tokenize(summary)
    occurred = int(occurred_at or now)
    norm_roles = _normalize_roles(roles or [])
    norm_source_ids = _normalize_source_ids(source_ids or [])
    dedupe_signature = _dedupe_signature(
        event_type_norm=event_type_norm,
        is_negated=bool(is_negated),
        status=status_norm,
        summary=summary,
        roles=norm_roles,
    )
    default_raw_event: dict[str, Any] = {
        "summary": summary,
        "event_type": event_type,
        "is_negated": bool(is_negated),
        "status": status_norm,
        "confidence": confidence,
        "roles": roles or [],
    }
    if norm_source_ids:
        default_raw_event["source_id"] = ",".join(norm_source_ids)
    raw_json = raw_event_json or json.dumps(
        default_raw_event,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT event_id, occurrences FROM MemoryEvents
            WHERE conv_type=? AND conv_id=? AND dedupe_signature=? AND is_deleted=0
            LIMIT 1
            """,
            (conv_type, conv_id, dedupe_signature),
        ) as cur:
            row = await cur.fetchone()
        if row:
            event_id = int(row["event_id"])
            await db.execute(
                """
                UPDATE MemoryEvents
                SET occurrences=?, last_seen_at=?, confidence=?
                WHERE event_id=?
                """,
                (int(row["occurrences"]) + 1, now, _bounded_confidence(confidence), event_id),
            )
            await _insert_event_sources(db, event_id, norm_source_ids, source_meta, now)
            await db.commit()
            return event_id

        cur = await db.execute(
            """
            INSERT INTO MemoryEvents (
                summary, summary_tok, event_type, event_type_norm, is_negated, status,
                confidence, occurred_at, created_at, last_seen_at, source, reason,
                conv_type, conv_id, conv_name, raw_event_json, dedupe_signature, is_deleted
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
            """,
            (
                summary,
                summary_tok,
                event_type,
                event_type_norm,
                1 if is_negated else 0,
                status_norm,
                _bounded_confidence(confidence),
                occurred,
                now,
                now,
                source,
                reason,
                conv_type,
                conv_id,
                conv_name,
                raw_json,
                dedupe_signature,
            ),
        )
        event_id = int(cur.lastrowid)
        for role in norm_roles:
            await db.execute(
                """
                INSERT INTO MemoryParticipants (
                    event_id, role, entity, value_text, value_tok, raw_participant_json
                ) VALUES (?,?,?,?,?,?)
                """,
                (
                    event_id,
                    role["role"],
                    role.get("entity"),
                    role.get("value_text"),
                    tokenize(role.get("value_text") or ""),
                    role["raw_participant_json"],
                ),
            )
        predicate_id = await _upsert_predicate(db, event_type_norm, event_type, now)
        if supersedes:
            await db.execute(
                """
                INSERT INTO MemoryRelations (
                    src_event_id, dst_event_id, relation_type, created_at, reason
                ) VALUES (?, ?, 'supersedes', ?, ?)
                """,
                (event_id, int(supersedes), now, reason),
            )
        await _insert_event_sources(db, event_id, norm_source_ids, source_meta, now)
        await db.commit()
        await _write_embedding(db, "event", event_id, "summary", summary)
        await _write_embedding(db, "predicate", predicate_id, "predicate", event_type_norm)
        await db.commit()
        return event_id


async def merge_event_occurrence(event_id: int) -> bool:
    await ensure_schema()
    now = _ms()
    async with _connect() as db:
        cur = await db.execute(
            """
            UPDATE MemoryEvents
            SET occurrences=occurrences+1, last_seen_at=?
            WHERE event_id=? AND is_deleted=0
            """,
            (now, int(event_id)),
        )
        await db.commit()
        return cur.rowcount > 0


async def soft_delete_event(event_id: int) -> bool:
    await ensure_schema()
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE MemoryEvents SET is_deleted=1 WHERE event_id=? AND is_deleted=0",
            (int(event_id),),
        )
        await db.commit()
        return cur.rowcount > 0


async def load_events_for_recall(
    sender_entity: str = "",
    context_scope: str = "",
    limit: int = 6,
    related_entities: list[str] | None = None,
    query: str = "",
) -> list[dict]:
    await ensure_schema()
    settings = _settings()
    limit = max(1, min(int(limit or settings["max_results"]), int(settings["max_results"])))
    related = ["self"]
    if sender_entity:
        related.append(sender_entity)
    if related_entities:
        related.extend(str(x) for x in related_entities if str(x).strip())
    related = list(dict.fromkeys(related))

    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        seeds: dict[int, float] = {}
        reasons: dict[int, set[str]] = defaultdict(set)

        for event, score in await _seed_by_text_match(db, query, context_scope, limit * 4):
            _add_seed(seeds, reasons, int(event["event_id"]), score, "text_match")
        for event, score, reason_name in await _seed_by_fts(db, query, context_scope, limit * 4):
            _add_seed(seeds, reasons, int(event["event_id"]), score, reason_name)
        for event, score in await _seed_by_entities(db, related, context_scope, limit * 4):
            _add_seed(seeds, reasons, int(event["event_id"]), score, "entity")
        for event_id, score in await _seed_by_summary_vector(db, query, context_scope, limit * 4):
            _add_seed(seeds, reasons, event_id, score, "summary_vector")
        for event_id, score in await _seed_by_predicate_vector(
            db, query, context_scope, settings["predicate_threshold"], limit * 4
        ):
            _add_seed(seeds, reasons, event_id, score, "predicate_vector")

        if not seeds and settings["recent_fallback"]:
            for event in await _recent_events(db, context_scope, limit):
                _add_seed(seeds, reasons, int(event["event_id"]), 0.15, "recent")

        if not seeds:
            return []

        graph_hits = await _expand_graph(db, list(seeds), context_scope)
        events = await _load_events(db, list(graph_hits))
        scores = await _rerank(db, events, seeds, reasons, query, graph_hits)
        top = [event for _, event in scores[:limit]]
        if top:
            ids = [int(e["event_id"]) for e in top]
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"""
                UPDATE MemoryEvents
                SET last_accessed=?, access_count=access_count + 1
                WHERE event_id IN ({placeholders})
                """,
                [_ms(), *ids],
            )
            await db.commit()
        return top


async def prefetch_candidates_for_archiver(
    sender_entity: str,
    context_scope: str,
    dialogue_text: str,
    limit: int = 8,
) -> list[dict]:
    return await load_events_for_recall(
        sender_entity=sender_entity,
        context_scope=context_scope,
        limit=limit,
        query=dialogue_text,
    )


async def rebuild_embeddings() -> dict[str, int]:
    """Mark all current event summary and predicate vectors for rebuild."""

    await ensure_schema()
    now = _ms()
    async with _connect() as db:
        await db.execute("DELETE FROM MemoryVectors")
        await db.execute("DELETE FROM MemoryEmbeddingJobs")
        event_rows = await _fetch_owner_texts(db, "event", "summary")
        pred_rows = await _fetch_owner_texts(db, "predicate", "predicate")
        for owner_type, kind, rows in (
            ("event", "summary", event_rows),
            ("predicate", "predicate", pred_rows),
        ):
            for owner_id, _text in rows:
                await db.execute(
                    """
                    INSERT INTO MemoryEmbeddingJobs (
                        owner_type, owner_id, embedding_kind, status, created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (owner_type, int(owner_id), kind, now, now),
                )
        await db.commit()
    result = await run_embedding_backfill(limit=10_000)
    result["queued_events"] = len(event_rows)
    result["queued_predicates"] = len(pred_rows)
    return result


async def run_embedding_backfill(limit: int = 100) -> dict[str, int]:
    """Process pending/failed/stale embedding jobs."""

    await ensure_schema()
    queued = await _queue_missing_or_stale_embedding_jobs()
    processed = 0
    ready = 0
    failed = 0
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT * FROM MemoryEmbeddingJobs
            WHERE status IN ('pending', 'failed', 'stale')
            ORDER BY updated_at ASC, job_id ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ) as cur:
            jobs = [dict(r) for r in await cur.fetchall()]
        work: list[tuple[dict[str, Any], str]] = []
        for job in jobs:
            text = await _owner_text(db, job["owner_type"], int(job["owner_id"]), job["embedding_kind"])
            if not text:
                await _mark_embedding_job_failed(db, int(job["job_id"]), "owner text is empty")
                failed += 1
                continue
            work.append((job, text))
        processed = len(work) + failed
        for i in range(0, len(work), 32):
            batch_jobs = work[i : i + 32]
            batch_ready, batch_failed = await _write_embedding_job_batch(db, batch_jobs)
            ready += batch_ready
            failed += batch_failed
        await db.commit()
    return {"queued": queued, "processed": processed, "ready": ready, "failed": failed}


async def _queue_missing_or_stale_embedding_jobs() -> int:
    now = _ms()
    model, model_version = _embedding_model_identity()
    queued = 0
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        owners = [
            ("event", "summary", await _fetch_owner_texts(db, "event", "summary")),
            ("predicate", "predicate", await _fetch_owner_texts(db, "predicate", "predicate")),
        ]
        for owner_type, embedding_kind, rows in owners:
            for owner_id, text in rows:
                text_hash = source_hash(text)
                async with db.execute(
                    """
                    SELECT vector_id FROM MemoryVectors
                    WHERE owner_type=? AND owner_id=? AND embedding_kind=?
                      AND model=? AND model_version=? AND source_hash=?
                    LIMIT 1
                    """,
                    (owner_type, int(owner_id), embedding_kind, model, model_version, text_hash),
                ) as cur:
                    ready_row = await cur.fetchone()
                if ready_row:
                    continue
                async with db.execute(
                    """
                    SELECT job_id FROM MemoryEmbeddingJobs
                    WHERE owner_type=? AND owner_id=? AND embedding_kind=?
                      AND status IN ('pending', 'failed', 'stale')
                    LIMIT 1
                    """,
                    (owner_type, int(owner_id), embedding_kind),
                ) as cur:
                    job_row = await cur.fetchone()
                if job_row:
                    await db.execute(
                        "UPDATE MemoryEmbeddingJobs SET status='stale', updated_at=? WHERE job_id=?",
                        (now, int(job_row["job_id"])),
                    )
                else:
                    await db.execute(
                        """
                        INSERT INTO MemoryEmbeddingJobs (
                            owner_type, owner_id, embedding_kind, status, created_at, updated_at
                        ) VALUES (?, ?, ?, 'stale', ?, ?)
                        """,
                        (owner_type, int(owner_id), embedding_kind, now, now),
                    )
                queued += 1
        await db.commit()
    return queued


def normalize_event_type(event_type: str) -> str:
    text = str(event_type or "").strip().lower()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text or "unknown"


def normalize_status(status: str | None) -> str:
    raw = str(status or "actual").strip().lower()
    return raw if raw in {"occurred", "ongoing", "future", "hypothetical", "conditional", "actual", "possible"} else "actual"


def _normalize_roles(roles: list[dict]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role in roles:
        if not isinstance(role, dict):
            continue
        role_name = str(role.get("role") or "").strip().lower()
        if not role_name:
            continue
        entity = role.get("entity")
        entity_s = str(entity).strip() if entity is not None else ""
        value = role.get("value_text")
        value_s = str(value).strip() if value is not None else ""
        if not entity_s and not value_s:
            continue
        out.append(
            {
                "role": role_name,
                "entity": entity_s or None,
                "value_text": value_s or None,
                "raw_participant_json": json.dumps(role, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return out


def _normalize_source_ids(*items: Any) -> list[str]:
    out: list[str] = []
    pending = list(items)
    while pending:
        item = pending.pop(0)
        if item is None or isinstance(item, bool):
            continue
        if isinstance(item, (list, tuple, set)):
            pending[:0] = list(item)
            continue
        if not isinstance(item, str):
            item = str(item)
        for text in re.findall(r"\d+", item):
            if text and text not in out:
                out.append(text)
    return out


def _source_seq(source_id: str) -> int | None:
    try:
        return int(str(source_id).strip())
    except (TypeError, ValueError):
        return None


async def _insert_event_sources(
    db: aiosqlite.Connection,
    event_id: int,
    source_ids: list[str],
    source_meta: dict[str, dict[str, str]] | None,
    now: int,
) -> None:
    for prompt_source_id in _normalize_source_ids(source_ids):
        meta = source_meta.get(prompt_source_id, {}) if isinstance(source_meta, dict) else {}
        source_uid = str(meta.get("source_uid") or "").strip()
        if not source_uid:
            continue
        source_timestamp = str(meta.get("timestamp") or "")
        async with db.execute(
            """
            SELECT event_source_id
            FROM MemoryEventSources
            WHERE event_id=? AND source_uid=?
            LIMIT 1
            """,
            (int(event_id), source_uid),
        ) as cur:
            row = await cur.fetchone()
        if row:
            await db.execute(
                """
                UPDATE MemoryEventSources
                SET prompt_source_id=?,
                    source_seq=?,
                    source_timestamp=CASE
                        WHEN ?<>'' THEN ?
                        ELSE source_timestamp
                    END
                WHERE event_source_id=?
                """,
                (
                    prompt_source_id,
                    _source_seq(prompt_source_id),
                    source_timestamp,
                    source_timestamp,
                    int(row[0]),
                ),
            )
            continue
        await db.execute(
            """
            INSERT INTO MemoryEventSources (
                event_id, source_kind, source_uid, source_id, prompt_source_id,
                source_seq, source_timestamp, created_at
            ) VALUES (?, 'cognition', ?, ?, ?, ?, ?, ?)
            """,
            (
                int(event_id),
                source_uid,
                source_uid,
                prompt_source_id,
                _source_seq(prompt_source_id),
                source_timestamp,
                now,
            ),
        )


def _dedupe_signature(
    *,
    event_type_norm: str,
    is_negated: bool,
    status: str,
    summary: str,
    roles: list[dict[str, Any]],
) -> str:
    role_bits = sorted(
        (
            r["role"],
            _norm_text(r.get("entity") or ""),
            _norm_text(r.get("value_text") or ""),
        )
        for r in roles
    )
    payload = json.dumps(
        {
            "p": event_type_norm,
            "n": bool(is_negated),
            "s": status,
            "m": _norm_text(summary),
            "r": role_bits,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return source_hash(payload)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip().lower())


def _bounded_confidence(value: Any) -> float:
    return max(0.0, min(1.0, _float(value, 0.5)))


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


async def _upsert_predicate(
    db: aiosqlite.Connection, event_type_norm: str, display_event_type: str, now: int
) -> int:
    await db.execute(
        """
        INSERT INTO MemoryPredicates (
            event_type_norm, display_event_type, created_at, last_seen_at, occurrences
        ) VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(event_type_norm) DO UPDATE SET
            last_seen_at=excluded.last_seen_at,
            occurrences=MemoryPredicates.occurrences+1
        """,
        (event_type_norm, display_event_type, now, now),
    )
    async with db.execute(
        "SELECT predicate_id FROM MemoryPredicates WHERE event_type_norm=?",
        (event_type_norm,),
    ) as cur:
        row = await cur.fetchone()
    return int(row[0])


async def _write_embedding(
    db: aiosqlite.Connection,
    owner_type: str,
    owner_id: int,
    embedding_kind: str,
    text: str,
) -> None:
    now = _ms()
    await db.execute(
        """
        DELETE FROM MemoryEmbeddingJobs
        WHERE owner_type=? AND owner_id=? AND embedding_kind=? AND status!='ready'
        """,
        (owner_type, int(owner_id), embedding_kind),
    )
    cur = await db.execute(
        """
        INSERT INTO MemoryEmbeddingJobs (
            owner_type, owner_id, embedding_kind, status, created_at, updated_at
        ) VALUES (?, ?, ?, 'pending', ?, ?)
        """,
        (owner_type, int(owner_id), embedding_kind, now, now),
    )
    job_id = int(cur.lastrowid)
    job = {
        "job_id": job_id,
        "owner_type": owner_type,
        "owner_id": int(owner_id),
        "embedding_kind": embedding_kind,
    }
    await _write_embedding_for_job(db, job, text)


async def _write_embedding_job_batch(
    db: aiosqlite.Connection,
    jobs_and_texts: list[tuple[dict[str, Any], str]],
) -> tuple[int, int]:
    if not jobs_and_texts:
        return 0, 0
    now = _ms()
    try:
        texts = [text for _job, text in jobs_and_texts]
        batch = _embedding_client().embed_texts(texts)
        if len(batch.vectors) != len(jobs_and_texts):
            raise ValueError(f"embedding count mismatch: {len(batch.vectors)} != {len(jobs_and_texts)}")
    except Exception as exc:
        for job, _text in jobs_and_texts:
            await _mark_embedding_job_failed(db, int(job["job_id"]), str(exc))
        return 0, len(jobs_and_texts)

    ready = 0
    failed = 0
    for (job, text), vector in zip(jobs_and_texts, batch.vectors):
        try:
            await db.execute(
                """
                INSERT OR REPLACE INTO MemoryVectors (
                    owner_type, owner_id, embedding_kind, embedding, dim, model,
                    model_version, normalized, source_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(job["owner_type"]),
                    int(job["owner_id"]),
                    str(job["embedding_kind"]),
                    pack_vector(vector),
                    batch.dim,
                    batch.model,
                    batch.model_version,
                    1 if batch.normalized else 0,
                    source_hash(text),
                    now,
                ),
            )
            await db.execute(
                "UPDATE MemoryEmbeddingJobs SET status='ready', updated_at=? WHERE job_id=?",
                (now, int(job["job_id"])),
            )
            ready += 1
        except Exception as exc:
            await _mark_embedding_job_failed(db, int(job["job_id"]), str(exc))
            failed += 1
    return ready, failed


async def _write_embedding_for_job(
    db: aiosqlite.Connection,
    job: dict[str, Any],
    text: str,
) -> bool:
    now = _ms()
    job_id = int(job["job_id"])
    owner_type = str(job["owner_type"])
    owner_id = int(job["owner_id"])
    embedding_kind = str(job["embedding_kind"])
    try:
        batch = _embedding_client().embed_texts([text])
        vector = batch.vectors[0]
        await db.execute(
            """
            INSERT OR REPLACE INTO MemoryVectors (
                owner_type, owner_id, embedding_kind, embedding, dim, model,
                model_version, normalized, source_hash, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                owner_type,
                int(owner_id),
                embedding_kind,
                pack_vector(vector),
                batch.dim,
                batch.model,
                batch.model_version,
                1 if batch.normalized else 0,
                source_hash(text),
                now,
            ),
        )
        await db.execute(
            "UPDATE MemoryEmbeddingJobs SET status='ready', updated_at=? WHERE job_id=?",
            (now, job_id),
        )
        return True
    except Exception as exc:
        await _mark_embedding_job_failed(db, job_id, str(exc))
        logger.debug("[memory] embedding failed for %s#%s", owner_type, owner_id, exc_info=True)
        return False


async def _mark_embedding_job_failed(
    db: aiosqlite.Connection, job_id: int, error: str
) -> None:
    await db.execute(
        """
        UPDATE MemoryEmbeddingJobs
        SET status='failed', retry_count=retry_count+1, last_error=?, updated_at=?
        WHERE job_id=?
        """,
        (str(error)[:1000], _ms(), int(job_id)),
    )


async def _seed_by_fts(
    db: aiosqlite.Connection, query: str, context_scope: str, limit: int
) -> list[tuple[dict, float, str]]:
    fts_q = build_fts_query(query or "")
    if not fts_q:
        return []
    scope_sql, params = _scope_clause(context_scope, alias="")
    try:
        async with db.execute(
            f"""
            SELECT e.*, bm25(MemorySearch) AS rank
            FROM MemorySearch
            JOIN MemoryEvents e ON e.event_id=MemorySearch.rowid
            WHERE MemorySearch MATCH ? AND e.is_deleted=0 {scope_sql}
            ORDER BY rank
            LIMIT ?
            """,
            [fts_q, *params, int(limit)],
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    except Exception:
        logger.debug("[memory] FTS query failed: %r", fts_q, exc_info=True)
        return []
    if not rows:
        return []
    ranks = [float(r.get("rank") or 0.0) for r in rows]
    r_min, r_max = min(ranks), max(ranks)
    span = (r_max - r_min) or 1.0
    return [(r, 1.0 - ((float(r.get("rank") or 0.0) - r_min) / span), "fts") for r in rows]


def _text_match_terms(query: str) -> list[str]:
    terms: list[str] = []
    for part in re.split(r"[\s,，。！？!?;；:：、/\\|]+", str(query or "").strip()):
        part = part.strip()
        if len(part) >= 2:
            terms.append(part)
    return list(dict.fromkeys(terms))[:8]


async def _seed_by_text_match(
    db: aiosqlite.Connection, query: str, context_scope: str, limit: int
) -> list[tuple[dict, float]]:
    terms = _text_match_terms(query)
    if not terms:
        return []
    scope_sql, params = _scope_clause(context_scope, alias="e")
    clauses: list[str] = []
    args: list[str] = []
    for term in terms:
        like = f"%{term}%"
        clauses.append(
            """
            (
              e.summary LIKE ? OR e.event_type LIKE ? OR e.event_type_norm LIKE ?
              OR EXISTS (
                SELECT 1 FROM MemoryParticipants p
                WHERE p.event_id=e.event_id
                  AND (p.entity LIKE ? OR p.value_text LIKE ?)
              )
            )
            """
        )
        args.extend([like, like, like, like, like])
    sql = f"""
        SELECT e.*
        FROM MemoryEvents e
        WHERE e.is_deleted=0 {scope_sql}
          AND ({' OR '.join(clauses)})
        ORDER BY e.occurred_at DESC
        LIMIT ?
    """
    async with db.execute(sql, [*params, *args, int(limit)]) as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    out: list[tuple[dict, float]] = []
    for row in rows:
        haystack = " ".join(
            str(row.get(k) or "")
            for k in ("summary", "event_type", "event_type_norm", "conv_name")
        )
        matched = sum(1 for term in terms if term in haystack)
        out.append((row, min(1.2, 0.85 + matched * 0.1)))
    return out


async def _seed_by_entities(
    db: aiosqlite.Connection, entities: list[str], context_scope: str, limit: int
) -> list[tuple[dict, float]]:
    ents = [e for e in dict.fromkeys(str(x).strip() for x in entities) if e]
    if not ents:
        return []
    placeholders = ",".join("?" * len(ents))
    scope_sql, params = _scope_clause(context_scope)
    out: dict[int, tuple[dict, float]] = {}
    async with db.execute(
        f"""
        SELECT DISTINCT e.*
        FROM MemoryEvents e
        JOIN MemoryParticipants p ON p.event_id=e.event_id
        WHERE e.is_deleted=0 AND p.entity IN ({placeholders}) {scope_sql}
        ORDER BY e.occurred_at DESC
        LIMIT ?
        """,
        [*ents, *params, int(limit)],
    ) as cur:
        for row in await cur.fetchall():
            event = dict(row)
            out[int(event["event_id"])] = (event, 0.55)
    fuzzy_terms = [e for e in ents if len(e) >= 2]
    for term in fuzzy_terms:
        async with db.execute(
            f"""
            SELECT DISTINCT e.*
            FROM MemoryEvents e
            JOIN MemoryParticipants p ON p.event_id=e.event_id
            WHERE e.is_deleted=0 AND p.entity IS NOT NULL
              AND p.entity LIKE ? AND p.entity NOT IN ({placeholders}) {scope_sql}
            ORDER BY e.occurred_at DESC
            LIMIT ?
            """,
            [f"%{term}%", *ents, *params, int(limit)],
        ) as cur:
            for row in await cur.fetchall():
                event = dict(row)
                out.setdefault(int(event["event_id"]), (event, 0.38))
    return list(out.values())[:limit]


async def _seed_by_summary_vector(
    db: aiosqlite.Connection, query: str, context_scope: str, limit: int
) -> list[tuple[int, float]]:
    qvec = _query_vector(query)
    if qvec is None:
        return []
    rows = await _vector_rows(db, "event", "summary", context_scope)
    scored: list[tuple[int, float]] = []
    for row in rows:
        try:
            sim = dot(qvec, unpack_vector(row["embedding"], int(row["dim"])))
        except Exception:
            continue
        scored.append((int(row["owner_id"]), sim * _SUMMARY_VECTOR_WEIGHT))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


async def _seed_by_predicate_vector(
    db: aiosqlite.Connection,
    query: str,
    context_scope: str,
    threshold: float,
    limit: int,
) -> list[tuple[int, float]]:
    qvec = _query_vector(query)
    if qvec is None:
        return []
    rows = await _vector_rows(db, "predicate", "predicate", "")
    predicate_ids: list[int] = []
    predicate_scores: dict[int, float] = {}
    for row in rows:
        try:
            sim = dot(qvec, unpack_vector(row["embedding"], int(row["dim"])))
        except Exception:
            continue
        if sim >= threshold:
            pid = int(row["owner_id"])
            predicate_ids.append(pid)
            predicate_scores[pid] = sim
    if not predicate_ids:
        return []
    placeholders = ",".join("?" * len(predicate_ids))
    scope_sql, params = _scope_clause(context_scope)
    async with db.execute(
        f"""
        SELECT e.event_id, p.predicate_id
        FROM MemoryPredicates p
        JOIN MemoryEvents e ON e.event_type_norm=p.event_type_norm
        WHERE p.predicate_id IN ({placeholders}) AND e.is_deleted=0 {scope_sql}
        LIMIT ?
        """,
        [*predicate_ids, *params, int(limit)],
    ) as cur:
        out = [
            (
                int(row["event_id"]),
                predicate_scores.get(int(row["predicate_id"]), 0.0) * _PREDICATE_VECTOR_WEIGHT,
            )
            for row in await cur.fetchall()
        ]
    out.sort(key=lambda item: item[1], reverse=True)
    return out


async def _recent_events(
    db: aiosqlite.Connection, context_scope: str, limit: int
) -> list[dict]:
    scope_sql, params = _scope_clause(context_scope, alias="")
    async with db.execute(
        f"""
        SELECT * FROM MemoryEvents
        WHERE is_deleted=0 AND lower(status) NOT IN ('hypothetical','conditional','future') {scope_sql}
        ORDER BY occurred_at DESC
        LIMIT ?
        """,
        [*params, int(limit)],
    ) as cur:
        return [dict(r) for r in await cur.fetchall()]


async def _vector_rows(
    db: aiosqlite.Connection, owner_type: str, embedding_kind: str, context_scope: str
) -> list[dict]:
    model, model_version = _embedding_model_identity()
    if owner_type == "event":
        scope_sql, params = _scope_clause(context_scope, alias="e")
        sql = f"""
            SELECT v.*, e.summary AS owner_text
            FROM MemoryVectors v
            JOIN MemoryEvents e ON e.event_id=v.owner_id
            WHERE v.owner_type='event' AND v.embedding_kind=?
              AND v.model=? AND v.model_version=?
              AND e.is_deleted=0 {scope_sql}
        """
        args = [embedding_kind, model, model_version, *params]
    else:
        sql = """
            SELECT v.*, p.event_type_norm AS owner_text
            FROM MemoryVectors v
            JOIN MemoryPredicates p ON p.predicate_id=v.owner_id
            WHERE v.owner_type=? AND v.embedding_kind=?
              AND v.model=? AND v.model_version=?
        """
        args = [owner_type, embedding_kind, model, model_version]
    async with db.execute(sql, args) as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    return [
        row
        for row in rows
        if str(row.get("source_hash") or "") == source_hash(str(row.get("owner_text") or ""))
    ]


def _query_vector(query: str) -> list[float] | None:
    if not isinstance(query, str) or not query.strip():
        return None
    try:
        return _embedding_client().embed_texts([query]).vectors[0]
    except Exception:
        return None


def _embedding_model_identity() -> tuple[str, str]:
    client = _embedding_client()
    return str(getattr(client, "model", "") or ""), str(getattr(client, "model_version", "") or "")


def _embedding_client():
    global _EMBED_CLIENT, _EMBED_CLIENT_KEY
    try:
        import app_state

        cfg = getattr(app_state, "config", {}) or {}
        memory = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
        legacy = memory.get(_LEGACY_MEMORY_CONFIG_KEY, {}) if isinstance(memory, dict) else {}
        emb_src = memory.get("embedding", {}) if isinstance(memory, dict) else {}
        if (not isinstance(emb_src, dict) or not emb_src) and isinstance(legacy, dict):
            emb_src = legacy.get("embedding", {})
        emb_cfg = dict(emb_src if isinstance(emb_src, dict) else {})
        provider = str(emb_cfg.get("provider") or "hash")
        if provider != "hash" and "provider_config" not in emb_cfg:
            providers = cfg.get("model_providers", {}) if isinstance(cfg, dict) else {}
            if isinstance(providers, dict) and provider in providers:
                emb_cfg["provider_config"] = dict(providers[provider] or {})
    except Exception:
        emb_cfg = {"provider": "hash"}
    key = json.dumps(emb_cfg, sort_keys=True, ensure_ascii=False)
    if key != _EMBED_CLIENT_KEY:
        try:
            _EMBED_CLIENT = build_embedding_client(emb_cfg)
        except Exception:
            logger.warning("[memory] embedding config invalid; fallback to hash", exc_info=True)
            _EMBED_CLIENT = HashEmbeddingClient()
            key = "hash-fallback"
        _EMBED_CLIENT_KEY = key
    return _EMBED_CLIENT


async def _fetch_owner_texts(
    db: aiosqlite.Connection, owner_type: str, embedding_kind: str
) -> list[tuple[int, str]]:
    if owner_type == "event" and embedding_kind == "summary":
        async with db.execute(
            "SELECT event_id, summary FROM MemoryEvents WHERE is_deleted=0"
        ) as cur:
            return [(int(r[0]), str(r[1] or "")) for r in await cur.fetchall()]
    if owner_type == "predicate" and embedding_kind == "predicate":
        async with db.execute(
            "SELECT predicate_id, event_type_norm FROM MemoryPredicates"
        ) as cur:
            return [(int(r[0]), str(r[1] or "")) for r in await cur.fetchall()]
    return []


async def _owner_text(
    db: aiosqlite.Connection, owner_type: str, owner_id: int, embedding_kind: str
) -> str:
    if owner_type == "event" and embedding_kind == "summary":
        async with db.execute(
            "SELECT summary FROM MemoryEvents WHERE event_id=? AND is_deleted=0",
            (int(owner_id),),
        ) as cur:
            row = await cur.fetchone()
        return str(row[0] or "") if row else ""
    if owner_type == "predicate" and embedding_kind == "predicate":
        async with db.execute(
            "SELECT event_type_norm FROM MemoryPredicates WHERE predicate_id=?",
            (int(owner_id),),
        ) as cur:
            row = await cur.fetchone()
        return str(row[0] or "") if row else ""
    return ""


async def _expand_graph(
    db: aiosqlite.Connection, seed_event_ids: list[int], context_scope: str
) -> dict[int, dict[str, Any]]:
    seed_set = set(int(x) for x in seed_event_ids)
    events = await _load_events(db, seed_event_ids)
    roles = await _load_roles(db, list(events))
    entities = {r["entity"] for role_list in roles.values() for r in role_list if r.get("entity")}
    predicates = {events[eid]["event_type_norm"] for eid in events}
    if entities:
        placeholders = ",".join("?" * len(entities))
        scope_sql, params = _scope_clause(context_scope)
        async with db.execute(
            f"""
            SELECT DISTINCT e.*
            FROM MemoryEvents e
            JOIN MemoryParticipants p ON p.event_id=e.event_id
            WHERE e.is_deleted=0 AND p.entity IN ({placeholders}) {scope_sql}
            LIMIT 160
            """,
            [*entities, *params],
        ) as cur:
            for row in await cur.fetchall():
                events[int(row["event_id"])] = dict(row)
    if predicates:
        placeholders = ",".join("?" * len(predicates))
        scope_sql, params = _scope_clause(context_scope, alias="")
        async with db.execute(
            f"""
            SELECT * FROM MemoryEvents
            WHERE is_deleted=0 AND event_type_norm IN ({placeholders}) {scope_sql}
            LIMIT 160
            """,
            [*predicates, *params],
        ) as cur:
            for row in await cur.fetchall():
                events[int(row["event_id"])] = dict(row)
    roles = await _load_roles(db, list(events))
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    entity_degree: dict[str, int] = defaultdict(int)
    value_degree: dict[str, int] = defaultdict(int)
    pred_degree: dict[str, int] = defaultdict(int)
    session_degree: dict[str, int] = defaultdict(int)
    status_by_event = {
        int(eid): str(event.get("status") or "actual").strip().lower()
        for eid, event in events.items()
    }
    now = _ms()
    for eid, ev_roles in roles.items():
        event_node = f"E:{eid}"
        pred_node = f"P:{events[eid]['event_type_norm']}"
        edge_age_cost = _event_age_cost(events[eid], now)
        pred_degree[pred_node] += 1
        _link(adj, event_node, pred_node, _PREDICATE_EDGE_BASE_COST + edge_age_cost)
        conv_type = str(events[eid].get("conv_type") or "")
        conv_id = str(events[eid].get("conv_id") or "")
        if conv_type or conv_id:
            session_node = f"C:{conv_type}:{conv_id}"
            session_degree[session_node] += 1
            _link(adj, event_node, session_node, _SESSION_EDGE_BASE_COST + edge_age_cost)
        for role in ev_roles:
            ent = role.get("entity")
            if ent:
                node = f"N:{ent}"
                entity_degree[node] += 1
                _link(adj, event_node, node, _ENTITY_EDGE_BASE_COST + edge_age_cost)
            value = str(role.get("value_text") or "").strip()
            if value:
                node = f"V:{value}"
                value_degree[node] += 1
                _link(adj, event_node, node, _ENTITY_EDGE_BASE_COST + 0.2 + edge_age_cost)
    await _add_relation_edges(db, adj, events, context_scope)
    await _add_similar_predicate_edges(db, adj, {str(events[eid]["event_type_norm"]) for eid in events})
    for node, degree in {**entity_degree, **value_degree, **pred_degree, **session_degree}.items():
        if degree <= 1:
            continue
        for i, (dst, cost) in enumerate(adj.get(node, [])):
            adj[node][i] = (dst, cost + _HUB_PENALTY_WEIGHT * math.log10(degree + 1))

    pq: list[tuple[float, int, str]] = []
    dist: dict[str, float] = {}
    depth: dict[str, int] = {}
    parent: dict[str, str] = {}
    for eid in seed_set:
        node = f"E:{eid}"
        dist[node] = 0.0
        depth[node] = 0
        heapq.heappush(pq, (0.0, 0, node))
    expanded = 0
    while pq and expanded < _BFS_MAX_NODES:
        cost, dep, node = heapq.heappop(pq)
        if cost > _BFS_MAX_ENERGY or cost != dist.get(node):
            continue
        expanded += 1
        if dep >= _BFS_MAX_DEPTH:
            continue
        for nxt, edge_cost in adj.get(node, []):
            if not _status_traversal_allowed(node, nxt, status_by_event, seed_set):
                continue
            next_cost = cost + edge_cost
            if next_cost > _BFS_MAX_ENERGY:
                continue
            if next_cost < dist.get(nxt, float("inf")):
                dist[nxt] = next_cost
                depth[nxt] = dep + 1
                parent[nxt] = node
                heapq.heappush(pq, (next_cost, dep + 1, nxt))
    out: dict[int, dict[str, Any]] = {}
    for node, cost in dist.items():
        if not node.startswith("E:"):
            continue
        event_id = int(node[2:])
        out[event_id] = {
            "path_cost": float(cost),
            "path_depth": int(depth.get(node, 0)),
            "path": _reconstruct_path(node, parent),
        }
    for event_id in seed_set:
        out.setdefault(event_id, {"path_cost": 0.0, "path_depth": 0, "path": [f"E:{event_id}"]})
    return out


def _event_age_cost(event: dict[str, Any], now: int) -> float:
    try:
        age_days = max(0.0, (now - int(event.get("occurred_at") or now)) / 86_400_000)
    except Exception:
        age_days = 0.0
    return min(0.6, math.log10(age_days + 1.0) * _TIME_DECAY_WEIGHT)


def _status_traversal_allowed(
    node: str,
    nxt: str,
    status_by_event: dict[int, str],
    seed_set: set[int],
) -> bool:
    guarded = {"hypothetical", "conditional", "future"}
    if not node.startswith("E:") and not nxt.startswith("E:"):
        return True
    cur_status = ""
    next_status = ""
    if node.startswith("E:"):
        cur_status = status_by_event.get(int(node[2:]), "actual")
    if nxt.startswith("E:"):
        next_event_id = int(nxt[2:])
        next_status = status_by_event.get(next_event_id, "actual")
        if next_status in guarded and next_event_id not in seed_set:
            return False
    if cur_status in guarded and next_status and next_status != cur_status:
        return False
    return True


async def _add_relation_edges(
    db: aiosqlite.Connection,
    adj: dict[str, list[tuple[str, float]]],
    events: dict[int, dict],
    context_scope: str,
) -> None:
    if not events:
        return
    placeholders = ",".join("?" * len(events))
    scope_sql, params = _scope_clause(context_scope, alias="e")
    async with db.execute(
        f"""
        SELECT r.src_event_id, r.dst_event_id, r.relation_type, e.*
        FROM MemoryRelations r
        JOIN MemoryEvents e ON e.event_id=r.dst_event_id
        WHERE r.src_event_id IN ({placeholders}) AND e.is_deleted=0 {scope_sql}
        """,
        [*events, *params],
    ) as cur:
        rows = await cur.fetchall()
    for row in rows:
        src = int(row["src_event_id"])
        dst = int(row["dst_event_id"])
        if dst not in events:
            events[dst] = dict(row)
        relation_type = str(row["relation_type"] or "")
        cost = 0.8 if relation_type in {"merge_into", "supersedes"} else 1.2
        _link(adj, f"E:{src}", f"E:{dst}", cost)


def _reconstruct_path(node: str, parent: dict[str, str]) -> list[str]:
    path = [node]
    while node in parent:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


async def _add_similar_predicate_edges(
    db: aiosqlite.Connection,
    adj: dict[str, list[tuple[str, float]]],
    predicates: set[str],
) -> None:
    if not predicates:
        return
    settings = _settings()
    threshold = float(settings["predicate_threshold"])
    rows = await _vector_rows(db, "predicate", "predicate", "")
    vectors: dict[str, list[float]] = {}
    pid_to_pred: dict[int, str] = {}
    if not rows:
        return
    predicate_ids = [int(r["owner_id"]) for r in rows]
    placeholders = ",".join("?" * len(predicate_ids))
    async with db.execute(
        f"""
        SELECT predicate_id, event_type_norm
        FROM MemoryPredicates
        WHERE predicate_id IN ({placeholders})
        """,
        predicate_ids,
    ) as cur:
        for row in await cur.fetchall():
            pid_to_pred[int(row["predicate_id"])] = str(row["event_type_norm"])
    for row in rows:
        pred = pid_to_pred.get(int(row["owner_id"]))
        if not pred:
            continue
        try:
            vectors[pred] = unpack_vector(row["embedding"], int(row["dim"]))
        except Exception:
            continue
    if not vectors:
        return
    frontier = set(predicates)
    for pred in list(frontier):
        vec = vectors.get(pred)
        if vec is None:
            continue
        scored: list[tuple[float, str]] = []
        for other, other_vec in vectors.items():
            if other == pred:
                continue
            sim = dot(vec, other_vec)
            if sim >= threshold:
                scored.append((sim, other))
        scored.sort(reverse=True)
        for sim, other in scored[:8]:
            cost = _PREDICATE_EDGE_BASE_COST + max(0.0, 1.0 - sim)
            _link(adj, f"P:{pred}", f"P:{other}", cost)
            await _attach_events_for_predicate(db, adj, other)


async def _attach_events_for_predicate(
    db: aiosqlite.Connection,
    adj: dict[str, list[tuple[str, float]]],
    event_type_norm: str,
) -> None:
    async with db.execute(
        """
        SELECT event_id, status FROM MemoryEvents
        WHERE event_type_norm=? AND is_deleted=0
        ORDER BY occurred_at DESC
        LIMIT 50
        """,
        (event_type_norm,),
    ) as cur:
        rows = await cur.fetchall()
    pred_node = f"P:{event_type_norm}"
    for row in rows:
        if str(row["status"] or "actual").strip().lower() == "hypothetical":
            continue
        _link(adj, pred_node, f"E:{int(row[0])}", _PREDICATE_EDGE_BASE_COST)


async def _load_events(db: aiosqlite.Connection, event_ids: list[int]) -> dict[int, dict]:
    ids = [int(x) for x in dict.fromkeys(event_ids)]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT * FROM MemoryEvents WHERE event_id IN ({placeholders}) AND is_deleted=0",
        ids,
    ) as cur:
        return {int(r["event_id"]): dict(r) for r in await cur.fetchall()}


async def _load_roles(db: aiosqlite.Connection, event_ids: list[int]) -> dict[int, list[dict]]:
    ids = [int(x) for x in dict.fromkeys(event_ids)]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT * FROM MemoryParticipants WHERE event_id IN ({placeholders})",
        ids,
    ) as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    out: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        out[int(row["event_id"])].append(row)
    return out


async def _rerank(
    db: aiosqlite.Connection,
    events: dict[int, dict],
    seeds: dict[int, float],
    reasons: dict[int, set[str]],
    query: str,
    graph_hits: dict[int, dict[str, Any]],
) -> list[tuple[float, dict]]:
    del db, query
    now = _ms()
    scored: list[tuple[float, dict]] = []
    for event_id, event in events.items():
        score = seeds.get(event_id, 0.0)
        graph = graph_hits.get(event_id, {})
        path_cost = float(graph.get("path_cost", 0.0) or 0.0)
        path_depth = int(graph.get("path_depth", 0) or 0)
        if event_id not in seeds:
            score += max(0.0, 0.35 - path_cost * 0.08)
        else:
            score += max(0.0, 0.12 - path_cost * 0.03)
        age_days = max(0.0, (now - int(event.get("occurred_at") or now)) / 86_400_000)
        score += max(0.0, 0.2 - min(0.2, age_days / 365.0))
        score += min(0.16, max(0, int(event.get("occurrences") or 1) - 1) * 0.02)
        if str(event.get("status") or "actual").lower() == "hypothetical":
            score -= 0.35 if event_id not in seeds else 0.12
        if len(str(event.get("summary") or "")) < 8:
            score -= 0.15
        score -= min(0.18, path_depth * 0.03)
        event["recall_reasons"] = sorted(reasons.get(event_id, []))
        event["recall_score"] = round(score, 6)
        event["recall_path_cost"] = round(path_cost, 6)
        event["recall_path_depth"] = path_depth
        event["recall_path"] = list(graph.get("path") or [f"E:{event_id}"])
        scored.append((score, event))
    scored.sort(
        key=lambda item: (
            item[0],
            int(item[1].get("occurred_at") or 0),
            int(item[1].get("event_id") or 0),
        ),
        reverse=True,
    )
    return scored


def _add_seed(
    seeds: dict[int, float], reasons: dict[int, set[str]], event_id: int, score: float, reason: str
) -> None:
    seeds[event_id] = max(seeds.get(event_id, 0.0), float(score))
    reasons[event_id].add(reason)


def _link(adj: dict[str, list[tuple[str, float]]], a: str, b: str, cost: float) -> None:
    adj[a].append((b, cost))
    adj[b].append((a, cost))


def _scope_clause(context_scope: str, alias: str = "e") -> tuple[str, list[str]]:
    if not context_scope or ":" not in context_scope:
        return "", []
    conv_type, conv_id = context_scope.split(":", 1)
    conv_id = conv_id.removeprefix("qq_")
    prefix = f"{alias}." if alias else ""
    return f"AND (({prefix}conv_type='' AND {prefix}conv_id='') OR {prefix}conv_type='flow' OR ({prefix}conv_type=? AND {prefix}conv_id=?))", [
        conv_type,
        conv_id,
    ]


def _settings() -> dict[str, Any]:
    try:
        import app_state

        cfg = getattr(app_state, "config", {}) or {}
        memory = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
        legacy = memory.get(_LEGACY_MEMORY_CONFIG_KEY, {}) if isinstance(memory, dict) else {}
    except Exception:
        memory = {}
        legacy = {}
    if not isinstance(memory, dict):
        memory = {}
    if not isinstance(legacy, dict):
        legacy = {}
    return {
        "predicate_threshold": max(
            0.5,
            min(
                0.95,
                _float(
                    memory.get(
                        "memory_predicate_similarity_threshold",
                        legacy.get("memory_predicate_similarity_threshold"),
                    ),
                    _PREDICATE_THRESHOLD_DEFAULT,
                ),
            ),
        ),
        "max_results": max(
            1,
            min(
                30,
                int(memory.get("memory_recall_max_results", legacy.get("memory_recall_max_results", _MAX_RESULTS_DEFAULT))),
            ),
        ),
        "recent_fallback": bool(
            memory.get(
                "memory_recall_recent_fallback",
                legacy.get("memory_recall_recent_fallback", _RECENT_FALLBACK_DEFAULT),
            )
        ),
    }


def escape_summary(summary: str) -> str:
    return html.escape(str(summary or ""))
