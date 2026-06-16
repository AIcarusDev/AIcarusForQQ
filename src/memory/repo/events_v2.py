"""Memory V2 repository and recall implementation."""

from __future__ import annotations

import heapq
import html
import json
import math
import re
from collections import defaultdict
from typing import Any

from memory.embedding_v2 import (
    HashEmbeddingClient,
    dot,
    pack_vector,
    source_hash,
    unpack_vector,
)
from memory.tokenizer import build_fts_query, tokenize

from ._common import _connect, _ms, aiosqlite, logger

__all__ = [
    "ensure_schema",
    "load_events_for_recall",
    "merge_event_occurrence",
    "prefetch_candidates_for_archiver",
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

_SCHEMA_READY = False
_EMBED_CLIENT = HashEmbeddingClient()


async def ensure_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    async with _connect() as db:
        await db.execute("PRAGMA foreign_keys=ON")
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS MemoryV2Events (
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
            CREATE INDEX IF NOT EXISTS idx_mv2_events_time
                ON MemoryV2Events(is_deleted, occurred_at);
            CREATE INDEX IF NOT EXISTS idx_mv2_events_conv
                ON MemoryV2Events(conv_type, conv_id, is_deleted, occurred_at);
            CREATE INDEX IF NOT EXISTS idx_mv2_events_pred
                ON MemoryV2Events(event_type_norm, is_deleted);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mv2_events_dedupe
                ON MemoryV2Events(conv_type, conv_id, dedupe_signature)
                WHERE is_deleted=0 AND dedupe_signature<>'';

            CREATE TABLE IF NOT EXISTS MemoryV2Participants (
                participant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES MemoryV2Events(event_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                entity TEXT,
                value_text TEXT,
                value_tok TEXT NOT NULL DEFAULT '',
                raw_participant_json TEXT NOT NULL,
                CHECK (entity IS NOT NULL OR value_text IS NOT NULL)
            );
            CREATE INDEX IF NOT EXISTS idx_mv2_part_event
                ON MemoryV2Participants(event_id);
            CREATE INDEX IF NOT EXISTS idx_mv2_part_entity
                ON MemoryV2Participants(entity, role);

            CREATE TABLE IF NOT EXISTS MemoryV2Predicates (
                predicate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type_norm TEXT NOT NULL UNIQUE,
                display_event_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                occurrences INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS MemoryV2Relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                src_event_id INTEGER NOT NULL REFERENCES MemoryV2Events(event_id) ON DELETE CASCADE,
                dst_event_id INTEGER NOT NULL REFERENCES MemoryV2Events(event_id) ON DELETE CASCADE,
                relation_type TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                reason TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_mv2_rel_src ON MemoryV2Relations(src_event_id);
            CREATE INDEX IF NOT EXISTS idx_mv2_rel_dst ON MemoryV2Relations(dst_event_id);

            CREATE TABLE IF NOT EXISTS MemoryV2Vectors (
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
            CREATE INDEX IF NOT EXISTS idx_mv2_vec_owner
                ON MemoryV2Vectors(owner_type, owner_id, embedding_kind);

            CREATE TABLE IF NOT EXISTS MemoryV2EmbeddingJobs (
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
            CREATE INDEX IF NOT EXISTS idx_mv2_embed_jobs
                ON MemoryV2EmbeddingJobs(status, owner_type, owner_id, embedding_kind);
            """
        )
        await _ensure_fts(db)
        await db.commit()
    _SCHEMA_READY = True


async def _ensure_fts(db: aiosqlite.Connection) -> None:
    try:
        await db.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS MemoryV2Search USING fts5(
                summary_tok,
                content='MemoryV2Events',
                content_rowid='event_id',
                tokenize='unicode61'
            )
            """
        )
        await db.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS mv2_fts_insert
            AFTER INSERT ON MemoryV2Events BEGIN
                INSERT INTO MemoryV2Search(rowid, summary_tok)
                VALUES (new.event_id, new.summary_tok);
            END;
            CREATE TRIGGER IF NOT EXISTS mv2_fts_delete
            AFTER DELETE ON MemoryV2Events BEGIN
                INSERT INTO MemoryV2Search(MemoryV2Search, rowid, summary_tok)
                VALUES ('delete', old.event_id, old.summary_tok);
            END;
            CREATE TRIGGER IF NOT EXISTS mv2_fts_update
            AFTER UPDATE OF summary_tok ON MemoryV2Events BEGIN
                INSERT INTO MemoryV2Search(MemoryV2Search, rowid, summary_tok)
                VALUES ('delete', old.event_id, old.summary_tok);
                INSERT INTO MemoryV2Search(rowid, summary_tok)
                VALUES (new.event_id, new.summary_tok);
            END;
            """
        )
    except Exception:
        logger.warning("[memory_v2] FTS5 unavailable; keyword recall will be degraded", exc_info=True)


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
    dedupe_signature = _dedupe_signature(
        event_type_norm=event_type_norm,
        is_negated=bool(is_negated),
        status=status_norm,
        summary=summary,
        roles=norm_roles,
    )
    raw_json = raw_event_json or json.dumps(
        {
            "summary": summary,
            "event_type": event_type,
            "is_negated": bool(is_negated),
            "status": status_norm,
            "confidence": confidence,
            "roles": roles or [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT event_id, occurrences FROM MemoryV2Events
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
                UPDATE MemoryV2Events
                SET occurrences=?, last_seen_at=?, confidence=?
                WHERE event_id=?
                """,
                (int(row["occurrences"]) + 1, now, _bounded_confidence(confidence), event_id),
            )
            await db.commit()
            return event_id

        cur = await db.execute(
            """
            INSERT INTO MemoryV2Events (
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
                INSERT INTO MemoryV2Participants (
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
                INSERT INTO MemoryV2Relations (
                    src_event_id, dst_event_id, relation_type, created_at, reason
                ) VALUES (?, ?, 'supersedes', ?, ?)
                """,
                (event_id, int(supersedes), now, reason),
            )
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
            UPDATE MemoryV2Events
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
            "UPDATE MemoryV2Events SET is_deleted=1 WHERE event_id=? AND is_deleted=0",
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

        event_ids = await _expand_graph(db, list(seeds), context_scope)
        events = await _load_events(db, event_ids)
        scores = await _rerank(db, events, seeds, reasons, query)
        top = [event for _, event in scores[:limit]]
        if top:
            ids = [int(e["event_id"]) for e in top]
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"UPDATE MemoryV2Events SET last_accessed=? WHERE event_id IN ({placeholders})",
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


def normalize_event_type(event_type: str) -> str:
    text = str(event_type or "").strip().lower()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text or "unknown"


def normalize_status(status: str | None) -> str:
    raw = str(status or "actual").strip().lower()
    return raw if raw in {"actual", "possible", "hypothetical"} else "actual"


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
        INSERT INTO MemoryV2Predicates (
            event_type_norm, display_event_type, created_at, last_seen_at, occurrences
        ) VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(event_type_norm) DO UPDATE SET
            last_seen_at=excluded.last_seen_at,
            occurrences=MemoryV2Predicates.occurrences+1
        """,
        (event_type_norm, display_event_type, now, now),
    )
    async with db.execute(
        "SELECT predicate_id FROM MemoryV2Predicates WHERE event_type_norm=?",
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
    cur = await db.execute(
        """
        INSERT INTO MemoryV2EmbeddingJobs (
            owner_type, owner_id, embedding_kind, status, created_at, updated_at
        ) VALUES (?, ?, ?, 'pending', ?, ?)
        """,
        (owner_type, int(owner_id), embedding_kind, now, now),
    )
    job_id = int(cur.lastrowid)
    try:
        batch = _EMBED_CLIENT.embed_texts([text])
        vector = batch.vectors[0]
        await db.execute(
            """
            INSERT OR REPLACE INTO MemoryV2Vectors (
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
            "UPDATE MemoryV2EmbeddingJobs SET status='ready', updated_at=? WHERE job_id=?",
            (now, job_id),
        )
    except Exception as exc:
        await db.execute(
            """
            UPDATE MemoryV2EmbeddingJobs
            SET status='failed', retry_count=retry_count+1, last_error=?, updated_at=?
            WHERE job_id=?
            """,
            (str(exc)[:1000], now, job_id),
        )
        logger.debug("[memory_v2] embedding failed for %s#%s", owner_type, owner_id, exc_info=True)


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
            SELECT e.*, bm25(MemoryV2Search) AS rank
            FROM MemoryV2Search
            JOIN MemoryV2Events e ON e.event_id=MemoryV2Search.rowid
            WHERE MemoryV2Search MATCH ? AND e.is_deleted=0 {scope_sql}
            ORDER BY rank
            LIMIT ?
            """,
            [fts_q, *params, int(limit)],
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    except Exception:
        logger.debug("[memory_v2] FTS query failed: %r", fts_q, exc_info=True)
        return []
    if not rows:
        return []
    ranks = [float(r.get("rank") or 0.0) for r in rows]
    r_min, r_max = min(ranks), max(ranks)
    span = (r_max - r_min) or 1.0
    return [(r, 1.0 - ((float(r.get("rank") or 0.0) - r_min) / span), "fts") for r in rows]


async def _seed_by_entities(
    db: aiosqlite.Connection, entities: list[str], context_scope: str, limit: int
) -> list[tuple[dict, float]]:
    ents = [e for e in dict.fromkeys(str(x).strip() for x in entities) if e]
    if not ents:
        return []
    placeholders = ",".join("?" * len(ents))
    scope_sql, params = _scope_clause(context_scope)
    async with db.execute(
        f"""
        SELECT DISTINCT e.*
        FROM MemoryV2Events e
        JOIN MemoryV2Participants p ON p.event_id=e.event_id
        WHERE e.is_deleted=0 AND p.entity IN ({placeholders}) {scope_sql}
        ORDER BY e.occurred_at DESC
        LIMIT ?
        """,
        [*ents, *params, int(limit)],
    ) as cur:
        return [(dict(r), 0.55) for r in await cur.fetchall()]


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
        FROM MemoryV2Predicates p
        JOIN MemoryV2Events e ON e.event_type_norm=p.event_type_norm
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
    scope_sql, params = _scope_clause(context_scope)
    async with db.execute(
        f"""
        SELECT * FROM MemoryV2Events
        WHERE is_deleted=0 {scope_sql}
        ORDER BY occurred_at DESC
        LIMIT ?
        """,
        [*params, int(limit)],
    ) as cur:
        return [dict(r) for r in await cur.fetchall()]


async def _vector_rows(
    db: aiosqlite.Connection, owner_type: str, embedding_kind: str, context_scope: str
) -> list[dict]:
    if owner_type == "event":
        scope_sql, params = _scope_clause(context_scope, alias="e")
        sql = f"""
            SELECT v.*
            FROM MemoryV2Vectors v
            JOIN MemoryV2Events e ON e.event_id=v.owner_id
            WHERE v.owner_type='event' AND v.embedding_kind=?
              AND e.is_deleted=0 {scope_sql}
        """
        args = [embedding_kind, *params]
    else:
        sql = """
            SELECT * FROM MemoryV2Vectors
            WHERE owner_type=? AND embedding_kind=?
        """
        args = [owner_type, embedding_kind]
    async with db.execute(sql, args) as cur:
        return [dict(r) for r in await cur.fetchall()]


def _query_vector(query: str) -> list[float] | None:
    if not isinstance(query, str) or not query.strip():
        return None
    try:
        return _EMBED_CLIENT.embed_texts([query]).vectors[0]
    except Exception:
        return None


async def _expand_graph(
    db: aiosqlite.Connection, seed_event_ids: list[int], context_scope: str
) -> list[int]:
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
            FROM MemoryV2Events e
            JOIN MemoryV2Participants p ON p.event_id=e.event_id
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
            SELECT * FROM MemoryV2Events
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
    pred_degree: dict[str, int] = defaultdict(int)
    for eid, ev_roles in roles.items():
        event_node = f"E:{eid}"
        pred_node = f"P:{events[eid]['event_type_norm']}"
        pred_degree[pred_node] += 1
        _link(adj, event_node, pred_node, _PREDICATE_EDGE_BASE_COST)
        for role in ev_roles:
            ent = role.get("entity")
            if ent:
                node = f"N:{ent}"
                entity_degree[node] += 1
                _link(adj, event_node, node, _ENTITY_EDGE_BASE_COST)
    for node, degree in {**entity_degree, **pred_degree}.items():
        if degree <= 1:
            continue
        for i, (dst, cost) in enumerate(adj.get(node, [])):
            adj[node][i] = (dst, cost + _HUB_PENALTY_WEIGHT * math.log10(degree + 1))

    pq: list[tuple[float, int, str]] = []
    dist: dict[str, float] = {}
    depth: dict[str, int] = {}
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
            next_cost = cost + edge_cost
            if next_cost > _BFS_MAX_ENERGY:
                continue
            if next_cost < dist.get(nxt, float("inf")):
                dist[nxt] = next_cost
                depth[nxt] = dep + 1
                heapq.heappush(pq, (next_cost, dep + 1, nxt))
    event_ids = {int(node[2:]) for node in dist if node.startswith("E:")}
    return list(seed_set | event_ids)


async def _load_events(db: aiosqlite.Connection, event_ids: list[int]) -> dict[int, dict]:
    ids = [int(x) for x in dict.fromkeys(event_ids)]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT * FROM MemoryV2Events WHERE event_id IN ({placeholders}) AND is_deleted=0",
        ids,
    ) as cur:
        return {int(r["event_id"]): dict(r) for r in await cur.fetchall()}


async def _load_roles(db: aiosqlite.Connection, event_ids: list[int]) -> dict[int, list[dict]]:
    ids = [int(x) for x in dict.fromkeys(event_ids)]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"SELECT * FROM MemoryV2Participants WHERE event_id IN ({placeholders})",
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
) -> list[tuple[float, dict]]:
    del db, query
    now = _ms()
    scored: list[tuple[float, dict]] = []
    for event_id, event in events.items():
        score = seeds.get(event_id, 0.0)
        age_days = max(0.0, (now - int(event.get("occurred_at") or now)) / 86_400_000)
        score += max(0.0, 0.2 - min(0.2, age_days / 365.0))
        score += min(0.16, max(0, int(event.get("occurrences") or 1) - 1) * 0.02)
        if str(event.get("status") or "actual").lower() == "hypothetical":
            score -= 0.2
        if len(str(event.get("summary") or "")) < 8:
            score -= 0.15
        event["recall_reasons"] = sorted(reasons.get(event_id, []))
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
    return f"AND (({prefix}conv_type='' AND {prefix}conv_id='') OR ({prefix}conv_type=? AND {prefix}conv_id=?))", [
        conv_type,
        conv_id,
    ]


def _settings() -> dict[str, Any]:
    try:
        import app_state

        cfg = getattr(app_state, "config", {}) or {}
        memory = cfg.get("memory", {}) if isinstance(cfg, dict) else {}
        v2 = memory.get("v2", {}) if isinstance(memory, dict) else {}
    except Exception:
        v2 = {}
    return {
        "predicate_threshold": max(
            0.5,
            min(0.95, _float(v2.get("memory_predicate_similarity_threshold"), _PREDICATE_THRESHOLD_DEFAULT)),
        ),
        "max_results": max(1, min(30, int(v2.get("memory_recall_max_results", _MAX_RESULTS_DEFAULT)))),
        "recent_fallback": bool(v2.get("memory_recall_recent_fallback", _RECENT_FALLBACK_DEFAULT)),
    }


def escape_summary(summary: str) -> str:
    return html.escape(str(summary or ""))
