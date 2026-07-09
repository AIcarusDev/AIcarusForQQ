"""Recall helpers for ready Memory cluster summaries."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

from ..sleep.consolidation import cluster_summary_from_json
from memory.repo._common import _connect, _ms, aiosqlite


_TERM_RE = re.compile(r"[\w\u4e00-\u9fff]{2,}", re.UNICODE)


async def load_ready_summaries_covering_events(
    *,
    event_ids: Iterable[int],
    sender_entity: str = "",
    context_scope: str = "",
    query: str = "",
    limit: int = 16,
    max_scan: int = 1024,
) -> list[dict[str, Any]]:
    """Return ready cluster summaries for clusters containing recalled events."""

    from memory.repo.events import ensure_schema
    from ..sleep.summary_worker import summary_id_for_source

    wanted_ids = {int(item) for item in event_ids if _positive_int(item)}
    if not wanted_ids:
        return []

    await ensure_schema()
    terms = _query_terms(query)
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        cluster_members = await _load_cluster_members_covering_events(db, wanted_ids)
        if not cluster_members:
            return []
        summary_ids = {
            summary_id_for_source("cluster", cluster_id)
            for cluster_id in cluster_members
            if cluster_id
        }
        rows = await _ready_summary_rows_by_summary_id(db, summary_ids, max(1, int(max_scan or 1)))
        if not rows:
            return []

        card_records: list[tuple[dict[str, Any], Any, set[int]]] = []
        all_event_ids: set[int] = set()
        for row in rows:
            try:
                card = cluster_summary_from_json(str(row["cluster_summary_json"] or "{}"))
            except Exception:
                continue
            if not card.summary_id or not card.short_summary:
                continue
            cluster_id = str(card.source_id or "")
            source_ids = set(cluster_members.get(cluster_id, set()))
            if not cluster_id or not source_ids.intersection(wanted_ids):
                continue
            all_event_ids.update(source_ids)
            card_records.append((dict(row), card, source_ids))

        event_meta = await _load_event_meta(db, sorted(all_event_ids))
        relation_counts = await _load_active_relation_counts(
            db,
            {card.source_id or card.summary_id for _row, card, _ids in card_records},
        )

    scored: list[tuple[float, dict[str, Any]]] = []
    for row, card, source_ids in card_records:
        if not _matches_scope(source_ids, event_meta, context_scope):
            continue
        text = " ".join(
            [
                card.title,
                card.short_summary,
                *card.core_entities,
                *card.confirmed_claims,
                *card.uncertain_claims,
                *card.disputed_claims,
                card.current_state,
            ]
        )
        score, reasons = _summary_score(
            text=text,
            terms=terms,
            card_entities=card.core_entities,
            sender_entity=sender_entity,
            event_ids=source_ids,
            related_event_ids=wanted_ids,
            relation_count=relation_counts.get(card.source_id or card.summary_id, 0),
            updated_at_ms=int(row.get("updated_at_ms") or 0),
        )
        if score <= 0.0:
            continue
        reasons = sorted(set(reasons) | {"summary:replaces_cluster_member"})
        occurred_at = max(
            (int(event_meta.get(event_id, {}).get("occurred_at") or 0) for event_id in source_ids),
            default=int(row.get("updated_at_ms") or 0),
        )
        item = _summary_recall_item(
            row=row,
            card=card,
            event_ids=source_ids,
            relation_count=relation_counts.get(card.source_id or card.summary_id, 0),
            recall_score=score,
            recall_reasons=reasons,
            occurred_at=occurred_at,
        )
        scored.append((_replacement_sort_score(item, wanted_ids), item))

    scored.sort(
        key=lambda pair: (
            pair[0],
            int(pair[1].get("occurred_at") or 0),
            str(pair[1].get("summary_id") or ""),
        ),
        reverse=True,
    )
    return [item for _score, item in scored[: max(1, int(limit or 1))]]


async def _ready_summary_rows_by_summary_id(
    db: aiosqlite.Connection,
    summary_ids: set[str],
    limit: int,
) -> list[aiosqlite.Row]:
    ids = sorted(str(item) for item in summary_ids if str(item or "").strip())
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    async with db.execute(
        f"""
        SELECT *
        FROM MemorySummaryCache
        WHERE status='ready'
          AND cluster_summary_json <> '{{}}'
          AND summary_id IN ({placeholders})
        ORDER BY updated_at_ms DESC, summary_id DESC
        LIMIT ?
        """,
        [*ids, max(1, int(limit or 1))],
    ) as cur:
        return list(await cur.fetchall())


async def _load_cluster_members_covering_events(
    db: aiosqlite.Connection,
    event_ids: set[int],
) -> dict[str, set[int]]:
    if not event_ids:
        return {}
    placeholders = ",".join("?" * len(event_ids))
    try:
        async with db.execute(
            f"""
            WITH target_clusters AS (
                SELECT DISTINCT cluster_id
                FROM MemoryClusterMembers
                WHERE status='active' AND event_id IN ({placeholders})
            )
            SELECT m.cluster_id, m.event_id
            FROM target_clusters t
            JOIN MemoryClusters c ON c.cluster_id=t.cluster_id AND c.status='active'
            JOIN MemoryClusterMembers m ON m.cluster_id=t.cluster_id AND m.status='active'
            ORDER BY m.cluster_id ASC, m.rank ASC, m.event_id ASC
            """,
            sorted(event_ids),
        ) as cur:
            rows = await cur.fetchall()
    except Exception:
        return {}
    members: dict[str, set[int]] = {}
    for row in rows:
        cluster_id = str(row["cluster_id"] or "")
        event_id = int(row["event_id"] or 0)
        if cluster_id and event_id > 0:
            members.setdefault(cluster_id, set()).add(event_id)
    return members


async def _load_event_meta(db: aiosqlite.Connection, event_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not event_ids:
        return {}
    placeholders = ",".join("?" * len(event_ids))
    async with db.execute(
        f"""
        SELECT event_id, conv_type, conv_id, occurred_at, created_at
        FROM MemoryEvents
        WHERE event_id IN ({placeholders}) AND is_deleted=0
        """,
        event_ids,
    ) as cur:
        return {int(row["event_id"]): dict(row) for row in await cur.fetchall()}


async def _load_active_relation_counts(
    db: aiosqlite.Connection,
    cluster_ids: set[str],
) -> dict[str, int]:
    if not cluster_ids:
        return {}
    placeholders = ",".join("?" * len(cluster_ids))
    try:
        async with db.execute(
            f"""
            SELECT cluster_id, COUNT(*) AS n
            FROM MemoryClusterRelations
            WHERE cluster_id IN ({placeholders}) AND status IN ('active', 'weak')
            GROUP BY cluster_id
            """,
            sorted(cluster_ids),
        ) as cur:
            return {str(row["cluster_id"]): int(row["n"] or 0) for row in await cur.fetchall()}
    except Exception:
        return {}


def _summary_score(
    *,
    text: str,
    terms: list[str],
    card_entities: Iterable[str],
    sender_entity: str,
    event_ids: set[int],
    related_event_ids: set[int],
    relation_count: int,
    updated_at_ms: int,
) -> tuple[float, list[str]]:
    normalized_text = text.lower()
    reasons: set[str] = set()
    score = 0.0
    if terms:
        matched = [term for term in terms if term.lower() in normalized_text]
        if matched:
            score += min(1.25, 0.48 + 0.14 * len(matched))
            reasons.add("summary:text_match")
    overlap = event_ids & related_event_ids
    if overlap:
        score += min(0.95, 0.45 + 0.08 * len(overlap))
        reasons.add("summary:source_event_overlap")
    if sender_entity:
        sender_key = _entity_key(sender_entity)
        if sender_key and sender_key in {_entity_key(entity) for entity in card_entities}:
            score += 0.24
            reasons.add("summary:sender_entity")
    if relation_count > 0:
        score += min(0.24, math.log1p(relation_count) * 0.08)
        reasons.add("summary:solidified_relation")
    if updated_at_ms > 0:
        age_days = max(0.0, (_ms() - updated_at_ms) / 86_400_000)
        score += max(0.0, 0.12 - min(0.12, age_days / 120.0))
    if not reasons:
        return 0.0, []
    return score, sorted(reasons)


def _summary_recall_item(
    *,
    row: dict[str, Any],
    card: Any,
    event_ids: set[int],
    relation_count: int,
    recall_score: float,
    recall_reasons: list[str],
    occurred_at: int,
) -> dict[str, Any]:
    return {
        "memory_kind": "summary",
        "event_id": f"summary:{card.summary_id}",
        "summary_id": card.summary_id,
        "source_kind": card.source_kind,
        "source_id": card.source_id,
        "summary": card.short_summary,
        "event_type": "cluster_summary",
        "event_type_norm": "cluster_summary",
        "status": "summary",
        "confidence": min(0.95, 0.66 + 0.03 * min(6, relation_count)),
        "occurred_at": occurred_at,
        "created_at": int(row.get("created_at_ms") or row.get("updated_at_ms") or 0),
        "updated_at": int(row.get("updated_at_ms") or 0),
        "roles": [{"role": "core_entity", "entity": entity} for entity in card.core_entities],
        "recall_score": round(recall_score, 6),
        "recall_reasons": recall_reasons,
        "recall_path": [f"S:{card.summary_id}"],
        "recall_path_cost": 0.0,
        "recall_path_depth": 0,
        "source_event_ids": sorted(event_ids),
        "cluster_summary": {
            "source_kind": card.source_kind,
            "source_id": card.source_id,
            "title": card.title,
            "current_state": card.current_state,
            "confirmed_claims": list(card.confirmed_claims),
            "uncertain_claims": list(card.uncertain_claims),
            "disputed_claims": list(card.disputed_claims),
            "open_slots": list(card.open_slots),
        },
    }


def _replacement_sort_score(item: dict[str, Any], wanted_ids: set[int]) -> float:
    source_ids = {int(event_id) for event_id in item.get("source_event_ids") or () if _positive_int(event_id)}
    overlap = len(source_ids & wanted_ids)
    source_kind = str(item.get("source_kind") or item.get("cluster_summary", {}).get("source_kind") or "")
    cluster_bonus = 0.2 if source_kind == "cluster" else 0.0
    return float(item.get("recall_score") or 0.0) + overlap * 0.5 + cluster_bonus


def _matches_scope(
    event_ids: set[int],
    event_meta: dict[int, dict[str, Any]],
    context_scope: str,
) -> bool:
    if not context_scope or ":" not in context_scope:
        return True
    if not event_ids:
        return True
    conv_type, conv_id = context_scope.split(":", 1)
    conv_id = conv_id.removeprefix("qq_")
    for event_id in event_ids:
        meta = event_meta.get(event_id)
        if not meta:
            continue
        ev_type = str(meta.get("conv_type") or "")
        ev_id = str(meta.get("conv_id") or "")
        if (not ev_type and not ev_id) or ev_type == "flow" or (ev_type == conv_type and ev_id == conv_id):
            return True
    return False


def _query_terms(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(match.group(0).lower() for match in _TERM_RE.finditer(text)))[:12]


def _entity_key(entity: object) -> str:
    return re.sub(r"\s+", "", str(entity or "").strip().lower())


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


__all__ = [
    "load_ready_summaries_covering_events",
]
