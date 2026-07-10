"""Deterministic Memory preprocessing and mount consolidation.

This module is the SQLite production-shaped adaptation of the entitySystem
experiments.  It keeps ``MemoryEvents`` as the immutable source of truth and
adds only rebuildable evidence/cache tables around it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

PREPROCESSING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS MemoryPreprocessRuns (
    run_id INTEGER PRIMARY KEY,
    component TEXT NOT NULL,
    trigger TEXT NOT NULL DEFAULT '',
    started_at_ms INTEGER NOT NULL DEFAULT 0,
    finished_at_ms INTEGER NOT NULL DEFAULT 0,
    min_event_id INTEGER NOT NULL DEFAULT 0,
    max_event_id INTEGER NOT NULL DEFAULT 0,
    lookback_event_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'running',
    params_json TEXT NOT NULL DEFAULT '{}',
    stats_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryPreprocessRuns_component
ON MemoryPreprocessRuns(component, status, started_at_ms);

CREATE TABLE IF NOT EXISTS MemoryCanonicalEntities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryEntityAliases (
    alias_key TEXT PRIMARY KEY,
    raw_entity TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    raw_type TEXT NOT NULL DEFAULT '',
    entity_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryEntityMentions (
    event_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    raw_entity TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (event_id, role, raw_entity, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_MemoryEntityAliases_entity
ON MemoryEntityAliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_MemoryEntityMentions_entity
ON MemoryEntityMentions(entity_id, event_id);

CREATE TABLE IF NOT EXISTS MemoryEventRelationRuns (
    run_id INTEGER PRIMARY KEY,
    trigger TEXT NOT NULL DEFAULT '',
    started_at_ms INTEGER NOT NULL DEFAULT 0,
    finished_at_ms INTEGER NOT NULL DEFAULT 0,
    min_event_id INTEGER NOT NULL DEFAULT 0,
    max_event_id INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryEventRelations (
    relation_id TEXT PRIMARY KEY,
    source_event_id INTEGER NOT NULL,
    target_event_id INTEGER NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    corrected_by_event_id INTEGER NOT NULL DEFAULT 0,
    first_seen_run_id INTEGER NOT NULL DEFAULT 0,
    last_seen_run_id INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryEpisodes (
    episode_id TEXT PRIMARY KEY,
    episode_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    event_count INTEGER NOT NULL,
    relation_count INTEGER NOT NULL,
    confidence REAL NOT NULL,
    first_seen_run_id INTEGER NOT NULL DEFAULT 0,
    last_seen_run_id INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryEpisodeMembers (
    episode_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    first_seen_run_id INTEGER NOT NULL DEFAULT 0,
    last_seen_run_id INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (episode_id, event_id)
);
CREATE TABLE IF NOT EXISTS MemoryRelationRevisions (
    revision_id TEXT PRIMARY KEY,
    revised_relation_id TEXT NOT NULL,
    revision_event_id INTEGER NOT NULL,
    revision_type TEXT NOT NULL,
    status_before TEXT NOT NULL,
    status_after TEXT NOT NULL,
    run_id INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryEventRelations_source
ON MemoryEventRelations(source_event_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_MemoryEventRelations_target
ON MemoryEventRelations(target_event_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_MemoryEventRelations_status
ON MemoryEventRelations(status, relation_type);
CREATE INDEX IF NOT EXISTS idx_MemoryEpisodeMembers_event
ON MemoryEpisodeMembers(event_id, status);

CREATE TABLE IF NOT EXISTS MemoryClusterRuns (
    run_id INTEGER PRIMARY KEY,
    profile TEXT NOT NULL,
    trigger TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    min_event_id INTEGER NOT NULL DEFAULT 0,
    max_event_id INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryClusters (
    cluster_id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    scheme_name TEXT NOT NULL DEFAULT '',
    anchor_key TEXT NOT NULL DEFAULT '',
    profile TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    first_seen_run_id INTEGER NOT NULL DEFAULT 0,
    last_seen_run_id INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 1,
    member_count INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    signature_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryClusterMembers (
    cluster_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    score REAL NOT NULL,
    rank INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    revision INTEGER NOT NULL DEFAULT 1,
    corrected_by_event_id INTEGER NOT NULL DEFAULT 0,
    first_seen_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    first_seen_run_id INTEGER NOT NULL DEFAULT 0,
    last_seen_run_id INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (cluster_id, event_id)
);
CREATE TABLE IF NOT EXISTS MemoryClusterMemberRevisions (
    revision_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    revision_event_id INTEGER NOT NULL DEFAULT 0,
    revision_type TEXT NOT NULL,
    status_before TEXT NOT NULL,
    status_after TEXT NOT NULL,
    run_id INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryClusters_scope
ON MemoryClusters(scope, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterMembers_event
ON MemoryClusterMembers(event_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterMembers_status
ON MemoryClusterMembers(status, cluster_id);

CREATE TABLE IF NOT EXISTS MemoryLocalClusterMounts (
    proposal_id TEXT PRIMARY KEY,
    event_ids_json TEXT NOT NULL DEFAULT '[]',
    title TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL,
    evidence_text TEXT NOT NULL DEFAULT '',
    uncertainty_reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryLocalClusterMounts_status
ON MemoryLocalClusterMounts(status, confidence DESC, created_at_ms);

CREATE TABLE IF NOT EXISTS MemoryMounts (
    mount_id TEXT PRIMARY KEY,
    new_event_id INTEGER NOT NULL,
    anchor_summary_id TEXT NOT NULL,
    anchor_source_kind TEXT NOT NULL DEFAULT '',
    anchor_source_id TEXT NOT NULL DEFAULT '',
    anchor_revision INTEGER NOT NULL DEFAULT 0,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_text TEXT NOT NULL DEFAULT '',
    uncertainty_reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryThreadStates (
    thread_id TEXT PRIMARY KEY,
    thread_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    state_json TEXT NOT NULL DEFAULT '{}',
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS MemoryThreadStateRevisions (
    revision_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    revision_type TEXT NOT NULL,
    triggered_by_mount_id TEXT NOT NULL DEFAULT '',
    before_json TEXT NOT NULL DEFAULT '{}',
    after_json TEXT NOT NULL DEFAULT '{}',
    created_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS MemoryClusterRelations (
    relation_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    source_event_id INTEGER NOT NULL,
    target_event_id INTEGER NOT NULL DEFAULT 0,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    revision INTEGER NOT NULL DEFAULT 1,
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryClusterRevisions (
    revision_id TEXT PRIMARY KEY,
    cluster_id TEXT NOT NULL,
    revision_type TEXT NOT NULL,
    before_revision INTEGER NOT NULL DEFAULT 0,
    after_revision INTEGER NOT NULL DEFAULT 0,
    triggered_by_mount_id TEXT NOT NULL DEFAULT '',
    triggered_by_event_id INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryMounts_anchor
ON MemoryMounts(anchor_summary_id, status, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_MemoryMounts_event
ON MemoryMounts(new_event_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterRelations_cluster
ON MemoryClusterRelations(cluster_id, status, relation_type);

CREATE TABLE IF NOT EXISTS MemoryClusterSummaryTasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_revision INTEGER NOT NULL DEFAULT 0,
    input_hash TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    confidence_tier TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS MemoryClusterSummaryTaskEvents (
    task_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    rank INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (task_id, event_id)
);
CREATE TABLE IF NOT EXISTS MemoryClusterSummaryTaskRelations (
    task_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    source_event_id INTEGER NOT NULL DEFAULT 0,
    target_event_id INTEGER NOT NULL DEFAULT 0,
    relation_type TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (task_id, relation_id)
);
CREATE TABLE IF NOT EXISTS MemorySummaryCache (
    summary_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    title TEXT NOT NULL DEFAULT '',
    short_summary TEXT NOT NULL DEFAULT '',
    digest_json TEXT NOT NULL DEFAULT '[]',
    salient_entities_json TEXT NOT NULL DEFAULT '[]',
    cluster_summary_json TEXT NOT NULL DEFAULT '{}',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    error_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterSummaryTasks_source
ON MemoryClusterSummaryTasks(cluster_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterSummaryTasks_queue
ON MemoryClusterSummaryTasks(status, priority DESC, updated_at_ms);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterSummaryTaskEvents_event
ON MemoryClusterSummaryTaskEvents(event_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryClusterSummaryTaskRelations_relation
ON MemoryClusterSummaryTaskRelations(relation_id, status);
CREATE INDEX IF NOT EXISTS idx_MemorySummaryCache_task
ON MemorySummaryCache(task_id, input_hash, status);
"""

SUMMARY_REFRESH_EVENT_WINDOW_LIMIT = 24
SUMMARY_REFRESH_EVENT_TOKEN_BUDGET = 2400
PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{1,32}):(.*)$")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    summary: str
    summary_tok: str
    event_type_norm: str
    status: str
    confidence: float
    occurred_at: int
    conv_type: str
    conv_id: str
    occurrences: int


@dataclass(frozen=True)
class RoleRecord:
    event_id: int
    role: str
    entity: str
    value_text: str = ""


@dataclass(frozen=True)
class ResolvedEntity:
    entity_id: str
    entity_type: str
    canonical_name: str
    confidence: float
    evidence_json: str


@dataclass(frozen=True)
class EntityAlias:
    alias_key: str
    raw_entity: str
    normalized_name: str
    raw_type: str
    entity_id: str
    source_kind: str
    confidence: float
    evidence_json: str


@dataclass(frozen=True)
class EntityMention:
    event_id: int
    role: str
    raw_entity: str
    entity_id: str
    confidence: float
    evidence_json: str


@dataclass(frozen=True)
class EntityResolutionResult:
    entities: list[ResolvedEntity]
    aliases: list[EntityAlias]
    mentions: list[EntityMention]


@dataclass(frozen=True)
class ClusterSummary:
    cluster_id: str
    scope: str
    scheme_name: str
    profile: str
    anchor_key: str
    member_count: int
    score: float
    event_ids: tuple[int, ...]
    evidence_json: str


@dataclass(frozen=True)
class ClusterMember:
    cluster_id: str
    event_id: int
    score: float
    rank: int
    evidence_json: str


@dataclass(frozen=True)
class ClusterSummaryRecord:
    summary_id: str
    source_kind: str
    source_id: str
    revision: int
    title: str
    short_summary: str
    core_entities: tuple[str, ...] = ()
    confirmed_claims: tuple[str, ...] = ()
    uncertain_claims: tuple[str, ...] = ()
    disputed_claims: tuple[str, ...] = ()
    current_state: str = ""
    open_slots: tuple[str, ...] = ()
    boundary_notes: tuple[str, ...] = ()
    source_event_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryAtom:
    event_id: int
    summary: str
    event_type_norm: str = ""
    status: str = ""
    entities: tuple[str, ...] = ()
    occurred_at: int = 0
    source: str = "synthetic"


@dataclass(frozen=True)
class MemoryMount:
    mount_id: str
    new_event_id: int
    anchor_summary_id: str
    anchor_source_kind: str
    anchor_source_id: str
    anchor_revision: int
    relation_type: str
    confidence: float
    evidence_text: str
    uncertainty_reason: str = ""
    status: str = "pending"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AttachAtomToClusterResult:
    candidates: int = 0
    staged: int = 0
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mount_candidates": self.candidates,
            "mounts_staged": self.staged,
            "mount_errors": list(self.errors),
        }


@dataclass(frozen=True)
class LocalClusterMount:
    proposal_id: str
    event_ids: tuple[int, ...]
    title: str
    confidence: float
    evidence_text: str
    uncertainty_reason: str = ""
    status: str = "pending"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterRelation:
    relation_id: str
    cluster_id: str
    source_event_id: int
    target_event_id: int
    relation_type: str
    confidence: float
    status: str = "active"
    revision: int = 1
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterRevision:
    revision_id: str
    cluster_id: str
    revision_type: str
    before_revision: int
    after_revision: int
    triggered_by_mount_id: str
    triggered_by_event_id: int
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MountStatusUpdate:
    mount_id: str
    status: str
    reason: str
    cluster_id: str = ""
    new_relation_ids: tuple[str, ...] = ()
    revised_relation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsolidationResult:
    mount_status_updates: tuple[MountStatusUpdate, ...]
    new_relations: tuple[ClusterRelation, ...]
    revised_relations: tuple[ClusterRelation, ...]
    cluster_revisions: tuple[ClusterRevision, ...]
    stale_summary_ids: tuple[str, ...]


@dataclass(frozen=True)
class PreprocessReport:
    events: int
    canonical_entities: int
    entity_mentions: int
    event_relations: int
    episodes: int
    algorithmic_clustering_enabled: bool
    algorithmic_cluster_ids: tuple[str, ...] = ()
    algorithmic_cluster_members: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "canonical_entities": self.canonical_entities,
            "entity_mentions": self.entity_mentions,
            "event_relations": self.event_relations,
            "episodes": self.episodes,
            "algorithmic_clustering_enabled": self.algorithmic_clustering_enabled,
            "clusters": len(self.algorithmic_cluster_ids),
            "cluster_members": self.algorithmic_cluster_members,
            "algorithmic_cluster_ids": list(self.algorithmic_cluster_ids),
        }


@dataclass(frozen=True)
class MountConsolidationReport:
    result: ConsolidationResult
    pending_mounts_loaded: int = 0
    pending_local_cluster_mounts_loaded: int = 0
    dry_run: bool = True
    solidify: bool = False
    cluster_relation_rows_written: int = 0
    cluster_revision_rows_written: int = 0
    mount_status_rows_updated: int = 0
    summary_cache_rows_stale: int = 0
    thread_state_rows_updated: int = 0
    summary_refresh_tasks_queued: int = 0
    summary_refresh_task_ids_queued: tuple[str, ...] = ()
    local_cluster_rows_written: int = 0
    local_cluster_member_rows_written: int = 0
    local_cluster_mount_status_rows_updated: int = 0
    local_cluster_ids_written: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **consolidation_payload(self.result),
            "pending_mounts_loaded": self.pending_mounts_loaded,
            "pending_local_cluster_mounts_loaded": self.pending_local_cluster_mounts_loaded,
            "dry_run": self.dry_run,
            "solidify": self.solidify,
            "cluster_relation_rows_written": self.cluster_relation_rows_written,
            "cluster_revision_rows_written": self.cluster_revision_rows_written,
            "mount_status_rows_updated": self.mount_status_rows_updated,
            "summary_cache_rows_stale": self.summary_cache_rows_stale,
            "thread_state_rows_updated": self.thread_state_rows_updated,
            "summary_refresh_tasks_queued": self.summary_refresh_tasks_queued,
            "summary_refresh_task_ids_queued": list(self.summary_refresh_task_ids_queued),
            "local_cluster_rows_written": self.local_cluster_rows_written,
            "local_cluster_member_rows_written": self.local_cluster_member_rows_written,
            "local_cluster_mount_status_rows_updated": self.local_cluster_mount_status_rows_updated,
            "local_cluster_ids_written": list(self.local_cluster_ids_written),
        }


def ensure_preprocessing_schema(con: sqlite3.Connection) -> None:
    con.executescript(PREPROCESSING_SCHEMA_SQL)


async def ensure_preprocessing_schema_async(db: Any) -> None:
    await db.executescript(PREPROCESSING_SCHEMA_SQL)


def load_memory_dataset(
    con: sqlite3.Connection,
    *,
    limit: int = 2000,
) -> tuple[dict[int, EventRecord], dict[int, list[RoleRecord]]]:
    con.row_factory = sqlite3.Row
    _require_tables(con, {"MemoryEvents", "MemoryParticipants"})
    rows = list(
        con.execute(
            f"""
            SELECT event_id, summary, summary_tok, event_type_norm, status, confidence,
                   occurred_at, conv_type, conv_id, occurrences
            FROM MemoryEvents
            WHERE is_deleted=0
            ORDER BY occurred_at DESC, event_id DESC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        )
    )
    events = {
        int(row["event_id"]): EventRecord(
            event_id=int(row["event_id"]),
            summary=str(row["summary"] or ""),
            summary_tok=str(row["summary_tok"] or ""),
            event_type_norm=str(row["event_type_norm"] or "").strip().lower(),
            status=str(row["status"] or "actual").strip().lower(),
            confidence=float(row["confidence"] or 0.0),
            occurred_at=int(row["occurred_at"] or 0),
            conv_type=str(row["conv_type"] or ""),
            conv_id=str(row["conv_id"] or ""),
            occurrences=max(1, int(row["occurrences"] or 1)),
        )
        for row in rows
    }
    return events, _load_roles(con, events)


def run_preprocessing(
    con: sqlite3.Connection,
    *,
    limit: int = 2000,
    trigger: str = "manual",
    canonical_entities: bool = True,
    algorithmic_clustering_enabled: bool = False,
) -> PreprocessReport:
    ensure_preprocessing_schema(con)
    started = _now_ms()
    events, roles = load_memory_dataset(con, limit=limit)
    min_event_id = min(events, default=0)
    max_event_id = max(events, default=0)
    cur = con.execute(
        """
        INSERT INTO MemoryPreprocessRuns (
            component, trigger, started_at_ms, min_event_id, max_event_id, params_json, status
        ) VALUES ('memory_preprocessing', ?, ?, ?, ?, ?, 'running')
        """,
        (
            trigger,
            started,
            min_event_id,
            max_event_id,
            _json(
                {
                    "limit": int(limit),
                    "canonical_entities": bool(canonical_entities),
                    "algorithmic_clustering_enabled": bool(algorithmic_clustering_enabled),
                }
            ),
        ),
    )
    run_id = int(cur.lastrowid)
    entity_result = build_entity_resolution(events, roles)
    write_entity_resolution(con, entity_result, now_ms=started)
    working_roles = canonicalize_roles(roles) if canonical_entities else roles
    clusters: list[ClusterSummary] = []
    cluster_members: list[ClusterMember] = []
    if algorithmic_clustering_enabled:
        clusters, cluster_members = materialize_algorithmic_clusters(events, working_roles)
        if clusters:
            cluster_run_id = create_cluster_run(
                con,
                profile="algorithmic",
                trigger=trigger,
                event_ids=(event_id for cluster in clusters for event_id in cluster.event_ids),
                now_ms=started,
                params={"preprocess_run_id": run_id},
            )
            write_cluster_cache(con, clusters, cluster_members, run_id=cluster_run_id, now_ms=started)
    report = PreprocessReport(
        events=len(events),
        canonical_entities=len(entity_result.entities),
        entity_mentions=len(entity_result.mentions),
        event_relations=0,
        episodes=0,
        algorithmic_clustering_enabled=bool(algorithmic_clustering_enabled),
        algorithmic_cluster_ids=tuple(cluster.cluster_id for cluster in clusters),
        algorithmic_cluster_members=len(cluster_members),
    )
    con.execute(
        """
        UPDATE MemoryPreprocessRuns
        SET finished_at_ms=?, status='finished', stats_json=?
        WHERE run_id=?
        """,
        (_now_ms(), _json(report.to_dict()), run_id),
    )
    return report


def build_entity_resolution(
    events: dict[int, EventRecord],
    roles: dict[int, list[RoleRecord]],
    *,
    include_sessions: bool = True,
) -> EntityResolutionResult:
    entity_by_id: dict[str, ResolvedEntity] = {}
    alias_by_key: dict[str, EntityAlias] = {}
    mention_by_key: dict[tuple[int, str, str, str], EntityMention] = {}
    for event_id, event_roles in roles.items():
        if event_id not in events:
            continue
        for role in event_roles:
            raw_entity = str(role.entity or "").strip()
            if not raw_entity:
                continue
            resolved, alias = resolve_raw_entity(raw_entity, role=role.role)
            entity_by_id.setdefault(resolved.entity_id, resolved)
            alias_by_key.setdefault(alias.alias_key, alias)
            mention_by_key[(event_id, role.role, raw_entity, resolved.entity_id)] = EntityMention(
                event_id=event_id,
                role=role.role,
                raw_entity=raw_entity,
                entity_id=resolved.entity_id,
                confidence=resolved.confidence,
                evidence_json=alias.evidence_json,
            )
    if include_sessions:
        for event in events.values():
            raw_session = _raw_session_entity(event)
            if not raw_session:
                continue
            resolved, alias = resolve_raw_entity(raw_session, role="session")
            entity_by_id.setdefault(resolved.entity_id, resolved)
            alias_by_key.setdefault(alias.alias_key, alias)
            mention_by_key[(event.event_id, "session", raw_session, resolved.entity_id)] = EntityMention(
                event_id=event.event_id,
                role="session",
                raw_entity=raw_session,
                entity_id=resolved.entity_id,
                confidence=resolved.confidence,
                evidence_json=alias.evidence_json,
            )
    entities = sorted(entity_by_id.values(), key=lambda item: (item.entity_type, item.canonical_name, item.entity_id))
    return EntityResolutionResult(
        entities=entities,
        aliases=sorted(alias_by_key.values(), key=lambda item: (item.entity_id, item.raw_entity)),
        mentions=sorted(mention_by_key.values(), key=lambda item: (item.event_id, item.role, item.raw_entity)),
    )


def canonicalize_roles(roles: dict[int, list[RoleRecord]]) -> dict[int, list[RoleRecord]]:
    out: dict[int, list[RoleRecord]] = {}
    for event_id, event_roles in roles.items():
        out[event_id] = [
            RoleRecord(
                event_id=role.event_id,
                role=role.role,
                entity=resolve_raw_entity(role.entity, role=role.role)[0].entity_id if role.entity else "",
                value_text=role.value_text,
            )
            for role in event_roles
        ]
    return out


def resolve_raw_entity(raw_entity: str, *, role: str = "") -> tuple[ResolvedEntity, EntityAlias]:
    parsed_type, name = _parse_prefixed_entity(raw_entity)
    normalized_name = _normalize_entity_name(name)
    raw_type = parsed_type or ""
    entity_type, rules, confidence = _canonical_type_for(raw_type, normalized_name)
    canonical_name = normalized_name or _normalize_entity_name(raw_entity) or "unknown"
    entity_id = _entity_id_for(entity_type, canonical_name)
    evidence = {"raw_entity": raw_entity, "raw_type": raw_type, "role": str(role or "").lower(), "rules": rules}
    evidence_json = _json(evidence)
    return (
        ResolvedEntity(entity_id, entity_type, canonical_name, round(confidence, 6), evidence_json),
        EntityAlias(_sha1("alias", entity_id, str(raw_entity or "").strip())[:20], str(raw_entity or "").strip(), canonical_name, raw_type, entity_id, "rule", round(confidence, 6), evidence_json),
    )


def write_entity_resolution(con: sqlite3.Connection, result: EntityResolutionResult, *, now_ms: int) -> None:
    con.executemany(
        """
        INSERT INTO MemoryCanonicalEntities (
            entity_id, entity_type, canonical_name, confidence, status, created_at, updated_at, evidence_json
        ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)
        ON CONFLICT(entity_id) DO UPDATE SET
            entity_type=excluded.entity_type,
            canonical_name=excluded.canonical_name,
            confidence=excluded.confidence,
            status='active',
            updated_at=excluded.updated_at,
            evidence_json=excluded.evidence_json
        """,
        [(item.entity_id, item.entity_type, item.canonical_name, item.confidence, now_ms, now_ms, item.evidence_json) for item in result.entities],
    )
    con.executemany(
        """
        INSERT INTO MemoryEntityAliases (
            alias_key, raw_entity, normalized_name, raw_type, entity_id, source_kind, confidence, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(alias_key) DO UPDATE SET
            normalized_name=excluded.normalized_name,
            entity_id=excluded.entity_id,
            confidence=excluded.confidence,
            evidence_json=excluded.evidence_json
        """,
        [(item.alias_key, item.raw_entity, item.normalized_name, item.raw_type, item.entity_id, item.source_kind, item.confidence, item.evidence_json) for item in result.aliases],
    )
    con.executemany(
        """
        INSERT OR REPLACE INTO MemoryEntityMentions (
            event_id, role, raw_entity, entity_id, confidence, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(item.event_id, item.role, item.raw_entity, item.entity_id, item.confidence, item.evidence_json) for item in result.mentions],
    )
def materialize_algorithmic_clusters(
    events: dict[int, EventRecord],
    roles: dict[int, list[RoleRecord]],
) -> tuple[list[ClusterSummary], list[ClusterMember]]:
    summaries: list[ClusterSummary] = []
    members: list[ClusterMember] = []
    for cluster_events, scope, scheme, profile, anchor_key, score in _algorithmic_cluster_groups(events, roles):
        event_ids = tuple(sorted(cluster_events))
        if len(event_ids) < 2:
            continue
        cluster_id = f"{scope}:{_sha1('cluster', scheme, anchor_key, str(min(event_ids)))[:16]}"
        evidence = {
            "generator": "algorithmic_clustering",
            "anchor_key": anchor_key,
            "scheme_name": scheme,
            "profile": profile,
            "event_ids": list(event_ids),
        }
        summaries.append(
            ClusterSummary(
                cluster_id,
                scope,
                scheme,
                profile,
                anchor_key,
                len(event_ids),
                score,
                event_ids,
                _json(evidence),
            )
        )
        ordered_event_ids = sorted(event_ids, key=lambda event_id: (events[event_id].occurred_at, event_id))
        for rank, event_id in enumerate(ordered_event_ids, start=1):
            members.append(ClusterMember(cluster_id, event_id, score, rank, _json(evidence)))
    return summaries, members


def write_cluster_cache(
    con: sqlite3.Connection,
    summaries: list[ClusterSummary],
    members: list[ClusterMember],
    *,
    run_id: int,
    now_ms: int,
) -> None:
    members_by_cluster: dict[str, set[int]] = defaultdict(set)
    for member in members:
        members_by_cluster[member.cluster_id].add(member.event_id)
    for summary in summaries:
        current_event_ids = sorted(members_by_cluster.get(summary.cluster_id, set()))
        if current_event_ids:
            placeholders = ",".join("?" * len(current_event_ids))
            con.execute(
                f"""
                UPDATE MemoryClusterMembers
                SET status='inactive', last_seen_at=?, last_seen_run_id=?
                WHERE cluster_id=? AND status='active'
                  AND event_id NOT IN ({placeholders})
                """,
                [now_ms, run_id, summary.cluster_id, *current_event_ids],
            )
    con.executemany(
        """
        INSERT INTO MemoryClusters (
            cluster_id, scope, scheme_name, anchor_key, profile, status, created_at, updated_at,
            first_seen_run_id, last_seen_run_id, revision, member_count, score, signature_json
        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET
            member_count=excluded.member_count,
            score=excluded.score,
            status='active',
            last_seen_run_id=excluded.last_seen_run_id,
            revision=CASE WHEN MemoryClusters.member_count != excluded.member_count OR MemoryClusters.signature_json != excluded.signature_json THEN MemoryClusters.revision + 1 ELSE MemoryClusters.revision END,
            updated_at=excluded.updated_at,
            signature_json=excluded.signature_json
        """,
        [(item.cluster_id, item.scope, item.scheme_name, item.anchor_key, item.profile, now_ms, now_ms, run_id, run_id, item.member_count, item.score, item.evidence_json) for item in summaries],
    )
    con.executemany(
        """
        INSERT INTO MemoryClusterMembers (
            cluster_id, event_id, score, rank, status, revision, corrected_by_event_id,
            first_seen_at, last_seen_at, first_seen_run_id, last_seen_run_id, evidence_json
        ) VALUES (?, ?, ?, ?, 'active', 1, 0, ?, ?, ?, ?, ?)
        ON CONFLICT(cluster_id, event_id) DO UPDATE SET
            score=excluded.score,
            rank=excluded.rank,
            status='active',
            revision=CASE
                WHEN MemoryClusterMembers.score != excluded.score
                  OR MemoryClusterMembers.rank != excluded.rank
                  OR MemoryClusterMembers.status != 'active'
                THEN MemoryClusterMembers.revision + 1
                ELSE MemoryClusterMembers.revision
            END,
            last_seen_at=excluded.last_seen_at,
            last_seen_run_id=excluded.last_seen_run_id,
            evidence_json=excluded.evidence_json
        """,
        [(item.cluster_id, item.event_id, item.score, item.rank, now_ms, now_ms, run_id, run_id, item.evidence_json) for item in members],
    )


def create_cluster_run(
    con: sqlite3.Connection,
    *,
    profile: str,
    trigger: str,
    event_ids: Iterable[int],
    now_ms: int,
    params: dict[str, Any] | None = None,
) -> int:
    ids = sorted({int(event_id) for event_id in event_ids if int(event_id) > 0})
    cur = con.execute(
        """
        INSERT INTO MemoryClusterRuns (
            profile, trigger, started_at, finished_at, min_event_id, max_event_id, params_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(profile or "unknown"),
            str(trigger or "unknown"),
            now_ms,
            now_ms,
            min(ids, default=0),
            max(ids, default=0),
            _json(params or {}),
        ),
    )
    return int(cur.lastrowid)


def cluster_summary_to_json(card: ClusterSummaryRecord) -> str:
    return _json(asdict(card))


def cluster_summary_from_json(payload: str | dict[str, Any]) -> ClusterSummaryRecord:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return ClusterSummaryRecord(
        summary_id=str(data.get("summary_id") or ""),
        source_kind=str(data.get("source_kind") or ""),
        source_id=str(data.get("source_id") or ""),
        revision=int(data.get("revision") or 0),
        title=str(data.get("title") or ""),
        short_summary=str(data.get("short_summary") or ""),
        core_entities=tuple(str(x) for x in data.get("core_entities") or ()),
        confirmed_claims=tuple(str(x) for x in data.get("confirmed_claims") or ()),
        uncertain_claims=tuple(str(x) for x in data.get("uncertain_claims") or ()),
        disputed_claims=tuple(str(x) for x in data.get("disputed_claims") or ()),
        current_state=str(data.get("current_state") or ""),
        open_slots=tuple(str(x) for x in data.get("open_slots") or ()),
        boundary_notes=tuple(str(x) for x in data.get("boundary_notes") or ()),
        source_event_ids=tuple(int(x) for x in data.get("source_event_ids") or () if str(x).lstrip("-").isdigit()),
    )


def write_memory_mounts(con: sqlite3.Connection, mounts: Iterable[MemoryMount], *, now_ms: int) -> int:
    rows = [
        (
            item.mount_id,
            item.new_event_id,
            item.anchor_summary_id,
            item.anchor_source_kind,
            item.anchor_source_id,
            item.anchor_revision,
            item.relation_type,
            item.confidence,
            item.evidence_text,
            item.uncertainty_reason,
            item.status,
            now_ms,
            now_ms,
            _json(item.evidence),
        )
        for item in mounts
    ]
    con.executemany(
        """
        INSERT INTO MemoryMounts (
            mount_id, new_event_id, anchor_summary_id, anchor_source_kind,
            anchor_source_id, anchor_revision, relation_type, confidence,
            evidence_text, uncertainty_reason, status, created_at_ms, updated_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mount_id) DO UPDATE SET
            relation_type=excluded.relation_type,
            confidence=excluded.confidence,
            evidence_text=excluded.evidence_text,
            uncertainty_reason=excluded.uncertainty_reason,
            status=excluded.status,
            updated_at_ms=excluded.updated_at_ms,
            evidence_json=excluded.evidence_json
        """,
        rows,
    )
    return len(rows)


def write_local_cluster_mounts(
    con: sqlite3.Connection,
    mounts: Iterable[LocalClusterMount],
    *,
    now_ms: int,
) -> int:
    rows = [
        (
            item.proposal_id,
            _json(list(item.event_ids)),
            item.title,
            item.confidence,
            item.evidence_text,
            item.uncertainty_reason,
            item.status,
            now_ms,
            now_ms,
            _json(item.evidence),
        )
        for item in mounts
    ]
    con.executemany(
        """
        INSERT INTO MemoryLocalClusterMounts (
            proposal_id, event_ids_json, title, confidence, evidence_text,
            uncertainty_reason, status, created_at_ms, updated_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(proposal_id) DO UPDATE SET
            event_ids_json=excluded.event_ids_json,
            title=excluded.title,
            confidence=excluded.confidence,
            evidence_text=excluded.evidence_text,
            uncertainty_reason=excluded.uncertainty_reason,
            status=excluded.status,
            updated_at_ms=excluded.updated_at_ms,
            evidence_json=excluded.evidence_json
        """,
        rows,
    )
    return len(rows)


def stage_atom_to_cluster_mounts(
    con: sqlite3.Connection,
    candidates: Iterable[dict[str, Any]],
    *,
    local_event_ids: dict[str, int],
    now_ms: int | None = None,
    max_mounts_per_atom: int = 3,
) -> AttachAtomToClusterResult:
    """Validate atom-to-cluster commands and write pending mounts.

    A mount proposer only knows local atom ids. This function maps those ids to
    events already written in the current archive batch, reloads the anchor
    ClusterSummaryRecord from SQLite, and refuses hallucinated anchors or stale revisions.
    """

    ensure_preprocessing_schema(con)
    candidate_list = [item for item in candidates if isinstance(item, dict)]
    now = int(now_ms or _now_ms())
    errors: list[str] = []
    anchor_ids = {
        str(item.get("anchor_summary_id") or "").strip()
        for item in candidate_list
        if str(item.get("anchor_summary_id") or "").strip()
    }
    cards_by_id = {card.summary_id: card for card in _load_ready_cluster_summaries_for_anchor(con, anchor_ids)}
    mounts_by_event: dict[int, list[MemoryMount]] = defaultdict(list)
    for index, candidate in enumerate(candidate_list, start=1):
        local_id = str(candidate.get("new_atom_local_id") or candidate.get("new_event_local_id") or "").strip()
        if not local_id or local_id not in local_event_ids:
            errors.append(f"candidate#{index}: unknown new_atom_local_id {local_id!r}")
            continue
        anchor_summary_id = str(candidate.get("anchor_summary_id") or "").strip()
        card = cards_by_id.get(anchor_summary_id)
        if card is None:
            errors.append(f"candidate#{index}: anchor summary not found {anchor_summary_id!r}")
            continue
        try:
            anchor_revision = int(candidate.get("anchor_revision"))
        except (TypeError, ValueError):
            errors.append(f"candidate#{index}: anchor_revision must be an integer")
            continue
        if anchor_revision != int(card.revision):
            errors.append(
                f"candidate#{index}: anchor revision mismatch {anchor_summary_id!r} "
                f"got={anchor_revision} current={card.revision}"
            )
            continue
        relation_type = str(candidate.get("relation_type") or "").strip()
        if not relation_type:
            errors.append(f"candidate#{index}: relation_type must describe a mount relation")
            continue
        confidence_raw = candidate.get("confidence")
        if isinstance(confidence_raw, bool):
            errors.append(f"candidate#{index}: confidence must be numeric")
            continue
        try:
            confidence = max(0.0, min(1.0, float(confidence_raw)))
        except (TypeError, ValueError):
            errors.append(f"candidate#{index}: confidence must be numeric")
            continue
        evidence_text = str(candidate.get("evidence_text") or "").strip()
        if not evidence_text:
            errors.append(f"candidate#{index}: evidence_text is required")
            continue
        event_id = int(local_event_ids[local_id])
        uncertainty = str(candidate.get("uncertainty_reason") or "")
        evidence = {
            "generator": "post_archive_mount_workflow.model_candidate",
            "new_atom_local_id": local_id,
            "raw_mount_json": str(candidate.get("_raw_mount_json") or ""),
        }
        if candidate.get("source_event_ids") is not None:
            evidence["source_event_ids"] = candidate.get("source_event_ids")
        mount = MemoryMount(
            mount_id=_sha1("archive-mount", str(event_id), card.summary_id, relation_type, str(anchor_revision), evidence_text)[:24],
            new_event_id=event_id,
            anchor_summary_id=card.summary_id,
            anchor_source_kind=card.source_kind,
            anchor_source_id=card.source_id,
            anchor_revision=card.revision,
            relation_type=relation_type,
            confidence=round(confidence, 6),
            evidence_text=evidence_text,
            uncertainty_reason=uncertainty,
            evidence=evidence,
        )
        mounts_by_event[event_id].append(mount)

    limit = max(1, int(max_mounts_per_atom or 1))
    mounts: list[MemoryMount] = []
    for event_mounts in mounts_by_event.values():
        event_mounts.sort(key=_mount_sort_key)
        mounts.extend(event_mounts[:limit])
    written = write_memory_mounts(con, mounts, now_ms=now)
    return AttachAtomToClusterResult(
        candidates=len(candidate_list),
        staged=written,
        errors=tuple(errors),
    )


def run_mount_consolidation(
    con: sqlite3.Connection,
    *,
    max_mounts: int = 100,
    dry_run: bool = True,
    solidify: bool = False,
    accept_threshold: float = 0.62,
) -> MountConsolidationReport:
    ensure_preprocessing_schema(con)
    mounts = _load_pending_mounts(con, max_mounts=max_mounts)
    local_mounts = _load_pending_local_cluster_mounts(con, max_mounts=max_mounts)
    cards = _load_ready_cluster_summaries_for_anchor(con, {mount.anchor_summary_id for mount in mounts})
    result = consolidate_memory_mounts(cards, mounts, accept_threshold=accept_threshold)
    should_write = bool(solidify) and not bool(dry_run)
    if should_write:
        write_stats = write_consolidation_result(con, result, now_ms=_now_ms())
        local_stats = _write_local_cluster_consolidation(con, local_mounts, now_ms=_now_ms())
    else:
        write_stats = _empty_consolidation_write_stats()
        local_stats = _empty_local_cluster_write_stats()
    return MountConsolidationReport(
        result=result,
        pending_mounts_loaded=len(mounts),
        pending_local_cluster_mounts_loaded=len(local_mounts),
        dry_run=not should_write,
        solidify=bool(solidify),
        cluster_relation_rows_written=int(write_stats.get("cluster_relation_rows_written") or 0),
        cluster_revision_rows_written=int(write_stats.get("cluster_revision_rows_written") or 0),
        mount_status_rows_updated=int(write_stats.get("mount_status_rows_updated") or 0),
        summary_cache_rows_stale=int(write_stats.get("summary_cache_rows_stale") or 0),
        thread_state_rows_updated=int(write_stats.get("thread_state_rows_updated") or 0),
        summary_refresh_tasks_queued=int(write_stats.get("summary_refresh_tasks_queued") or 0),
        summary_refresh_task_ids_queued=tuple(str(item) for item in write_stats.get("summary_refresh_task_ids_queued") or ()),
        local_cluster_rows_written=int(local_stats.get("local_cluster_rows_written") or 0),
        local_cluster_member_rows_written=int(local_stats.get("local_cluster_member_rows_written") or 0),
        local_cluster_mount_status_rows_updated=int(local_stats.get("local_cluster_mount_status_rows_updated") or 0),
        local_cluster_ids_written=tuple(str(item) for item in local_stats.get("local_cluster_ids_written") or ()),
    )


def consolidate_memory_mounts(
    cards: Iterable[ClusterSummaryRecord],
    mounts: Iterable[MemoryMount],
    *,
    accept_threshold: float = 0.62,
) -> ConsolidationResult:
    card_by_id = {card.summary_id: card for card in cards}
    mount_status_updates: list[MountStatusUpdate] = []
    new_relations: list[ClusterRelation] = []
    revised_relations: list[ClusterRelation] = []
    revisions: list[ClusterRevision] = []
    stale_summary_ids: set[str] = set()
    for mount in sorted((m for m in mounts if m.status == "pending"), key=_mount_sort_key):
        card = card_by_id.get(mount.anchor_summary_id)
        if card is None:
            mount_status_updates.append(
                MountStatusUpdate(
                    mount_id=mount.mount_id,
                    status="obsolete",
                    reason="anchor summary is missing",
                    cluster_id=mount.anchor_source_id,
                )
            )
            continue
        cluster_id = card.source_id or card.summary_id
        if int(mount.anchor_revision) != int(card.revision):
            mount_status_updates.append(
                MountStatusUpdate(
                    mount_id=mount.mount_id,
                    status="obsolete",
                    reason="anchor revision mismatch",
                    cluster_id=cluster_id,
                )
            )
            continue
        if not mount.relation_type:
            mount_status_updates.append(
                MountStatusUpdate(
                    mount_id=mount.mount_id,
                    status="rejected",
                    reason="mount relation is empty",
                    cluster_id=cluster_id,
                )
            )
            continue
        target_event_id = int(card.source_event_ids[-1]) if card.source_event_ids else 0
        relation = ClusterRelation(
            relation_id=_sha1("cluster-relation", cluster_id, str(mount.new_event_id), str(target_event_id), mount.relation_type)[:24],
            cluster_id=cluster_id,
            source_event_id=mount.new_event_id,
            target_event_id=target_event_id,
            relation_type=mount.relation_type,
            confidence=mount.confidence,
            status="active" if mount.confidence >= accept_threshold else "weak",
            evidence={
                "mount_id": mount.mount_id,
                "anchor_summary_id": mount.anchor_summary_id,
                "anchor_source_kind": mount.anchor_source_kind,
                "anchor_source_id": mount.anchor_source_id,
                "evidence_text": mount.evidence_text,
                **mount.evidence,
            },
        )
        new_relations.append(relation)
        stale_summary_ids.add(card.summary_id)
        mount_status_updates.append(
            MountStatusUpdate(
                mount_id=mount.mount_id,
                status="accepted",
                reason="mount has local anchor evidence",
                cluster_id=cluster_id,
                new_relation_ids=(relation.relation_id,),
            )
        )
    return ConsolidationResult(tuple(mount_status_updates), tuple(new_relations), tuple(revised_relations), tuple(revisions), tuple(sorted(stale_summary_ids)))


def write_consolidation_result(con: sqlite3.Connection, result: ConsolidationResult, *, now_ms: int) -> dict[str, Any]:
    relation_count = write_cluster_relations(con, [*result.new_relations, *result.revised_relations], now_ms=now_ms)
    revision_rows = [
        (
            item.revision_id,
            item.cluster_id,
            item.revision_type,
            item.before_revision,
            item.after_revision,
            item.triggered_by_mount_id,
            item.triggered_by_event_id,
            now_ms,
            _json(item.evidence),
        )
        for item in result.cluster_revisions
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO MemoryClusterRevisions (
            revision_id, cluster_id, revision_type, before_revision, after_revision,
            triggered_by_mount_id, triggered_by_event_id, created_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        revision_rows,
    )
    mount_status = {item.mount_id: item.status for item in result.mount_status_updates}
    con.executemany(
        """
        UPDATE MemoryMounts
        SET status=?, updated_at_ms=?
        WHERE mount_id=?
        """,
        [(status, now_ms, mount_id) for mount_id, status in mount_status.items()],
    )
    stale_count = _mark_summary_cache_stale(con, result.stale_summary_ids, now_ms=now_ms)
    thread_rows = _update_thread_states(con, result.new_relations, now_ms=now_ms)
    summary_inputs = _queue_summary_refresh_inputs(con, result.stale_summary_ids, now_ms=now_ms)
    summary_refresh_task_ids = list(result.stale_summary_ids)
    return {
        "cluster_relation_rows_written": relation_count,
        "cluster_revision_rows_written": len(revision_rows),
        "mount_status_rows_updated": len(mount_status),
        "summary_cache_rows_stale": stale_count,
        "thread_state_rows_updated": thread_rows,
        "summary_refresh_tasks_queued": summary_inputs,
        "summary_refresh_task_ids_queued": summary_refresh_task_ids,
    }


def _empty_consolidation_write_stats() -> dict[str, Any]:
    return {
        "cluster_relation_rows_written": 0,
        "cluster_revision_rows_written": 0,
        "mount_status_rows_updated": 0,
        "summary_cache_rows_stale": 0,
        "thread_state_rows_updated": 0,
        "summary_refresh_tasks_queued": 0,
        "summary_refresh_task_ids_queued": [],
    }


def write_cluster_relations(con: sqlite3.Connection, relations: Iterable[ClusterRelation], *, now_ms: int) -> int:
    rows = [
        (
            item.relation_id,
            item.cluster_id,
            item.source_event_id,
            item.target_event_id,
            item.relation_type,
            item.confidence,
            item.status,
            item.revision,
            now_ms,
            now_ms,
            _json(item.evidence),
        )
        for item in relations
    ]
    con.executemany(
        """
        INSERT INTO MemoryClusterRelations (
            relation_id, cluster_id, source_event_id, target_event_id, relation_type,
            confidence, status, revision, created_at_ms, updated_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(relation_id) DO UPDATE SET
            confidence=excluded.confidence,
            status=excluded.status,
            revision=excluded.revision,
            updated_at_ms=excluded.updated_at_ms,
            evidence_json=excluded.evidence_json
        """,
        rows,
    )
    return len(rows)


def consolidation_payload(result: ConsolidationResult) -> dict[str, Any]:
    mount_status_counts = Counter(item.status for item in result.mount_status_updates)
    return {
        "mount_status_counts": dict(sorted(mount_status_counts.items())),
        "stale_summary_ids": list(result.stale_summary_ids),
        "mount_status_updates": [asdict(item) for item in result.mount_status_updates],
        "new_relations": [asdict(item) for item in result.new_relations],
        "revised_relations": [asdict(item) for item in result.revised_relations],
        "cluster_revisions": [asdict(item) for item in result.cluster_revisions],
    }


def _require_tables(con: sqlite3.Connection, names: set[str]) -> None:
    found = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = names - found
    if missing:
        raise RuntimeError(f"missing required Memory tables: {', '.join(sorted(missing))}")


def _load_roles(con: sqlite3.Connection, events: dict[int, EventRecord]) -> dict[int, list[RoleRecord]]:
    if not events:
        return {}
    placeholders = ",".join("?" * len(events))
    out: dict[int, list[RoleRecord]] = defaultdict(list)
    for row in con.execute(
        f"""
        SELECT event_id, role, entity, value_text
        FROM MemoryParticipants
        WHERE event_id IN ({placeholders})
        """,
        list(events),
    ):
        out[int(row["event_id"])].append(RoleRecord(int(row["event_id"]), str(row["role"] or "").strip().lower(), str(row["entity"] or "").strip(), str(row["value_text"] or "").strip()))
    return dict(out)


def _algorithmic_cluster_groups(events: dict[int, EventRecord], roles: dict[int, list[RoleRecord]]):
    by_conversation_window: dict[str, set[int]] = defaultdict(set)
    by_recurrent_anchor: dict[str, set[int]] = defaultdict(set)
    for event in events.values():
        bucket = event.occurred_at // (30 * 60 * 1000) if event.occurred_at else 0
        window_key = f"session:{event.conv_type}:{event.conv_id}:bucket:{bucket}"
        by_conversation_window[window_key].add(event.event_id)
    for event_id, event_roles in roles.items():
        for role in event_roles:
            if not _is_algorithmic_anchor(role.entity):
                continue
            by_recurrent_anchor[f"role_entity:{role.role}:{role.entity}"].add(event_id)
    for anchor_key, event_ids in by_conversation_window.items():
        if 2 <= len(event_ids) <= 6:
            yield event_ids, "session", "session_fragment_precise", "session", anchor_key, 0.42
    for anchor_key, event_ids in by_recurrent_anchor.items():
        if 2 <= len(event_ids) <= 6:
            yield event_ids, "recurrent-anchor", "recurrent_anchor_candidate", "topic-strict", anchor_key, 0.45
def _is_algorithmic_anchor(entity: str) -> bool:
    _entity_type, entity_name = _parse_prefixed_entity(entity)
    normalized_name = _normalize_entity_name(entity_name)
    return bool(normalized_name) and normalized_name.casefold() != "self"


def _load_pending_mounts(con: sqlite3.Connection, *, max_mounts: int) -> list[MemoryMount]:
    con.row_factory = sqlite3.Row
    rows = list(
        con.execute(
            """
            SELECT *
            FROM MemoryMounts
            WHERE status='pending'
            ORDER BY confidence DESC, created_at_ms ASC
            LIMIT ?
            """,
            (max(1, int(max_mounts)),),
        )
    )
    mounts: list[MemoryMount] = []
    for row in rows:
        try:
            evidence = json.loads(str(row["evidence_json"] or "{}"))
        except json.JSONDecodeError:
            evidence = {}
        mounts.append(
            MemoryMount(
                mount_id=str(row["mount_id"]),
                new_event_id=int(row["new_event_id"]),
                anchor_summary_id=str(row["anchor_summary_id"]),
                anchor_source_kind=str(row["anchor_source_kind"] or ""),
                anchor_source_id=str(row["anchor_source_id"] or ""),
                anchor_revision=int(row["anchor_revision"] or 0),
                relation_type=str(row["relation_type"]),
                confidence=float(row["confidence"] or 0.0),
                evidence_text=str(row["evidence_text"] or ""),
                uncertainty_reason=str(row["uncertainty_reason"] or ""),
                status=str(row["status"] or "pending"),
                evidence=evidence if isinstance(evidence, dict) else {},
            )
        )
    return mounts


def _load_pending_local_cluster_mounts(
    con: sqlite3.Connection,
    *,
    max_mounts: int,
) -> list[LocalClusterMount]:
    con.row_factory = sqlite3.Row
    rows = list(
        con.execute(
            """
            SELECT *
            FROM MemoryLocalClusterMounts
            WHERE status='pending'
            ORDER BY confidence DESC, created_at_ms ASC
            LIMIT ?
            """,
            (max(1, int(max_mounts)),),
        )
    )
    mounts: list[LocalClusterMount] = []
    for row in rows:
        event_ids = _json_int_tuple(str(row["event_ids_json"] or "[]"))
        try:
            evidence = json.loads(str(row["evidence_json"] or "{}"))
        except json.JSONDecodeError:
            evidence = {}
        mounts.append(
            LocalClusterMount(
                proposal_id=str(row["proposal_id"]),
                event_ids=event_ids,
                title=str(row["title"] or ""),
                confidence=float(row["confidence"] or 0.0),
                evidence_text=str(row["evidence_text"] or ""),
                uncertainty_reason=str(row["uncertainty_reason"] or ""),
                status=str(row["status"] or "pending"),
                evidence=evidence if isinstance(evidence, dict) else {},
            )
        )
    return mounts


def _empty_local_cluster_write_stats() -> dict[str, Any]:
    return {
        "local_cluster_rows_written": 0,
        "local_cluster_member_rows_written": 0,
        "local_cluster_mount_status_rows_updated": 0,
        "local_cluster_ids_written": [],
    }


def _write_local_cluster_consolidation(
    con: sqlite3.Connection,
    mounts: list[LocalClusterMount],
    *,
    now_ms: int,
) -> dict[str, Any]:
    if not mounts:
        return {
            "local_cluster_rows_written": 0,
            "local_cluster_member_rows_written": 0,
            "local_cluster_mount_status_rows_updated": 0,
            "local_cluster_ids_written": [],
        }
    valid_events = _existing_event_ids(con, {event_id for mount in mounts for event_id in mount.event_ids})
    clusters: list[ClusterSummary] = []
    members: list[ClusterMember] = []
    accepted: list[LocalClusterMount] = []
    rejected: list[LocalClusterMount] = []
    for mount in mounts:
        event_ids = tuple(sorted({event_id for event_id in mount.event_ids if event_id in valid_events}))
        if len(event_ids) < 2:
            rejected.append(mount)
            continue
        cluster_hash = _sha1("local-cluster", *(str(event_id) for event_id in event_ids))
        cluster_id = f"local:{cluster_hash[:16]}"
        anchor_key = f"local:{cluster_hash[:20]}"
        evidence = dict(mount.evidence)
        evidence.update(
            {
                "generator": "sleep_mount_consolidation.local_cluster",
                "proposal_id": mount.proposal_id,
                "event_ids": list(event_ids),
                "title": mount.title,
                "evidence_text": mount.evidence_text,
            }
        )
        clusters.append(
            ClusterSummary(
                cluster_id=cluster_id,
                scope="local",
                scheme_name="llm_local_cluster",
                profile="sleep-consolidated",
                anchor_key=anchor_key,
                member_count=len(event_ids),
                score=round(mount.confidence, 6),
                event_ids=event_ids,
                evidence_json=_json(evidence),
            )
        )
        for rank, event_id in enumerate(event_ids, start=1):
            members.append(
                ClusterMember(
                    cluster_id=cluster_id,
                    event_id=event_id,
                    score=round(mount.confidence, 6),
                    rank=rank,
                    evidence_json=_json(evidence),
                )
            )
        accepted.append(mount)
    if clusters:
        cluster_run_id = create_cluster_run(
            con,
            profile="sleep-consolidated",
            trigger="local_cluster_mount",
            event_ids=(event_id for cluster in clusters for event_id in cluster.event_ids),
            now_ms=now_ms,
            params={"generator": "sleep_mount_consolidation.local_cluster"},
        )
        write_cluster_cache(con, clusters, members, run_id=cluster_run_id, now_ms=now_ms)
    status_rows = 0
    for status, items in (("accepted", accepted), ("rejected", rejected)):
        if not items:
            continue
        placeholders = ",".join("?" * len(items))
        con.execute(
            f"""
            UPDATE MemoryLocalClusterMounts
            SET status=?, updated_at_ms=?
            WHERE proposal_id IN ({placeholders})
            """,
            [status, now_ms, *(item.proposal_id for item in items)],
        )
        status_rows += len(items)
    return {
        "local_cluster_rows_written": len(clusters),
        "local_cluster_member_rows_written": len(members),
        "local_cluster_mount_status_rows_updated": status_rows,
        "local_cluster_ids_written": [cluster.cluster_id for cluster in clusters],
    }


def _existing_event_ids(con: sqlite3.Connection, event_ids: set[int]) -> set[int]:
    if not event_ids:
        return set()
    placeholders = ",".join("?" * len(event_ids))
    return {
        int(row[0])
        for row in con.execute(
            f"""
            SELECT event_id
            FROM MemoryEvents
            WHERE is_deleted=0 AND event_id IN ({placeholders})
            """,
            sorted(event_ids),
        )
    }


def _json_int_tuple(payload: str) -> tuple[int, ...]:
    try:
        values = json.loads(payload)
    except json.JSONDecodeError:
        return ()
    if not isinstance(values, list):
        return ()
    out: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item > 0 and item not in out:
            out.append(item)
    return tuple(out)


def _load_ready_cluster_summaries_for_anchor(con: sqlite3.Connection, summary_ids: set[str]) -> list[ClusterSummaryRecord]:
    cards: list[ClusterSummaryRecord] = []
    for summary_id in sorted(summary_ids):
        row = con.execute(
            """
            SELECT cluster_summary_json
            FROM MemorySummaryCache
            WHERE task_id=? AND status='ready' AND cluster_summary_json <> '{}'
            ORDER BY updated_at_ms DESC, summary_id DESC
            LIMIT 1
            """,
            (summary_id,),
        ).fetchone()
        if row:
            cards.append(cluster_summary_from_json(str(row[0])))
    return cards


def _load_summary_prior_cards_for_refresh(con: sqlite3.Connection, summary_ids: set[str]) -> list[ClusterSummaryRecord]:
    cards: list[ClusterSummaryRecord] = []
    for summary_id in sorted(summary_ids):
        row = con.execute(
            """
            SELECT cluster_summary_json
            FROM MemorySummaryCache
            WHERE task_id=? AND cluster_summary_json <> '{}'
            ORDER BY
              CASE status
                WHEN 'stale' THEN 0
                WHEN 'ready' THEN 1
                ELSE 2
              END,
              updated_at_ms DESC,
              summary_id DESC
            LIMIT 1
            """,
            (summary_id,),
        ).fetchone()
        if row:
            cards.append(cluster_summary_from_json(str(row[0])))
            continue
    return cards


def _mark_summary_cache_stale(con: sqlite3.Connection, summary_ids: Iterable[str], *, now_ms: int) -> int:
    changed = 0
    for summary_id in summary_ids:
        cur = con.execute(
            """
            UPDATE MemorySummaryCache
            SET status='stale', updated_at_ms=?
            WHERE task_id=? AND status='ready'
            """,
            (now_ms, summary_id),
        )
        changed += cur.rowcount if cur.rowcount is not None else 0
    return changed


def _update_thread_states(con: sqlite3.Connection, relations: Iterable[ClusterRelation], *, now_ms: int) -> int:
    changed = 0
    for relation in relations:
        if relation.status == "rejected":
            continue
        anchor_kind = str(relation.evidence.get("anchor_source_kind") or "")
        if anchor_kind != "thread" and not relation.cluster_id.startswith("thread:"):
            continue
        thread_id = relation.cluster_id
        existing = con.execute("SELECT state_json, revision FROM MemoryThreadStates WHERE thread_id=?", (thread_id,)).fetchone()
        before_json = str(existing[0]) if existing else "{}"
        before_revision = int(existing[1]) if existing else 0
        try:
            state = json.loads(before_json)
        except json.JSONDecodeError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        milestones = list(state.get("milestones") or [])
        milestones.append({"event_id": relation.source_event_id, "relation_type": relation.relation_type})
        state.update(
            {
                "thread_id": thread_id,
                "state": relation.relation_type,
                "milestones": milestones,
            }
        )
        after_json = _json(state)
        con.execute(
            """
            INSERT INTO MemoryThreadStates (thread_id, thread_key, status, state_json, revision, updated_at_ms)
            VALUES (?, ?, 'active', ?, 1, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                state_json=excluded.state_json,
                revision=MemoryThreadStates.revision + 1,
                updated_at_ms=excluded.updated_at_ms
            """,
            (thread_id, thread_id.removeprefix("thread:"), after_json, now_ms),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO MemoryThreadStateRevisions (
                revision_id, thread_id, revision_type, triggered_by_mount_id, before_json, after_json, created_at_ms
            ) VALUES (?, ?, 'mount_update', ?, ?, ?, ?)
            """,
            (_sha1("thread-revision", thread_id, str(before_revision + 1), str(relation.source_event_id))[:24], thread_id, str(relation.evidence.get("mount_id") or ""), before_json, after_json, now_ms),
        )
        changed += 1
    return changed


def _queue_summary_refresh_inputs(con: sqlite3.Connection, summary_ids: Iterable[str], *, now_ms: int) -> int:
    queued = 0
    for summary in _load_summary_prior_cards_for_refresh(con, set(summary_ids)):
        cluster_id = summary.source_id or summary.summary_id
        relations = []
        for row in con.execute(
            """
            SELECT relation_id, source_event_id, target_event_id, relation_type, status, confidence
            FROM MemoryClusterRelations
            WHERE cluster_id=?
            ORDER BY updated_at_ms DESC, relation_id
            """,
            (cluster_id,),
        ):
            relations.append(
                {
                    "relation_id": str(row[0]),
                    "source_event_id": int(row[1] or 0),
                    "target_event_id": int(row[2] or 0),
                    "relation_type": str(row[3] or ""),
                    "status": str(row[4] or ""),
                    "confidence": float(row[5] or 0.0),
                }
            )
        event_window = _build_summary_refresh_event_window(
            con,
            summary,
            relations,
            now_ms=now_ms,
            max_events=SUMMARY_REFRESH_EVENT_WINDOW_LIMIT,
            token_budget=SUMMARY_REFRESH_EVENT_TOKEN_BUDGET,
        )
        input_events = event_window or [
            {
                "event_id": event_id,
                "window_role": "previous_summary_source_stub",
            }
            for event_id in summary.source_event_ids
        ]
        task_id = summary.summary_id
        input_hash = _sha1(
            "cluster-summary-task",
            cluster_id,
            str(summary.revision),
            ",".join(str(int(item["event_id"])) for item in input_events if int(item.get("event_id") or 0) > 0),
            ",".join(str(rel.get("relation_id") or "") for rel in relations if str(rel.get("relation_id") or "")),
        )
        con.execute(
            """
            INSERT INTO MemoryClusterSummaryTasks (
                task_id, task_type, cluster_id, cluster_revision, input_hash, priority,
                confidence_tier, status, retry_count, last_error, created_at_ms, updated_at_ms
            ) VALUES (?, 'refresh', ?, ?, ?, 90, 'high', 'active', 0, '', ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                task_type='refresh',
                cluster_id=excluded.cluster_id,
                cluster_revision=excluded.cluster_revision,
                input_hash=excluded.input_hash,
                status='active',
                retry_count=0,
                last_error='',
                updated_at_ms=excluded.updated_at_ms
            """,
            (
                task_id,
                cluster_id,
                summary.revision,
                input_hash,
                now_ms,
                now_ms,
            ),
        )
        con.execute("DELETE FROM MemoryClusterSummaryTaskEvents WHERE task_id=?", (task_id,))
        con.executemany(
            """
            INSERT OR REPLACE INTO MemoryClusterSummaryTaskEvents (task_id, event_id, rank, role, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            [
                (
                    task_id,
                    int(item["event_id"]),
                    index,
                    str(item.get("window_role") or "source"),
                )
                for index, item in enumerate(input_events, start=1)
                if int(item.get("event_id") or 0) > 0
            ],
        )
        con.execute("DELETE FROM MemoryClusterSummaryTaskRelations WHERE task_id=?", (task_id,))
        con.executemany(
            """
            INSERT OR REPLACE INTO MemoryClusterSummaryTaskRelations (
                task_id, relation_id, source_event_id, target_event_id, relation_type, status, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    task_id,
                    rel["relation_id"],
                    rel["source_event_id"],
                    rel["target_event_id"],
                    rel["relation_type"],
                    rel["status"],
                    rel["confidence"],
                )
                for rel in relations
            ],
        )
        queued += 1
    return queued


def _build_summary_refresh_event_window(
    con: sqlite3.Connection,
    card: ClusterSummaryRecord,
    relations: list[dict[str, Any]],
    *,
    now_ms: int,
    max_events: int,
    token_budget: int | None = None,
) -> list[dict[str, Any]]:
    if not _table_exists(con, "MemoryEvents"):
        return []

    delta_ids = {
        int(rel.get("source_event_id") or 0)
        for rel in relations
        if str(rel.get("status") or "") != "rejected"
        and int(rel.get("source_event_id") or 0) > 0
    }
    relation_ids = {
        int(value)
        for rel in relations
        for value in (rel.get("source_event_id"), rel.get("target_event_id"))
        if int(value or 0) > 0
    }
    source_ids = {int(event_id) for event_id in card.source_event_ids if int(event_id) > 0}
    wanted_ids = delta_ids | relation_ids | source_ids
    if not wanted_ids:
        return []

    event_columns = _table_columns(con, "MemoryEvents")
    access_expr = "access_count" if "access_count" in event_columns else "0 AS access_count"
    placeholders = ",".join("?" * len(wanted_ids))
    rows = list(
        con.execute(
            f"""
            SELECT event_id, summary, event_type_norm, status, confidence,
                   occurred_at, created_at, last_seen_at, last_accessed,
                   occurrences, {access_expr}
            FROM MemoryEvents
            WHERE event_id IN ({placeholders}) AND is_deleted=0
            """,
            sorted(wanted_ids),
        )
    )
    if not rows:
        return []

    roles_by_event = _load_event_role_briefs(con, [int(row[0]) for row in rows])
    relation_refs_by_event: dict[int, list[str]] = defaultdict(list)
    for rel in relations:
        relation_id = str(rel.get("relation_id") or "")
        if not relation_id:
            continue
        for key in ("source_event_id", "target_event_id"):
            event_id = int(rel.get(key) or 0)
            if event_id > 0:
                relation_refs_by_event[event_id].append(relation_id)

    items: list[dict[str, Any]] = []
    for row in rows:
        event_id = int(row[0])
        access_count = int(row[10] or 0)
        activation_score = _summary_refresh_activation_score(
            event_id=event_id,
            created_at=int(row[6] or 0),
            occurred_at=int(row[5] or 0),
            last_accessed=int(row[8] or 0),
            access_count=access_count,
            occurrences=int(row[9] or 1),
            confidence=float(row[4] or 0.0),
            delta_ids=delta_ids,
            now_ms=now_ms,
        )
        items.append(
            {
                "event_id": event_id,
                "summary": str(row[1] or ""),
                "event_type_norm": str(row[2] or ""),
                "status": str(row[3] or ""),
                "confidence": float(row[4] or 0.0),
                "occurred_at": int(row[5] or 0),
                "created_at": int(row[6] or 0),
                "last_seen_at": int(row[7] or 0),
                "last_accessed": int(row[8] or 0),
                "occurrences": int(row[9] or 1),
                "access_count": access_count,
                "activation_score": round(activation_score, 6),
                "window_role": _summary_refresh_window_role(event_id, delta_ids, source_ids, relation_ids),
                "roles": roles_by_event.get(event_id, []),
                "relation_refs": sorted(set(relation_refs_by_event.get(event_id, []))),
            }
        )

    limit = max(1, int(max_events or 1))
    items.sort(
        key=lambda item: (
            1 if int(item["event_id"]) in delta_ids else 0,
            float(item["activation_score"]),
            int(item["occurred_at"] or 0),
            int(item["event_id"]),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_tokens = 0
    budget = int(token_budget or 0)
    for item in items:
        is_delta = int(item["event_id"]) in delta_ids
        if len(selected) >= limit and not is_delta:
            continue
        cost = _summary_refresh_event_token_estimate(item)
        if budget > 0 and selected and used_tokens + cost > budget and not is_delta:
            continue
        selected.append(item)
        used_tokens += cost
        if len(selected) >= limit and budget <= 0:
            break
    selected.sort(key=lambda item: (int(item["occurred_at"] or 0), int(item["event_id"])))
    return selected


def _summary_refresh_window_role(
    event_id: int,
    delta_ids: set[int],
    source_ids: set[int],
    relation_ids: set[int],
) -> str:
    if event_id in delta_ids:
        return "delta_new_evidence"
    if event_id in source_ids:
        return "previous_summary_source"
    if event_id in relation_ids:
        return "activated_relation_context"
    return "activated_context"


def _summary_refresh_activation_score(
    *,
    event_id: int,
    created_at: int,
    occurred_at: int,
    last_accessed: int,
    access_count: int,
    occurrences: int,
    confidence: float,
    delta_ids: set[int],
    now_ms: int,
) -> float:
    score = 0.0
    if event_id in delta_ids:
        score += 100.0
    created_age_days = _age_days(created_at, now_ms)
    occurred_age_days = _age_days(occurred_at, now_ms)
    score += 6.0 / (1.0 + created_age_days)
    score += 2.0 / (1.0 + occurred_age_days)
    score += min(12.0, math.log1p(max(0, int(access_count))) * 3.0)
    if last_accessed > 0:
        score += 4.0 / (1.0 + _age_days(last_accessed, now_ms))
    score += min(4.0, math.log1p(max(1, int(occurrences))) * 1.2)
    score += max(0.0, min(1.0, float(confidence))) * 0.5
    return score


def _summary_refresh_event_token_estimate(item: dict[str, Any]) -> int:
    text = " ".join(
        (
            str(item.get("summary") or ""),
            str(item.get("event_type_norm") or ""),
            " ".join(
                f"{role.get('role','')} {role.get('entity','')} {role.get('value_text','')}"
                for role in item.get("roles") or []
                if isinstance(role, dict)
            ),
        )
    )
    # Conservative mixed Chinese/English estimate until the summary worker can
    # replace this with provider-specific tokenization.
    return max(8, int(len(text) / 1.8) + 12)


def _age_days(value_ms: int, now_ms: int) -> float:
    if value_ms <= 0 or now_ms <= 0:
        return 3650.0
    return max(0.0, (int(now_ms) - int(value_ms)) / 86_400_000)


def _load_event_role_briefs(con: sqlite3.Connection, event_ids: list[int]) -> dict[int, list[dict[str, str]]]:
    if not event_ids or not _table_exists(con, "MemoryParticipants"):
        return {}
    placeholders = ",".join("?" * len(event_ids))
    roles: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in con.execute(
        f"""
        SELECT event_id, role, entity, value_text
        FROM MemoryParticipants
        WHERE event_id IN ({placeholders})
        ORDER BY event_id ASC, participant_id ASC
        """,
        event_ids,
    ):
        roles[int(row[0])].append(
            {
                "role": str(row[1] or ""),
                "entity": str(row[2] or ""),
                "value_text": str(row[3] or ""),
            }
        )
    return roles


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table,),
        ).fetchone()
    )


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(con, table):
        return set()
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}


def _mount_sort_key(mount: MemoryMount) -> tuple[float, str]:
    return (-mount.confidence, mount.mount_id)


def _parse_prefixed_entity(raw_entity: str) -> tuple[str, str]:
    raw = str(raw_entity or "").strip()
    match = PREFIX_RE.match(raw)
    if not match:
        return "", raw
    return match.group(1).strip(), match.group(2).strip()


def _normalize_entity_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", str(name or "")).strip()
    return SPACE_RE.sub("", text)


def _canonical_type_for(raw_type: str, normalized_name: str) -> tuple[str, list[str], float]:
    explicit_type = unicodedata.normalize("NFKC", str(raw_type or "")).strip().casefold()
    if explicit_type:
        return explicit_type, ["explicit-entity-type"], 0.99
    if normalized_name.casefold() == "self":
        return "self", ["self-entity"], 1.0
    return "unknown", ["missing-entity-type"], 0.0


def _raw_session_entity(event: EventRecord) -> str:
    if not event.conv_type and not event.conv_id:
        return ""
    return f"Session:{event.conv_type}:{event.conv_id}"


def _entity_id_for(entity_type: str, canonical_name: str) -> str:
    return f"{entity_type}:{_sha1('entity', entity_type, canonical_name)[:16]}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha1(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ClusterRelation",
    "ConsolidationResult",
    "EventRecord",
    "AttachAtomToClusterResult",
    "LocalClusterMount",
    "MemoryAtom",
    "MemoryMount",
    "MountConsolidationReport",
    "MountStatusUpdate",
    "PREPROCESSING_SCHEMA_SQL",
    "PreprocessReport",
    "RoleRecord",
    "ClusterSummaryRecord",
    "build_entity_resolution",
    "canonicalize_roles",
    "consolidate_memory_mounts",
    "ensure_preprocessing_schema",
    "ensure_preprocessing_schema_async",
    "load_memory_dataset",
    "materialize_algorithmic_clusters",
    "run_mount_consolidation",
    "run_preprocessing",
    "stage_atom_to_cluster_mounts",
    "cluster_summary_from_json",
    "cluster_summary_to_json",
    "write_consolidation_result",
    "write_local_cluster_mounts",
    "write_memory_mounts",
]
