"""Deterministic Memory V2 preprocessing and mount consolidation.

This module is the SQLite production-shaped adaptation of the entitySystem
experiments.  It keeps ``MemoryV2Events`` as the immutable source of truth and
adds only rebuildable evidence/cache tables around it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


PREPROCESSING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS MemoryV2PreprocessRuns (
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
CREATE INDEX IF NOT EXISTS idx_MemoryV2PreprocessRuns_component
ON MemoryV2PreprocessRuns(component, status, started_at_ms);

CREATE TABLE IF NOT EXISTS MemoryV2CanonicalEntities (
    entity_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryV2EntityAliases (
    alias_key TEXT PRIMARY KEY,
    raw_entity TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    raw_type TEXT NOT NULL DEFAULT '',
    entity_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryV2EntityMentions (
    event_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    raw_entity TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (event_id, role, raw_entity, entity_id)
);
CREATE TABLE IF NOT EXISTS MemoryV2EntityMergeSuspicions (
    suspicion_id TEXT PRIMARY KEY,
    left_entity_id TEXT NOT NULL,
    right_entity_id TEXT NOT NULL,
    suspicion_type TEXT NOT NULL,
    score REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryV2EntityAliases_entity
ON MemoryV2EntityAliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_MemoryV2EntityMentions_entity
ON MemoryV2EntityMentions(entity_id, event_id);

CREATE TABLE IF NOT EXISTS MemoryV2EventRelationRuns (
    run_id INTEGER PRIMARY KEY,
    trigger TEXT NOT NULL DEFAULT '',
    started_at_ms INTEGER NOT NULL DEFAULT 0,
    finished_at_ms INTEGER NOT NULL DEFAULT 0,
    min_event_id INTEGER NOT NULL DEFAULT 0,
    max_event_id INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryV2EventRelations (
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
CREATE TABLE IF NOT EXISTS MemoryV2Episodes (
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
CREATE TABLE IF NOT EXISTS MemoryV2EpisodeMembers (
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
CREATE TABLE IF NOT EXISTS MemoryV2RelationRevisions (
    revision_id TEXT PRIMARY KEY,
    revised_relation_id TEXT NOT NULL,
    revision_event_id INTEGER NOT NULL,
    revision_type TEXT NOT NULL,
    status_before TEXT NOT NULL,
    status_after TEXT NOT NULL,
    run_id INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryV2EventRelations_source
ON MemoryV2EventRelations(source_event_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_MemoryV2EventRelations_target
ON MemoryV2EventRelations(target_event_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_MemoryV2EventRelations_status
ON MemoryV2EventRelations(status, relation_type);
CREATE INDEX IF NOT EXISTS idx_MemoryV2EpisodeMembers_event
ON MemoryV2EpisodeMembers(event_id, status);

CREATE TABLE IF NOT EXISTS MemoryV2ClusterRuns (
    run_id INTEGER PRIMARY KEY,
    profile TEXT NOT NULL,
    trigger TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    min_event_id INTEGER NOT NULL DEFAULT 0,
    max_event_id INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryV2Clusters (
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
CREATE TABLE IF NOT EXISTS MemoryV2ClusterMembers (
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
CREATE TABLE IF NOT EXISTS MemoryV2ClusterMemberRevisions (
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
CREATE INDEX IF NOT EXISTS idx_MemoryV2Clusters_scope
ON MemoryV2Clusters(scope, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_MemoryV2ClusterMembers_event
ON MemoryV2ClusterMembers(event_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_MemoryV2ClusterMembers_status
ON MemoryV2ClusterMembers(status, cluster_id);

CREATE TABLE IF NOT EXISTS MemoryV2LocalClusterMounts (
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
CREATE INDEX IF NOT EXISTS idx_MemoryV2LocalClusterMounts_status
ON MemoryV2LocalClusterMounts(status, confidence DESC, created_at_ms);

CREATE TABLE IF NOT EXISTS MemoryV2MemoryMounts (
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
CREATE TABLE IF NOT EXISTS MemoryV2ThreadStates (
    thread_id TEXT PRIMARY KEY,
    thread_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    state_json TEXT NOT NULL DEFAULT '{}',
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS MemoryV2ThreadStateRevisions (
    revision_id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    revision_type TEXT NOT NULL,
    triggered_by_mount_id TEXT NOT NULL DEFAULT '',
    before_json TEXT NOT NULL DEFAULT '{}',
    after_json TEXT NOT NULL DEFAULT '{}',
    created_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS MemoryV2ClusterRelations (
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
CREATE TABLE IF NOT EXISTS MemoryV2ClusterRevisions (
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
CREATE INDEX IF NOT EXISTS idx_MemoryV2MemoryMounts_anchor
ON MemoryV2MemoryMounts(anchor_summary_id, status, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_MemoryV2MemoryMounts_event
ON MemoryV2MemoryMounts(new_event_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryV2ClusterRelations_cluster
ON MemoryV2ClusterRelations(cluster_id, status, relation_type);

CREATE TABLE IF NOT EXISTS MemoryV2SummaryInputs (
    packet_id TEXT PRIMARY KEY,
    packet_type TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_revision INTEGER NOT NULL DEFAULT 0,
    input_hash TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    confidence_tier TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    packet_json TEXT NOT NULL,
    invalidation_json TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryV2SummaryInputEvents (
    packet_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    rank INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (packet_id, event_id)
);
CREATE TABLE IF NOT EXISTS MemoryV2SummaryInputRelations (
    packet_id TEXT NOT NULL,
    relation_id TEXT NOT NULL,
    source_event_id INTEGER NOT NULL DEFAULT 0,
    target_event_id INTEGER NOT NULL DEFAULT 0,
    relation_type TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (packet_id, relation_id)
);
CREATE TABLE IF NOT EXISTS MemoryV2SummaryCache (
    summary_id TEXT PRIMARY KEY,
    packet_id TEXT NOT NULL,
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
CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryInputs_source
ON MemoryV2SummaryInputs(source_kind, source_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryInputs_queue
ON MemoryV2SummaryInputs(status, priority DESC, updated_at_ms);
CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryInputEvents_event
ON MemoryV2SummaryInputEvents(event_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryInputRelations_relation
ON MemoryV2SummaryInputRelations(relation_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryCache_packet
ON MemoryV2SummaryCache(packet_id, input_hash, status);
"""


GUARDED_STATUSES = {"hypothetical", "conditional", "future"}
QUESTION_TYPES = {"ask", "question", "request"}
ANSWER_TYPES = {"answer", "reply", "respond", "explain", "tell", "provide", "suggest", "advise"}
SEARCH_TYPES = {"search", "find", "browse", "check", "discover"}
SHARE_TYPES = {"share", "send", "post", "forward", "send_message"}
PARSE_TYPES = {"parse", "receive", "show"}
OBSERVE_TYPES = {"observe", "exist", "notice", "see", "be_quiet", "be_silent"}
DECISION_TYPES = {"decide", "judge", "plan", "refuse", "wait", "ignore"}
CORRECTION_TYPES = {"correct", "clarify", "deny", "point_out", "refute", "supersede"}
STRONG_CORRECTION_MARKERS = (
    "correct",
    "correction",
    "clarify",
    "refute",
    "actually",
    "not the same",
    "wrong one",
    "纠正",
    "更正",
    "澄清",
    "误会",
    "搞混",
    "不指",
    "不是指",
)
NON_CORRECTION_NEGATION_MARKERS = ("是不是", "不是不", "也不是不", "不是万能", "不是所有", "不是吗", "不是吧")
PROGRESS_MARKERS = ("continue", "progress", "complete", "finished", "done", "beat", "推进", "进度", "白金", "通关", "完成", "发芽")
ANSWER_MARKERS = ("answer", "reply", "found", "explain", "回答", "回复", "说明", "找到")
BACKGROUND_MARKERS = ("expensive", "pretty", "like", "discuss", "comment", "贵", "漂亮", "好看", "喜欢", "讨论", "提到", "吐槽", "评价")
CONTEXT_ENTITY_PREFIXES = ("Group:", "Platform:", "Session:", "Time:", "Location:")
SUMMARY_REFRESH_EVENT_WINDOW_LIMIT = 24
SUMMARY_REFRESH_EVENT_TOKEN_BUDGET = 2400
ALLOWED_MEMORY_MOUNT_RELATION_TYPES = {
    "continues",
    "updates_state",
    "progresses",
    "causes_or_results",
    "corrects",
    "corrects_identity",
    "refutes",
    "answers",
    "same_object",
    "same_goal",
    "background_only",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}")
TITLE_RE = re.compile(r"《([^》]{2,40})》")
PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]{1,32}):(.*)$")
SPACE_RE = re.compile(r"\s+")
SCRIPT_FOLD = str.maketrans({"來": "来", "織": "织", "臺": "台", "檯": "台", "裏": "里", "裡": "里"})
TYPE_MAP = {
    "person": "person",
    "user": "person",
    "bot": "person",
    "group": "group",
    "org": "group",
    "platform": "platform",
    "location": "location",
    "tool": "tool",
    "instrument": "tool",
    "work": "work",
    "title": "work",
    "game": "work",
    "concept": "concept",
    "topic": "concept",
    "session": "session",
}


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
class MergeSuspicion:
    suspicion_id: str
    left_entity_id: str
    right_entity_id: str
    suspicion_type: str
    score: float
    status: str
    evidence_json: str


@dataclass(frozen=True)
class EntityResolutionResult:
    entities: list[ResolvedEntity]
    aliases: list[EntityAlias]
    mentions: list[EntityMention]
    suspicions: list[MergeSuspicion]


@dataclass(frozen=True)
class EventRelation:
    relation_id: str
    source_event_id: int
    target_event_id: int
    relation_type: str
    confidence: float
    status: str
    evidence_json: str


@dataclass(frozen=True)
class EpisodeSummary:
    episode_id: str
    episode_type: str
    event_ids: tuple[int, ...]
    relation_ids: tuple[str, ...]
    confidence: float
    evidence_json: str


@dataclass(frozen=True)
class EpisodeMember:
    episode_id: str
    event_id: int
    rank: int
    role: str
    evidence_json: str = "{}"


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
class SleepDecision:
    mount_id: str
    decision: str
    affected_cluster_id: str
    affected_relation_ids: tuple[str, ...]
    new_relation_ids: tuple[str, ...]
    summary_refresh_required: bool
    reason: str


@dataclass(frozen=True)
class ConsolidationResult:
    decisions: tuple[SleepDecision, ...]
    new_relations: tuple[ClusterRelation, ...]
    revised_relations: tuple[ClusterRelation, ...]
    cluster_revisions: tuple[ClusterRevision, ...]
    stale_summary_ids: tuple[str, ...]


def ensure_preprocessing_schema(con: sqlite3.Connection) -> None:
    con.executescript(PREPROCESSING_SCHEMA_SQL)
    _ensure_column(con, "MemoryV2SummaryCache", "cluster_summary_json", "TEXT NOT NULL DEFAULT '{}'")
    _delete_legacy_summary_storage(con)


async def ensure_preprocessing_schema_async(db: Any) -> None:
    await db.executescript(PREPROCESSING_SCHEMA_SQL)
    await _ensure_column_async(db, "MemoryV2SummaryCache", "cluster_summary_json", "TEXT NOT NULL DEFAULT '{}'")
    await _delete_legacy_summary_storage_async(db)


def load_memory_v2_dataset(
    con: sqlite3.Connection,
    *,
    limit: int = 2000,
    include_guarded: bool = False,
) -> tuple[dict[int, EventRecord], dict[int, list[RoleRecord]], dict[str, set[int]]]:
    con.row_factory = sqlite3.Row
    _require_tables(con, {"MemoryV2Events", "MemoryV2Participants", "MemoryV2EventSources"})
    status_sql = "" if include_guarded else "AND lower(status) NOT IN ('hypothetical','conditional','future')"
    rows = list(
        con.execute(
            f"""
            SELECT event_id, summary, summary_tok, event_type_norm, status, confidence,
                   occurred_at, conv_type, conv_id, occurrences
            FROM MemoryV2Events
            WHERE is_deleted=0 {status_sql}
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
    return events, _load_roles(con, events), _load_sources(con, events)


def run_preprocessing(
    con: sqlite3.Connection,
    *,
    limit: int = 2000,
    trigger: str = "manual",
    canonical_entities: bool = True,
) -> dict[str, Any]:
    ensure_preprocessing_schema(con)
    started = _now_ms()
    events, roles, sources = load_memory_v2_dataset(con, limit=limit)
    min_event_id = min(events, default=0)
    max_event_id = max(events, default=0)
    cur = con.execute(
        """
        INSERT INTO MemoryV2PreprocessRuns (
            component, trigger, started_at_ms, min_event_id, max_event_id, params_json, status
        ) VALUES ('memory_preprocessing', ?, ?, ?, ?, ?, 'running')
        """,
        (
            trigger,
            started,
            min_event_id,
            max_event_id,
            _json({"limit": int(limit), "canonical_entities": bool(canonical_entities)}),
        ),
    )
    run_id = int(cur.lastrowid)
    entity_result = build_entity_resolution(events, roles)
    write_entity_resolution(con, entity_result, now_ms=started)
    working_roles = canonicalize_roles(roles) if canonical_entities else roles
    relations = build_event_relations(events, working_roles, sources)
    episodes, episode_members = materialize_episodes(events, relations)
    write_event_relations(con, relations, episodes, episode_members, run_id=run_id, now_ms=started)
    clusters, cluster_members = materialize_clusters(events, working_roles)
    write_cluster_cache(con, clusters, cluster_members, run_id=run_id, now_ms=started)
    stats = {
        "events": len(events),
        "canonical_entities": len(entity_result.entities),
        "entity_mentions": len(entity_result.mentions),
        "merge_suspicions": len(entity_result.suspicions),
        "event_relations": len(relations),
        "episodes": len(episodes),
        "clusters": len(clusters),
        "cluster_members": len(cluster_members),
    }
    con.execute(
        """
        UPDATE MemoryV2PreprocessRuns
        SET finished_at_ms=?, status='finished', stats_json=?
        WHERE run_id=?
        """,
        (_now_ms(), _json(stats), run_id),
    )
    return stats


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
        suspicions=build_merge_suspicions(entities),
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
    entity_type, rules, confidence = _canonical_type_for(raw_type, role, normalized_name)
    canonical_name = normalized_name or _normalize_entity_name(raw_entity) or "unknown"
    entity_id = _entity_id_for(entity_type, canonical_name)
    evidence = {"raw_entity": raw_entity, "raw_type": raw_type, "role": str(role or "").lower(), "rules": rules}
    evidence_json = _json(evidence)
    return (
        ResolvedEntity(entity_id, entity_type, canonical_name, round(confidence, 6), evidence_json),
        EntityAlias(_sha1("alias", entity_id, str(raw_entity or "").strip())[:20], str(raw_entity or "").strip(), canonical_name, raw_type, entity_id, "rule", round(confidence, 6), evidence_json),
    )


def build_merge_suspicions(entities: Iterable[ResolvedEntity]) -> list[MergeSuspicion]:
    by_type: dict[str, list[ResolvedEntity]] = defaultdict(list)
    for entity in entities:
        by_type[entity.entity_type].append(entity)
    out: list[MergeSuspicion] = []
    for entity_type, items in by_type.items():
        if entity_type not in {"person", "group", "work", "concept", "tool"}:
            continue
        ordered = sorted(items, key=lambda item: (item.canonical_name, item.entity_id))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                suspicion = _merge_suspicion_for(left, right)
                if suspicion:
                    out.append(suspicion)
    return sorted(out, key=lambda item: (-item.score, item.suspicion_id))


def write_entity_resolution(con: sqlite3.Connection, result: EntityResolutionResult, *, now_ms: int) -> None:
    con.executemany(
        """
        INSERT INTO MemoryV2CanonicalEntities (
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
        INSERT INTO MemoryV2EntityAliases (
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
        INSERT OR REPLACE INTO MemoryV2EntityMentions (
            event_id, role, raw_entity, entity_id, confidence, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [(item.event_id, item.role, item.raw_entity, item.entity_id, item.confidence, item.evidence_json) for item in result.mentions],
    )
    con.executemany(
        """
        INSERT INTO MemoryV2EntityMergeSuspicions (
            suspicion_id, left_entity_id, right_entity_id, suspicion_type, score, status, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(suspicion_id) DO UPDATE SET
            score=excluded.score,
            status=CASE WHEN MemoryV2EntityMergeSuspicions.status='resolved' THEN 'resolved' ELSE excluded.status END,
            evidence_json=excluded.evidence_json
        """,
        [(item.suspicion_id, item.left_entity_id, item.right_entity_id, item.suspicion_type, item.score, item.status, item.evidence_json) for item in result.suspicions],
    )


def build_event_relations(
    events: dict[int, EventRecord],
    roles: dict[int, list[RoleRecord]],
    sources: dict[str, set[int]],
    *,
    max_gap_ms: int = 30 * 60 * 1000,
) -> list[EventRelation]:
    relations: dict[tuple[int, int, str], EventRelation] = {}

    def add(source_id: int, target_id: int, relation_type: str, confidence: float, evidence: dict[str, object]) -> None:
        if source_id == target_id or source_id not in events or target_id not in events:
            return
        key = (int(source_id), int(target_id), relation_type)
        relation = EventRelation(
            relation_id=_sha1("event-relation", str(source_id), str(target_id), relation_type)[:20],
            source_event_id=int(source_id),
            target_event_id=int(target_id),
            relation_type=relation_type,
            confidence=round(float(confidence), 6),
            status="active",
            evidence_json=_json(evidence),
        )
        previous = relations.get(key)
        if previous is None or relation.confidence > previous.confidence:
            relations[key] = relation

    for source_uid, event_ids in sources.items():
        ordered = sorted((eid for eid in event_ids if eid in events), key=lambda eid: (events[eid].occurred_at, eid))
        for left, right in zip(ordered, ordered[1:]):
            add(left, right, "same_episode_as", 0.92, {"rule": "same-source-adjacent", "source_uid": source_uid})
    _add_conversation_trace_relations(events, roles, max_gap_ms, add)
    _add_correction_relations(events, roles, max_gap_ms, add)
    return sorted(relations.values(), key=lambda item: (item.source_event_id, item.target_event_id, item.relation_type))


def materialize_episodes(
    events: dict[int, EventRecord],
    relations: list[EventRelation],
) -> tuple[list[EpisodeSummary], list[EpisodeMember]]:
    parent = {event_id: event_id for event_id in events}

    def find(event_id: int) -> int:
        root = event_id
        while parent[root] != root:
            root = parent[root]
        while parent[event_id] != root:
            previous = parent[event_id]
            parent[event_id] = root
            event_id = previous
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for relation in relations:
        if relation.relation_type in {"same_episode_as", "possibly_refers_to", "answers", "triggers", "causes", "caused_by", "corrects", "refutes", "supersedes"}:
            union(relation.source_event_id, relation.target_event_id)
    component_relations: dict[int, list[EventRelation]] = defaultdict(list)
    for relation in relations:
        if find(relation.source_event_id) == find(relation.target_event_id):
            component_relations[find(relation.source_event_id)].append(relation)
    summaries: list[EpisodeSummary] = []
    members: list[EpisodeMember] = []
    for component_relation_list in component_relations.values():
        event_ids = tuple(
            sorted(
                {rel.source_event_id for rel in component_relation_list} | {rel.target_event_id for rel in component_relation_list},
                key=lambda eid: (events[eid].occurred_at, eid),
            )
        )
        if len(event_ids) < 2:
            continue
        relation_ids = tuple(rel.relation_id for rel in component_relation_list)
        episode_type = _episode_type_for(component_relation_list)
        confidence = max(rel.confidence for rel in component_relation_list)
        episode_id = f"{episode_type}:{_sha1('episode', episode_type, str(min(event_ids)))[:16]}"
        summaries.append(EpisodeSummary(episode_id, episode_type, event_ids, relation_ids, round(confidence, 6), _json({"relation_types": sorted({rel.relation_type for rel in component_relation_list})})))
        for rank, event_id in enumerate(event_ids, start=1):
            members.append(EpisodeMember(episode_id, event_id, rank, _episode_member_role(event_id, component_relation_list)))
    return sorted(summaries, key=lambda item: item.episode_id), sorted(members, key=lambda item: (item.episode_id, item.rank))


def write_event_relations(
    con: sqlite3.Connection,
    relations: list[EventRelation],
    episodes: list[EpisodeSummary],
    members: list[EpisodeMember],
    *,
    run_id: int,
    now_ms: int,
) -> None:
    con.execute(
        """
        INSERT INTO MemoryV2EventRelationRuns (
            run_id, trigger, started_at_ms, finished_at_ms, min_event_id, max_event_id, params_json
        ) VALUES (?, 'preprocess', ?, ?, 0, 0, '{}')
        ON CONFLICT(run_id) DO UPDATE SET finished_at_ms=excluded.finished_at_ms
        """,
        (run_id, now_ms, now_ms),
    )
    con.executemany(
        """
        INSERT INTO MemoryV2EventRelations (
            relation_id, source_event_id, target_event_id, relation_type, confidence, status,
            corrected_by_event_id, first_seen_run_id, last_seen_run_id, revision, updated_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 1, ?, ?)
        ON CONFLICT(relation_id) DO UPDATE SET
            confidence=excluded.confidence,
            status=CASE WHEN MemoryV2EventRelations.status IN ('deprecated', 'superseded') THEN 'active' ELSE MemoryV2EventRelations.status END,
            last_seen_run_id=excluded.last_seen_run_id,
            revision=CASE WHEN MemoryV2EventRelations.confidence != excluded.confidence OR MemoryV2EventRelations.evidence_json != excluded.evidence_json THEN MemoryV2EventRelations.revision + 1 ELSE MemoryV2EventRelations.revision END,
            updated_at_ms=excluded.updated_at_ms,
            evidence_json=excluded.evidence_json
        """,
        [(item.relation_id, item.source_event_id, item.target_event_id, item.relation_type, item.confidence, item.status, run_id, run_id, now_ms, item.evidence_json) for item in relations],
    )
    con.executemany(
        """
        INSERT INTO MemoryV2Episodes (
            episode_id, episode_type, status, event_count, relation_count, confidence,
            first_seen_run_id, last_seen_run_id, revision, updated_at_ms, evidence_json
        ) VALUES (?, ?, 'active', ?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(episode_id) DO UPDATE SET
            event_count=excluded.event_count,
            relation_count=excluded.relation_count,
            confidence=excluded.confidence,
            status='active',
            last_seen_run_id=excluded.last_seen_run_id,
            revision=CASE WHEN MemoryV2Episodes.event_count != excluded.event_count OR MemoryV2Episodes.relation_count != excluded.relation_count OR MemoryV2Episodes.evidence_json != excluded.evidence_json THEN MemoryV2Episodes.revision + 1 ELSE MemoryV2Episodes.revision END,
            updated_at_ms=excluded.updated_at_ms,
            evidence_json=excluded.evidence_json
        """,
        [(item.episode_id, item.episode_type, len(item.event_ids), len(item.relation_ids), item.confidence, run_id, run_id, now_ms, item.evidence_json) for item in episodes],
    )
    con.executemany(
        """
        INSERT INTO MemoryV2EpisodeMembers (
            episode_id, event_id, rank, role, status, first_seen_run_id, last_seen_run_id, updated_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?)
        ON CONFLICT(episode_id, event_id) DO UPDATE SET
            rank=excluded.rank,
            role=excluded.role,
            status='active',
            last_seen_run_id=excluded.last_seen_run_id,
            updated_at_ms=excluded.updated_at_ms,
            evidence_json=excluded.evidence_json
        """,
        [(item.episode_id, item.event_id, item.rank, item.role, run_id, run_id, now_ms, item.evidence_json) for item in members],
    )


def materialize_clusters(
    events: dict[int, EventRecord],
    roles: dict[int, list[RoleRecord]],
) -> tuple[list[ClusterSummary], list[ClusterMember]]:
    summaries: list[ClusterSummary] = []
    members: list[ClusterMember] = []
    for cluster_events, scope, scheme, profile, anchor_key, score in _cluster_groups(events, roles):
        event_ids = tuple(sorted(cluster_events))
        if len(event_ids) < 2:
            continue
        cluster_id = f"{scope}:{_sha1('cluster', scheme, anchor_key, str(min(event_ids)))[:16]}"
        evidence = {"anchor_key": anchor_key, "scheme_name": scheme, "profile": profile}
        summaries.append(
            ClusterSummary(cluster_id, scope, scheme, profile, anchor_key, len(event_ids), score, event_ids, _json(evidence))
        )
        for rank, event_id in enumerate(sorted(event_ids, key=lambda eid: (events[eid].occurred_at, eid)), start=1):
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
    con.execute(
        """
        INSERT INTO MemoryV2ClusterRuns (
            run_id, profile, trigger, started_at, finished_at, min_event_id, max_event_id, params_json
        ) VALUES (?, 'mixed', 'preprocess', ?, ?, 0, 0, '{}')
        ON CONFLICT(run_id) DO UPDATE SET finished_at=excluded.finished_at
        """,
        (run_id, now_ms, now_ms),
    )
    con.executemany(
        """
        INSERT INTO MemoryV2Clusters (
            cluster_id, scope, scheme_name, anchor_key, profile, status, created_at, updated_at,
            first_seen_run_id, last_seen_run_id, revision, member_count, score, signature_json
        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(cluster_id) DO UPDATE SET
            member_count=excluded.member_count,
            score=excluded.score,
            status='active',
            last_seen_run_id=excluded.last_seen_run_id,
            revision=CASE WHEN MemoryV2Clusters.member_count != excluded.member_count OR MemoryV2Clusters.signature_json != excluded.signature_json THEN MemoryV2Clusters.revision + 1 ELSE MemoryV2Clusters.revision END,
            updated_at=excluded.updated_at,
            signature_json=excluded.signature_json
        """,
        [(item.cluster_id, item.scope, item.scheme_name, item.anchor_key, item.profile, now_ms, now_ms, run_id, run_id, item.member_count, item.score, item.evidence_json) for item in summaries],
    )
    con.executemany(
        """
        INSERT INTO MemoryV2ClusterMembers (
            cluster_id, event_id, score, rank, status, revision, corrected_by_event_id,
            first_seen_at, last_seen_at, first_seen_run_id, last_seen_run_id, evidence_json
        ) VALUES (?, ?, ?, ?, 'active', 1, 0, ?, ?, ?, ?, ?)
        ON CONFLICT(cluster_id, event_id) DO UPDATE SET
            score=excluded.score,
            rank=excluded.rank,
            status='active',
            last_seen_at=excluded.last_seen_at,
            last_seen_run_id=excluded.last_seen_run_id,
            evidence_json=excluded.evidence_json
        """,
        [(item.cluster_id, item.event_id, item.score, item.rank, now_ms, now_ms, run_id, run_id, item.evidence_json) for item in members],
    )


def cluster_summary_to_json(card: ClusterSummaryRecord) -> str:
    return _json(asdict(card))


def cluster_summary_from_json(payload: str | dict[str, Any]) -> ClusterSummaryRecord:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return ClusterSummaryRecord(
        summary_id=str(data.get("summary_id") or data.get("packet_id") or ""),
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


def propose_memory_mounts(
    cards: Iterable[ClusterSummaryRecord],
    atoms: Iterable[MemoryAtom],
    *,
    max_mounts_per_atom: int = 3,
    min_confidence: float = 0.30,
) -> list[MemoryMount]:
    card_list = list(cards)
    mounts: list[MemoryMount] = []
    for atom in atoms:
        candidates: list[MemoryMount] = []
        for card in card_list:
            relation_type, confidence, evidence, uncertainty = _classify_mount(card, atom)
            if relation_type == "unrelated" or confidence < min_confidence:
                continue
            candidates.append(
                MemoryMount(
                    mount_id=_sha1("mount", str(atom.event_id), card.summary_id, relation_type)[:24],
                    new_event_id=atom.event_id,
                    anchor_summary_id=card.summary_id,
                    anchor_source_kind=card.source_kind,
                    anchor_source_id=card.source_id,
                    anchor_revision=card.revision,
                    relation_type=relation_type,
                    confidence=round(confidence, 4),
                    evidence_text=atom.summary,
                    uncertainty_reason=uncertainty,
                    evidence=evidence,
                )
            )
        candidates.sort(key=lambda item: (_relation_priority(item.relation_type), item.confidence), reverse=True)
        mounts.extend(candidates[: max(1, int(max_mounts_per_atom))])
    return mounts


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
        INSERT INTO MemoryV2MemoryMounts (
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
        INSERT INTO MemoryV2LocalClusterMounts (
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


def stage_memory_mount_candidates(
    con: sqlite3.Connection,
    candidates: Iterable[dict[str, Any]],
    *,
    local_event_ids: dict[str, int],
    now_ms: int | None = None,
    max_mounts_per_atom: int = 3,
) -> dict[str, Any]:
    """Validate second-step mount candidates and write safe pending mounts.

    A mount proposer only knows local atom ids.  This function maps those ids to
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
        if relation_type == "unrelated" or relation_type not in ALLOWED_MEMORY_MOUNT_RELATION_TYPES:
            errors.append(f"candidate#{index}: relation_type is not allowed {relation_type!r}")
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
    return {
        "mount_candidates": len(candidate_list),
        "mounts_staged": written,
        "mount_errors": errors,
    }


def run_mount_consolidation(
    con: sqlite3.Connection,
    *,
    max_mounts: int = 100,
    dry_run: bool = True,
    solidify: bool = False,
    accept_threshold: float = 0.62,
) -> dict[str, Any]:
    ensure_preprocessing_schema(con)
    mounts = _load_pending_mounts(con, max_mounts=max_mounts)
    local_mounts = _load_pending_local_cluster_mounts(con, max_mounts=max_mounts)
    cards = _load_ready_cluster_summaries_for_anchor(con, {mount.anchor_summary_id for mount in mounts})
    existing = _load_cluster_relations(con, {card.source_id or card.summary_id for card in cards})
    result = consolidate_memory_mounts(cards, mounts, existing_relations=existing, accept_threshold=accept_threshold)
    stats = consolidation_payload(result)
    stats["pending_mounts_loaded"] = len(mounts)
    stats.update(_local_cluster_mount_payload(local_mounts))
    should_write = bool(solidify) and not bool(dry_run)
    stats["dry_run"] = not should_write
    stats["solidify"] = bool(solidify)
    if should_write:
        stats.update(write_consolidation_result(con, result, now_ms=_now_ms()))
        stats.update(_write_local_cluster_consolidation(con, local_mounts, now_ms=_now_ms()))
    return stats


def consolidate_memory_mounts(
    cards: Iterable[ClusterSummaryRecord],
    mounts: Iterable[MemoryMount],
    *,
    existing_relations: Iterable[ClusterRelation] = (),
    accept_threshold: float = 0.62,
) -> ConsolidationResult:
    card_by_id = {card.summary_id: card for card in cards}
    relations_by_cluster: dict[str, list[ClusterRelation]] = defaultdict(list)
    for relation in existing_relations:
        relations_by_cluster[relation.cluster_id].append(relation)
    decisions: list[SleepDecision] = []
    new_relations: list[ClusterRelation] = []
    revised_relations: list[ClusterRelation] = []
    revisions: list[ClusterRevision] = []
    stale_summary_ids: set[str] = set()
    for mount in sorted((m for m in mounts if m.status == "pending"), key=_mount_sort_key):
        card = card_by_id.get(mount.anchor_summary_id)
        if card is None:
            decisions.append(SleepDecision(mount.mount_id, "reject_wrong_anchor", mount.anchor_source_id, (), (), False, "anchor summary is missing"))
            continue
        cluster_id = card.source_id or card.summary_id
        if int(mount.anchor_revision) != int(card.revision):
            decisions.append(SleepDecision(mount.mount_id, "reject_wrong_anchor", cluster_id, (), (), False, "anchor revision mismatch"))
            continue
        if mount.relation_type in {"background_only", "unrelated"}:
            decisions.append(SleepDecision(mount.mount_id, "reject_background", cluster_id, (), (), False, "background relation is not solidified"))
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
        if mount.relation_type in {"corrects", "corrects_identity", "refutes"}:
            affected = _relations_rejected_by_correction(relations_by_cluster.get(cluster_id, ()))
            revised_ids = []
            for old in affected:
                revised = ClusterRelation(
                    relation_id=old.relation_id,
                    cluster_id=old.cluster_id,
                    source_event_id=old.source_event_id,
                    target_event_id=old.target_event_id,
                    relation_type=old.relation_type,
                    confidence=old.confidence,
                    status="rejected",
                    revision=old.revision + 1,
                    evidence={**old.evidence, "rejected_by_mount_id": mount.mount_id, "rejected_by_event_id": mount.new_event_id},
                )
                revised_relations.append(revised)
                revised_ids.append(revised.relation_id)
                revisions.append(
                    ClusterRevision(
                        revision_id=_sha1("cluster-revision", cluster_id, mount.mount_id, old.relation_id)[:24],
                        cluster_id=cluster_id,
                        revision_type="revise_relation",
                        before_revision=old.revision,
                        after_revision=old.revision + 1,
                        triggered_by_mount_id=mount.mount_id,
                        triggered_by_event_id=mount.new_event_id,
                        evidence={"revised_relation_id": old.relation_id, "status_before": old.status, "status_after": "rejected"},
                    )
                )
            new_relations.append(relation)
            stale_summary_ids.add(card.summary_id)
            decisions.append(SleepDecision(mount.mount_id, "revise_existing_relation" if affected else "accept_with_uncertainty", cluster_id, tuple(revised_ids), (relation.relation_id,), True, "correction/refutation revised relation state"))
            continue
        new_relations.append(relation)
        stale_summary_ids.add(card.summary_id)
        decision = "accept_attach" if relation.status == "active" else "accept_with_uncertainty"
        decisions.append(SleepDecision(mount.mount_id, decision, cluster_id, (), (relation.relation_id,), True, "mount has local anchor evidence"))
    return ConsolidationResult(tuple(decisions), tuple(new_relations), tuple(revised_relations), tuple(revisions), tuple(sorted(stale_summary_ids)))


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
        INSERT OR REPLACE INTO MemoryV2ClusterRevisions (
            revision_id, cluster_id, revision_type, before_revision, after_revision,
            triggered_by_mount_id, triggered_by_event_id, created_at_ms, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        revision_rows,
    )
    mount_status = _decision_mount_status(result.decisions)
    con.executemany(
        """
        UPDATE MemoryV2MemoryMounts
        SET status=?, updated_at_ms=?
        WHERE mount_id=?
        """,
        [(status, now_ms, mount_id) for mount_id, status in mount_status.items()],
    )
    stale_count = _mark_summary_cache_stale(con, result.stale_summary_ids, now_ms=now_ms)
    thread_rows = _update_thread_states(con, result.new_relations, now_ms=now_ms)
    summary_inputs = _queue_summary_refresh_inputs(con, result.stale_summary_ids, now_ms=now_ms)
    summary_refresh_packet_ids = [f"summary-refresh:{summary_id}" for summary_id in result.stale_summary_ids]
    return {
        "cluster_relation_rows_written": relation_count,
        "cluster_revision_rows_written": len(revision_rows),
        "mount_status_rows_updated": len(mount_status),
        "summary_cache_rows_stale": stale_count,
        "thread_state_rows_updated": thread_rows,
        "summary_refresh_inputs_queued": summary_inputs,
        "summary_refresh_packet_ids_queued": summary_refresh_packet_ids,
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
        INSERT INTO MemoryV2ClusterRelations (
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
    decision_counts = Counter(item.decision for item in result.decisions)
    return {
        "decision_counts": dict(sorted(decision_counts.items())),
        "stale_summary_ids": list(result.stale_summary_ids),
        "decisions": [asdict(item) for item in result.decisions],
        "new_relations": [asdict(item) for item in result.new_relations],
        "revised_relations": [asdict(item) for item in result.revised_relations],
        "cluster_revisions": [asdict(item) for item in result.cluster_revisions],
    }


def _ensure_column(con: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


async def _ensure_column_async(db: Any, table: str, column: str, ddl: str) -> None:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        cols = {str(row[1]) for row in await cur.fetchall()}
    if column not in cols:
        await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _delete_legacy_summary_storage(con: sqlite3.Connection) -> None:
    if not _table_exists(con, "MemoryV2SummaryCache"):
        return
    legacy_json_column = _legacy_summary_json_column()
    if legacy_json_column in _table_columns(con, "MemoryV2SummaryCache"):
        try:
            con.execute(f"ALTER TABLE MemoryV2SummaryCache DROP COLUMN {legacy_json_column}")
        except sqlite3.OperationalError:
            _rebuild_summary_cache_without_legacy_columns(con)
    con.execute("DELETE FROM MemoryV2SummaryCache WHERE cluster_summary_json = '{}'")
    _delete_legacy_summary_input_packets(con)


async def _delete_legacy_summary_storage_async(db: Any) -> None:
    async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MemoryV2SummaryCache'") as cur:
        exists = await cur.fetchone()
    if not exists:
        return
    async with db.execute("PRAGMA table_info(MemoryV2SummaryCache)") as cur:
        cols = {str(row[1]) for row in await cur.fetchall()}
    legacy_json_column = _legacy_summary_json_column()
    if legacy_json_column in cols:
        try:
            await db.execute(f"ALTER TABLE MemoryV2SummaryCache DROP COLUMN {legacy_json_column}")
        except Exception:
            await _rebuild_summary_cache_without_legacy_columns_async(db)
    await db.execute("DELETE FROM MemoryV2SummaryCache WHERE cluster_summary_json = '{}'")
    await _delete_legacy_summary_input_packets_async(db)


def _rebuild_summary_cache_without_legacy_columns(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS MemoryV2SummaryCache__clean;
        CREATE TABLE MemoryV2SummaryCache__clean (
            summary_id TEXT PRIMARY KEY,
            packet_id TEXT NOT NULL,
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
        INSERT INTO MemoryV2SummaryCache__clean (
            summary_id, packet_id, input_hash, model, status, title, short_summary,
            digest_json, salient_entities_json, cluster_summary_json, created_at_ms,
            updated_at_ms, error_json
        )
        SELECT
            summary_id, packet_id, input_hash, model, status, title, short_summary,
            digest_json, salient_entities_json, cluster_summary_json, created_at_ms,
            updated_at_ms, error_json
        FROM MemoryV2SummaryCache;
        DROP TABLE MemoryV2SummaryCache;
        ALTER TABLE MemoryV2SummaryCache__clean RENAME TO MemoryV2SummaryCache;
        CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryCache_packet
        ON MemoryV2SummaryCache(packet_id, input_hash, status);
        """
    )


async def _rebuild_summary_cache_without_legacy_columns_async(db: Any) -> None:
    await db.executescript(
        """
        DROP TABLE IF EXISTS MemoryV2SummaryCache__clean;
        CREATE TABLE MemoryV2SummaryCache__clean (
            summary_id TEXT PRIMARY KEY,
            packet_id TEXT NOT NULL,
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
        INSERT INTO MemoryV2SummaryCache__clean (
            summary_id, packet_id, input_hash, model, status, title, short_summary,
            digest_json, salient_entities_json, cluster_summary_json, created_at_ms,
            updated_at_ms, error_json
        )
        SELECT
            summary_id, packet_id, input_hash, model, status, title, short_summary,
            digest_json, salient_entities_json, cluster_summary_json, created_at_ms,
            updated_at_ms, error_json
        FROM MemoryV2SummaryCache;
        DROP TABLE MemoryV2SummaryCache;
        ALTER TABLE MemoryV2SummaryCache__clean RENAME TO MemoryV2SummaryCache;
        CREATE INDEX IF NOT EXISTS idx_MemoryV2SummaryCache_packet
        ON MemoryV2SummaryCache(packet_id, input_hash, status);
        """
    )


def _delete_legacy_summary_input_packets(con: sqlite3.Connection) -> None:
    if not _table_exists(con, "MemoryV2SummaryInputs"):
        return
    legacy_packet_field = _legacy_summary_packet_field()
    legacy_prior_field = _legacy_summary_prior_field()
    packet_ids = [
        str(row[0])
        for row in con.execute(
            """
            SELECT packet_id
            FROM MemoryV2SummaryInputs
            WHERE packet_json LIKE ?
               OR packet_json LIKE ?
            """,
            (f'%"{legacy_packet_field}"%', f"%{legacy_prior_field}%"),
        )
    ]
    if not packet_ids:
        return
    placeholders = ",".join("?" * len(packet_ids))
    con.execute(f"DELETE FROM MemoryV2SummaryInputEvents WHERE packet_id IN ({placeholders})", packet_ids)
    con.execute(f"DELETE FROM MemoryV2SummaryInputRelations WHERE packet_id IN ({placeholders})", packet_ids)
    con.execute(f"DELETE FROM MemoryV2SummaryInputs WHERE packet_id IN ({placeholders})", packet_ids)


async def _delete_legacy_summary_input_packets_async(db: Any) -> None:
    async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='MemoryV2SummaryInputs'") as cur:
        exists = await cur.fetchone()
    if not exists:
        return
    legacy_packet_field = _legacy_summary_packet_field()
    legacy_prior_field = _legacy_summary_prior_field()
    async with db.execute(
        """
        SELECT packet_id
        FROM MemoryV2SummaryInputs
        WHERE packet_json LIKE ?
           OR packet_json LIKE ?
        """,
        (f'%"{legacy_packet_field}"%', f"%{legacy_prior_field}%"),
    ) as cur:
        rows = await cur.fetchall()
    packet_ids = [str(row[0]) for row in rows]
    if not packet_ids:
        return
    placeholders = ",".join("?" * len(packet_ids))
    await db.execute(f"DELETE FROM MemoryV2SummaryInputEvents WHERE packet_id IN ({placeholders})", packet_ids)
    await db.execute(f"DELETE FROM MemoryV2SummaryInputRelations WHERE packet_id IN ({placeholders})", packet_ids)
    await db.execute(f"DELETE FROM MemoryV2SummaryInputs WHERE packet_id IN ({placeholders})", packet_ids)


def _legacy_summary_json_column() -> str:
    return "summary_" + "card" + "_json"


def _legacy_summary_packet_field() -> str:
    return "summary_" + "card"


def _legacy_summary_prior_field() -> str:
    return "previous_" + "summary_stale_prior"


def _require_tables(con: sqlite3.Connection, names: set[str]) -> None:
    found = {str(row[0]) for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = names - found
    if missing:
        raise RuntimeError(f"missing required Memory V2 tables: {', '.join(sorted(missing))}")


def _load_roles(con: sqlite3.Connection, events: dict[int, EventRecord]) -> dict[int, list[RoleRecord]]:
    if not events:
        return {}
    placeholders = ",".join("?" * len(events))
    out: dict[int, list[RoleRecord]] = defaultdict(list)
    for row in con.execute(
        f"""
        SELECT event_id, role, entity, value_text
        FROM MemoryV2Participants
        WHERE event_id IN ({placeholders})
        """,
        list(events),
    ):
        out[int(row["event_id"])].append(RoleRecord(int(row["event_id"]), str(row["role"] or "").strip().lower(), str(row["entity"] or "").strip(), str(row["value_text"] or "").strip()))
    return dict(out)


def _load_sources(con: sqlite3.Connection, events: dict[int, EventRecord]) -> dict[str, set[int]]:
    if not events:
        return {}
    placeholders = ",".join("?" * len(events))
    out: dict[str, set[int]] = defaultdict(set)
    for row in con.execute(
        f"""
        SELECT event_id, source_uid
        FROM MemoryV2EventSources
        WHERE event_id IN ({placeholders}) AND source_uid <> ''
        """,
        list(events),
    ):
        out[str(row["source_uid"])].add(int(row["event_id"]))
    return {source: ids for source, ids in out.items() if len(ids) >= 2}


def _add_conversation_trace_relations(events, roles, max_gap_ms, add) -> None:
    by_conv: dict[tuple[str, str], list[EventRecord]] = defaultdict(list)
    for event in events.values():
        by_conv[(event.conv_type, event.conv_id)].append(event)
    for conv_events in by_conv.values():
        conv_events.sort(key=lambda item: (item.occurred_at, item.event_id))
        for index, event in enumerate(conv_events):
            previous_events = _previous_window(conv_events, index, max_gap_ms)
            if event.event_type_norm in ANSWER_TYPES:
                question = _nearest_event_of_type(previous_events, QUESTION_TYPES, roles, event.event_id, require_related=True)
                if question:
                    add(question.event_id, event.event_id, "answers", 0.74, {"rule": "question-answer-window"})
                search = _nearest_event_of_type(previous_events, SEARCH_TYPES, roles, event.event_id, require_related=True)
                if search:
                    add(search.event_id, event.event_id, "answers", 0.68, {"rule": "search-answer-window"})
            if event.event_type_norm in SEARCH_TYPES:
                question = _nearest_event_of_type(previous_events, QUESTION_TYPES, roles, event.event_id)
                if question:
                    add(question.event_id, event.event_id, "triggers", 0.70, {"rule": "question-search-window"})
            if event.event_type_norm in PARSE_TYPES:
                shared = _nearest_event_of_type(previous_events, SHARE_TYPES, roles, event.event_id)
                if shared:
                    add(shared.event_id, event.event_id, "triggers", 0.76, {"rule": "share-parse-window"})
            if event.event_type_norm in DECISION_TYPES:
                observation = _nearest_event_of_type(previous_events, OBSERVE_TYPES, roles, event.event_id)
                if observation:
                    add(observation.event_id, event.event_id, "causes", 0.62, {"rule": "observation-decision-window"})
                    add(event.event_id, observation.event_id, "caused_by", 0.62, {"rule": "observation-decision-window"})


def _add_correction_relations(events, roles, max_gap_ms, add) -> None:
    by_conv: dict[tuple[str, str], list[EventRecord]] = defaultdict(list)
    for event in events.values():
        by_conv[(event.conv_type, event.conv_id)].append(event)
    for conv_events in by_conv.values():
        conv_events.sort(key=lambda item: (item.occurred_at, item.event_id))
        for index, event in enumerate(conv_events):
            if not _looks_like_correction(event):
                continue
            candidate = _nearest_related_event(_previous_window(conv_events, index, max_gap_ms), roles, event.event_id)
            if not candidate:
                continue
            relation_type = "refutes" if event.event_type_norm in {"deny", "correct", "refute"} or "不是" in event.summary else "corrects"
            add(event.event_id, candidate.event_id, relation_type, 0.66, {"rule": "correction-marker-window"})
            add(event.event_id, candidate.event_id, "possibly_refers_to", 0.42, {"rule": "correction-target-candidate"})
            if relation_type == "corrects":
                add(event.event_id, candidate.event_id, "supersedes", 0.55, {"rule": "correction-marker-window"})


def _nearest_event_of_type(previous_events, event_types, roles, target_event_id, *, require_related: bool = False):
    candidates = [event for event in previous_events if event.event_type_norm in event_types]
    related = [event for event in candidates if _has_trace_relatedness(roles, event.event_id, target_event_id)]
    if related:
        return related[-1]
    if require_related:
        return None
    return candidates[-1] if candidates else None


def _nearest_related_event(previous_events, roles, target_event_id):
    for event in reversed(previous_events):
        if _has_trace_relatedness(roles, event.event_id, target_event_id):
            return event
    return None


def _previous_window(conv_events, index: int, max_gap_ms: int):
    event = conv_events[index]
    out = []
    for candidate in reversed(conv_events[:index]):
        if event.occurred_at - candidate.occurred_at > max_gap_ms:
            break
        out.append(candidate)
    return list(reversed(out))


def _has_trace_relatedness(roles, left_event_id: int, right_event_id: int) -> bool:
    return bool(_trace_entities_for(roles, left_event_id) & _trace_entities_for(roles, right_event_id))


def _trace_entities_for(roles, event_id: int) -> set[str]:
    return {role.entity for role in roles.get(event_id, ()) if _is_trace_entity(role.role, role.entity)}


def _is_trace_entity(role: str, entity: str) -> bool:
    if not entity or entity in {"self", "Person:self", "person:self"}:
        return False
    role_key = str(role or "").lower()
    if role_key in {"location", "time", "session", "platform", "source"}:
        return False
    return not entity.startswith(("Group:", "Platform:", "Location:", "Time:", "Session:", "group:", "platform:", "location:", "time:", "session:"))


def _looks_like_correction(event: EventRecord) -> bool:
    summary = event.summary or ""
    if any(marker in summary for marker in NON_CORRECTION_NEGATION_MARKERS):
        return event.event_type_norm in CORRECTION_TYPES and _has_any(summary, STRONG_CORRECTION_MARKERS)
    if event.event_type_norm in CORRECTION_TYPES:
        return True
    return _has_any(summary, STRONG_CORRECTION_MARKERS)


def _episode_type_for(relations: list[EventRelation]) -> str:
    types = {relation.relation_type for relation in relations}
    if types & {"corrects", "refutes", "supersedes"}:
        return "revision"
    if types & {"answers", "triggers"}:
        return "qa_trace"
    if types & {"causes", "caused_by"}:
        return "decision_trace"
    return "source_episode"


def _episode_member_role(event_id: int, relations: list[EventRelation]) -> str:
    outgoing = {relation.relation_type for relation in relations if relation.source_event_id == event_id}
    incoming = {relation.relation_type for relation in relations if relation.target_event_id == event_id}
    if outgoing & {"corrects", "refutes", "supersedes"}:
        return "correction"
    if incoming & {"corrects", "refutes", "supersedes"}:
        return "corrected"
    if outgoing & {"answers", "triggers", "causes", "caused_by"}:
        return "source"
    if incoming & {"answers", "triggers", "causes", "caused_by"}:
        return "result"
    return "member"


def _cluster_groups(events: dict[int, EventRecord], roles: dict[int, list[RoleRecord]]):
    by_conv_bucket: dict[str, set[int]] = defaultdict(set)
    by_anchor: dict[str, set[int]] = defaultdict(set)
    by_title: dict[str, set[int]] = defaultdict(set)
    for event in events.values():
        bucket = event.occurred_at // (30 * 60 * 1000) if event.occurred_at else 0
        by_conv_bucket[f"session:{event.conv_type}:{event.conv_id}:bucket:{bucket}"].add(event.event_id)
        for title in TITLE_RE.findall(event.summary):
            if 2 <= len(title) <= 40:
                by_title[f"title:{title}"].add(event.event_id)
    for event_id, event_roles in roles.items():
        for role in event_roles:
            if _is_cache_anchor_excluded(role.role, role.entity):
                continue
            by_anchor[f"role_entity:{role.role}:{role.entity}"].add(event_id)
    for anchor, ids in by_conv_bucket.items():
        if 2 <= len(ids) <= 6:
            yield ids, "session", "session_fragment_precise", "session", anchor, 0.42
    for anchor, ids in by_anchor.items():
        if 2 <= len(ids) <= 6:
            yield ids, "recurrent-anchor", "recurrent_anchor_candidate", "topic-strict", anchor, 0.45
    for anchor, ids in by_title.items():
        if 2 <= len(ids) <= 16:
            yield ids, "title", "title_topic_precise", "title", anchor, 0.72


def _is_cache_anchor_excluded(role: str, entity: str) -> bool:
    role = str(role or "").strip().lower()
    entity = str(entity or "").strip()
    if not entity or entity.lower() in {"self", "person:self", "bot", "model"}:
        return True
    if role in {"location", "time", "session", "platform", "source"}:
        return True
    return entity.startswith(("Group:", "Platform:", "Location:", "Time:", "Session:", "group:", "platform:", "location:", "time:", "session:"))


def _classify_mount(card: ClusterSummaryRecord, atom: MemoryAtom) -> tuple[str, float, dict[str, Any], str]:
    text = f"{atom.summary} {atom.event_type_norm}".strip()
    text_tokens = set(_tokens(text))
    card_tokens = set(_tokens(" ".join((card.title, card.short_summary, *card.confirmed_claims))))
    atom_entities = {_canonical_entity(item) for item in atom.entities if _is_informative_entity(item)}
    card_entities = {_canonical_entity(item) for item in card.core_entities if _is_informative_entity(item)}
    entity_overlap = atom_entities & card_entities
    non_person_overlap = {item for item in entity_overlap if not item.startswith("person:")}
    token_overlap = text_tokens & card_tokens
    slot_text = " ".join(card.open_slots)
    evidence = {"entity_overlap": sorted(entity_overlap), "token_overlap": sorted(token_overlap)[:12], "open_slots": list(card.open_slots), "atom_source": atom.source}
    has_correction = _is_explicit_correction(text, atom.event_type_norm)
    has_progress = _has_any(text, PROGRESS_MARKERS)
    has_answer = _has_any(text, ANSWER_MARKERS) or atom.event_type_norm in {"answer", "reply", "respond"}
    base = 0.0
    if entity_overlap:
        base += min(0.45, 0.22 + 0.12 * len(entity_overlap))
    if token_overlap:
        base += min(0.25, 0.05 * len(token_overlap))
    if card.title and card.title in atom.summary:
        base += 0.20
    if has_correction and entity_overlap:
        relation_type = "corrects_identity" if "identity" in slot_text or "不是指" in text or "not the same" in text else "refutes"
        return relation_type, min(max(0.78, base + 0.34), 0.96), evidence, ""
    if has_progress and entity_overlap:
        supports_progress = any(marker in slot_text for marker in ("progress", "growth", "completion", "result", "blocked", "state"))
        if not supports_progress:
            return "background_only", max(0.38, min(base + 0.06, 0.58)), evidence, "progress-like wording but anchor has no progress slot"
        return "updates_state", min(max(0.68, base + 0.22), 0.94), evidence, ""
    if has_answer and entity_overlap:
        if "answer" not in slot_text and "follow_up" not in slot_text:
            return "background_only", max(0.38, min(base + 0.06, 0.58)), evidence, "answer-like wording but anchor has no answer slot"
        if not non_person_overlap and len(token_overlap) < 2:
            return "background_only", max(0.36, min(base + 0.04, 0.52)), evidence, "answer lacks topic/object overlap"
        return "answers", min(max(0.64, base + 0.18), 0.90), evidence, ""
    if entity_overlap and _has_any(text, BACKGROUND_MARKERS):
        return "background_only", max(0.40, min(base + 0.08, 0.58)), evidence, "shared entity without state, answer, or correction evidence"
    if entity_overlap and base >= 0.50:
        return "same_object", min(base, 0.74), evidence, "shared entities but relation type is not explicit"
    if token_overlap and len(token_overlap) >= 2:
        return "background_only", min(0.44, 0.25 + 0.04 * len(token_overlap)), evidence, "text overlap only"
    return "unrelated", base, evidence, "no anchor evidence"


def _load_pending_mounts(con: sqlite3.Connection, *, max_mounts: int) -> list[MemoryMount]:
    con.row_factory = sqlite3.Row
    rows = list(
        con.execute(
            """
            SELECT *
            FROM MemoryV2MemoryMounts
            WHERE status='pending'
            ORDER BY
              CASE relation_type
                WHEN 'corrects_identity' THEN 0
                WHEN 'corrects' THEN 1
                WHEN 'refutes' THEN 2
                WHEN 'updates_state' THEN 3
                WHEN 'progresses' THEN 4
                WHEN 'background_only' THEN 9
                ELSE 5
              END,
              confidence DESC,
              created_at_ms ASC
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
            FROM MemoryV2LocalClusterMounts
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


def _local_cluster_mount_payload(mounts: list[LocalClusterMount]) -> dict[str, Any]:
    decisions = []
    counts: dict[str, int] = defaultdict(int)
    for mount in mounts:
        decision = "accept_local_cluster" if len(set(mount.event_ids)) >= 2 else "reject_invalid_local_cluster"
        counts[decision] += 1
        decisions.append(
            {
                "proposal_id": mount.proposal_id,
                "decision": decision,
                "event_ids": list(mount.event_ids),
                "confidence": mount.confidence,
                "reason": mount.evidence_text,
            }
        )
    return {
        "pending_local_cluster_mounts_loaded": len(mounts),
        "local_cluster_decision_counts": dict(counts),
        "local_cluster_decisions": decisions,
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
        cur = con.execute(
            """
            INSERT INTO MemoryV2ClusterRuns (
                profile, trigger, started_at, finished_at, min_event_id, max_event_id, params_json
            ) VALUES ('sleep-consolidated', 'local_cluster_mount', ?, ?, ?, ?, ?)
            """,
            (
                now_ms,
                now_ms,
                min(min(cluster.event_ids) for cluster in clusters),
                max(max(cluster.event_ids) for cluster in clusters),
                _json({"generator": "sleep_mount_consolidation.local_cluster"}),
            ),
        )
        write_cluster_cache(con, clusters, members, run_id=int(cur.lastrowid), now_ms=now_ms)
    status_rows = 0
    for status, items in (("accepted", accepted), ("rejected", rejected)):
        if not items:
            continue
        placeholders = ",".join("?" * len(items))
        con.execute(
            f"""
            UPDATE MemoryV2LocalClusterMounts
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
            FROM MemoryV2Events
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
            FROM MemoryV2SummaryCache
            WHERE packet_id=? AND status='ready' AND cluster_summary_json <> '{}'
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
            FROM MemoryV2SummaryCache
            WHERE packet_id=? AND cluster_summary_json <> '{}'
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
        row = con.execute(
            """
            SELECT packet_json
            FROM MemoryV2SummaryInputs
            WHERE packet_id=?
            ORDER BY updated_at_ms DESC
            LIMIT 1
            """,
            (summary_id,),
        ).fetchone()
        if row:
            card = _cluster_summary_from_packet_json(str(row[0]))
            if card:
                cards.append(card)
    return cards


def _cluster_summary_from_packet_json(packet_json: str) -> ClusterSummaryRecord | None:
    try:
        packet = json.loads(packet_json)
    except json.JSONDecodeError:
        return None
    card = packet.get("cluster_summary") if isinstance(packet, dict) else None
    if isinstance(card, dict):
        return cluster_summary_from_json(card)
    return None


def _load_cluster_relations(con: sqlite3.Connection, cluster_ids: set[str]) -> list[ClusterRelation]:
    if not cluster_ids:
        return []
    out: list[ClusterRelation] = []
    for cluster_id in sorted(cluster_ids):
        for row in con.execute(
            """
            SELECT *
            FROM MemoryV2ClusterRelations
            WHERE cluster_id=? AND status IN ('active', 'weak')
            """,
            (cluster_id,),
        ):
            try:
                evidence = json.loads(str(row[10] or "{}"))
            except json.JSONDecodeError:
                evidence = {}
            out.append(ClusterRelation(str(row[0]), str(row[1]), int(row[2]), int(row[3]), str(row[4]), float(row[5] or 0.0), str(row[6]), int(row[7] or 1), evidence if isinstance(evidence, dict) else {}))
    return out


def _relations_rejected_by_correction(relations: Iterable[ClusterRelation]) -> list[ClusterRelation]:
    rejectable = {"same_object", "comments_on", "possibly_refers_to", "updates_state", "progresses"}
    return [relation for relation in relations if relation.status == "active" and relation.relation_type in rejectable]


def _decision_mount_status(decisions: Iterable[SleepDecision]) -> dict[str, str]:
    status_by_decision = {
        "accept_attach": "accepted",
        "accept_with_uncertainty": "accepted",
        "revise_existing_relation": "accepted",
        "reject_background": "rejected",
        "reject_wrong_anchor": "obsolete",
    }
    return {item.mount_id: status_by_decision.get(item.decision, "pending") for item in decisions}


def _mark_summary_cache_stale(con: sqlite3.Connection, summary_ids: Iterable[str], *, now_ms: int) -> int:
    changed = 0
    for summary_id in summary_ids:
        cur = con.execute(
            """
            UPDATE MemoryV2SummaryCache
            SET status='stale', updated_at_ms=?
            WHERE packet_id=? AND status='ready'
            """,
            (now_ms, summary_id),
        )
        changed += cur.rowcount if cur.rowcount is not None else 0
    return changed


def _update_thread_states(con: sqlite3.Connection, relations: Iterable[ClusterRelation], *, now_ms: int) -> int:
    changed = 0
    for relation in relations:
        if relation.status not in {"active", "weak"}:
            continue
        if relation.relation_type not in {"continues", "updates_state", "progresses", "causes_or_results"}:
            continue
        anchor_kind = str(relation.evidence.get("anchor_source_kind") or "")
        if anchor_kind != "thread" and not relation.cluster_id.startswith("thread:"):
            continue
        thread_id = relation.cluster_id
        existing = con.execute("SELECT state_json, revision FROM MemoryV2ThreadStates WHERE thread_id=?", (thread_id,)).fetchone()
        before_json = str(existing[0]) if existing else "{}"
        before_revision = int(existing[1]) if existing else 0
        try:
            state = json.loads(before_json)
        except json.JSONDecodeError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        milestones = list(state.get("milestones") or [])
        milestones.append({"event_id": relation.source_event_id, "relation_type": relation.relation_type, "state": _state_for_relation(relation)})
        state.update(
            {
                "thread_id": thread_id,
                "state": _state_for_relation(relation),
                "milestones": milestones,
                "open_slots": _thread_open_slots_for_state(_state_for_relation(relation)),
            }
        )
        after_json = _json(state)
        con.execute(
            """
            INSERT INTO MemoryV2ThreadStates (thread_id, thread_key, status, state_json, revision, updated_at_ms)
            VALUES (?, ?, 'active', ?, 1, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                state_json=excluded.state_json,
                revision=MemoryV2ThreadStates.revision + 1,
                updated_at_ms=excluded.updated_at_ms
            """,
            (thread_id, thread_id.removeprefix("thread:"), after_json, now_ms),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO MemoryV2ThreadStateRevisions (
                revision_id, thread_id, revision_type, triggered_by_mount_id, before_json, after_json, created_at_ms
            ) VALUES (?, ?, 'mount_update', ?, ?, ?, ?)
            """,
            (_sha1("thread-revision", thread_id, str(before_revision + 1), str(relation.source_event_id))[:24], thread_id, str(relation.evidence.get("mount_id") or ""), before_json, after_json, now_ms),
        )
        changed += 1
    return changed


def _queue_summary_refresh_inputs(con: sqlite3.Connection, summary_ids: Iterable[str], *, now_ms: int) -> int:
    queued = 0
    for card in _load_summary_prior_cards_for_refresh(con, set(summary_ids)):
        cluster_id = card.source_id or card.summary_id
        relations = []
        for row in con.execute(
            """
            SELECT relation_id, source_event_id, target_event_id, relation_type, status, confidence
            FROM MemoryV2ClusterRelations
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
            card,
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
            for event_id in card.source_event_ids
        ]
        packet_id = f"summary-refresh:{card.summary_id}"
        packet = {
            "packet_id": packet_id,
            "packet_type": "summary_refresh_input",
            "source_kind": card.source_kind,
            "source_id": card.source_id,
            "summary_task": (
                "Refresh the cluster summary from the event window. The previous summary is a stale prior only; "
                "newer event evidence later in the window wins on conflict."
            ),
            "window_policy": {
                "order": "chronological_old_to_new",
                "selection": "mandatory_delta_then_activation_score",
                "max_events": SUMMARY_REFRESH_EVENT_WINDOW_LIMIT,
                "token_budget_estimate": SUMMARY_REFRESH_EVENT_TOKEN_BUDGET,
                "activation_score_is_relevance_not_truth": True,
            },
            "previous_cluster_summary_stale_prior": asdict(card),
            "cluster_summary": asdict(card),
            "relations": relations,
            "events": input_events,
            "provenance": {"llm_used": False, "generator": "memory.consolidation"},
        }
        input_hash = _sha1("summary-input", _json(packet))
        con.execute(
            """
            INSERT INTO MemoryV2SummaryInputs (
                packet_id, packet_type, source_kind, source_id, source_revision, input_hash, priority,
                confidence_tier, status, created_at_ms, updated_at_ms, packet_json, invalidation_json, provenance_json
            ) VALUES (?, 'summary_refresh_input', ?, ?, ?, ?, 90, 'high', 'active', ?, ?, ?, ?, ?)
            ON CONFLICT(packet_id) DO UPDATE SET
                source_revision=excluded.source_revision,
                input_hash=excluded.input_hash,
                status='active',
                updated_at_ms=excluded.updated_at_ms,
                packet_json=excluded.packet_json,
                invalidation_json=excluded.invalidation_json,
                provenance_json=excluded.provenance_json
            """,
            (
                packet_id,
                card.source_kind,
                card.source_id,
                card.revision,
                input_hash,
                now_ms,
                now_ms,
                _json(packet),
                _json({"summary_id": card.summary_id, "source_revision": card.revision}),
                _json({"llm_used": False, "generator": "memory.consolidation"}),
            ),
        )
        con.execute("DELETE FROM MemoryV2SummaryInputEvents WHERE packet_id=?", (packet_id,))
        con.executemany(
            """
            INSERT OR REPLACE INTO MemoryV2SummaryInputEvents (packet_id, event_id, rank, role, status)
            VALUES (?, ?, ?, ?, 'active')
            """,
            [
                (
                    packet_id,
                    int(item["event_id"]),
                    index,
                    str(item.get("window_role") or "source"),
                )
                for index, item in enumerate(input_events, start=1)
                if int(item.get("event_id") or 0) > 0
            ],
        )
        con.execute("DELETE FROM MemoryV2SummaryInputRelations WHERE packet_id=?", (packet_id,))
        con.executemany(
            """
            INSERT OR REPLACE INTO MemoryV2SummaryInputRelations (
                packet_id, relation_id, source_event_id, target_event_id, relation_type, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(packet_id, rel["relation_id"], rel["source_event_id"], rel["target_event_id"], rel["relation_type"], rel["status"]) for rel in relations],
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
    if not _table_exists(con, "MemoryV2Events"):
        return []

    delta_ids = {
        int(rel.get("source_event_id") or 0)
        for rel in relations
        if str(rel.get("status") or "") in {"active", "weak"}
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

    event_columns = _table_columns(con, "MemoryV2Events")
    access_expr = "access_count" if "access_count" in event_columns else "0 AS access_count"
    placeholders = ",".join("?" * len(wanted_ids))
    rows = list(
        con.execute(
            f"""
            SELECT event_id, summary, event_type_norm, status, confidence,
                   occurred_at, created_at, last_seen_at, last_accessed,
                   occurrences, {access_expr}
            FROM MemoryV2Events
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
    if not event_ids or not _table_exists(con, "MemoryV2Participants"):
        return {}
    placeholders = ",".join("?" * len(event_ids))
    roles: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in con.execute(
        f"""
        SELECT event_id, role, entity, value_text
        FROM MemoryV2Participants
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


def _state_for_relation(relation: ClusterRelation) -> str:
    text = str(relation.evidence.get("evidence_text") or "").lower()
    if any(marker in text for marker in ("complete", "finished", "done", "beat", "白金", "通关", "完成")):
        return "completed"
    if relation.relation_type in {"updates_state", "progresses", "continues"}:
        return "in_progress"
    return "updated"


def _thread_open_slots_for_state(state: str) -> list[str]:
    if state == "completed":
        return ["post_completion_comment", "correction"]
    if state == "in_progress":
        return ["progress_update", "blocked_point", "completion", "correction"]
    return ["new_evidence", "correction"]


def _mount_sort_key(mount: MemoryMount) -> tuple[int, float, str]:
    return (-_relation_priority(mount.relation_type), -mount.confidence, mount.mount_id)


def _relation_priority(relation_type: str) -> int:
    return {
        "corrects_identity": 100,
        "corrects": 95,
        "refutes": 95,
        "updates_state": 80,
        "progresses": 75,
        "causes_or_results": 70,
        "answers": 65,
        "same_object": 45,
        "background_only": 10,
        "unrelated": 0,
    }.get(relation_type, 0)


def _parse_prefixed_entity(raw_entity: str) -> tuple[str, str]:
    raw = str(raw_entity or "").strip()
    match = PREFIX_RE.match(raw)
    if not match:
        return "", raw
    return match.group(1).strip(), match.group(2).strip()


def _normalize_entity_name(name: str) -> str:
    text = str(name or "").strip().translate(SCRIPT_FOLD)
    text = text.replace("未來", "未来").replace("（", "(").replace("）", ")")
    return SPACE_RE.sub("", text)


def _canonical_type_for(raw_type: str, role: str, normalized_name: str) -> tuple[str, list[str], float]:
    raw_type_key = str(raw_type or "").strip().lower()
    role_key = str(role or "").strip().lower()
    mapped = TYPE_MAP.get(raw_type_key, "")
    if normalized_name.lower() in {"self", "person:self", "model"}:
        return "person", ["self-alias"], 0.99
    if mapped == "platform" and _looks_like_group_name(normalized_name):
        return "group", ["platform-group-name"], 0.86
    if mapped == "location" and role_key in {"location", "session"} and _looks_like_group_name(normalized_name):
        return "group", ["location-group-alias"], 0.78
    if mapped:
        return mapped, [f"prefix:{raw_type_key}"], 0.92
    if role_key in {"agent", "recipient", "speaker", "experiencer"}:
        return "person", ["role-person"], 0.62
    if role_key in {"instrument", "tool"}:
        return "tool", ["role-tool"], 0.62
    if role_key in {"theme", "topic"}:
        return "concept", ["role-concept"], 0.58
    if role_key in {"location", "session"} and _looks_like_group_name(normalized_name):
        return "group", ["role-location-group"], 0.62
    return "unknown", ["fallback-unknown"], 0.45


def _looks_like_group_name(name: str) -> bool:
    text = str(name or "").strip()
    return bool(text) and any(marker in text for marker in ("群", "资源社", "测试群", "社群"))


def _merge_suspicion_for(left: ResolvedEntity, right: ResolvedEntity) -> MergeSuspicion | None:
    left_name = left.canonical_name
    right_name = right.canonical_name
    if left_name == right_name or min(len(left_name), len(right_name)) < 3:
        return None
    shorter, longer = sorted((left_name.lower(), right_name.lower()), key=len)
    contained = shorter in longer and len(longer) - len(shorter) <= 8
    prefix = _common_prefix_ratio(left_name.lower(), right_name.lower())
    score = max(0.78 if contained else 0.0, prefix)
    if score < 0.78:
        return None
    left_id, right_id = sorted((left.entity_id, right.entity_id))
    suspicion_type = "contained-name" if contained else "name-prefix"
    return MergeSuspicion(_sha1("suspicion", left_id, right_id, suspicion_type)[:20], left_id, right_id, suspicion_type, round(score, 6), "pending", _json({"left_name": left_name, "right_name": right_name, "rule": suspicion_type}))


def _common_prefix_ratio(left: str, right: str) -> float:
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n / max(len(left), len(right), 1)


def _raw_session_entity(event: EventRecord) -> str:
    if not event.conv_type and not event.conv_id:
        return ""
    return f"Session:{event.conv_type}:{event.conv_id}"


def _entity_id_for(entity_type: str, canonical_name: str) -> str:
    return f"{entity_type}:{_sha1('entity', entity_type, canonical_name)[:16]}"


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(text or ""))]


def _has_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _is_explicit_correction(text: str, event_type_norm: str) -> bool:
    if str(event_type_norm or "").strip().lower() in {"correct", "deny", "refute", "supersede"}:
        return True
    return _has_any(text, STRONG_CORRECTION_MARKERS)


def _canonical_entity(entity: str) -> str:
    return str(entity or "").strip().lower().replace(" ", "")


def _is_informative_entity(entity: str) -> bool:
    entity = str(entity or "").strip()
    if not entity or entity.lower() in {"self", "bot", "person:self"}:
        return False
    return not entity.startswith(CONTEXT_ENTITY_PREFIXES)


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
    "LocalClusterMount",
    "MemoryAtom",
    "MemoryMount",
    "PREPROCESSING_SCHEMA_SQL",
    "RoleRecord",
    "SleepDecision",
    "ClusterSummaryRecord",
    "build_entity_resolution",
    "canonicalize_roles",
    "consolidate_memory_mounts",
    "ensure_preprocessing_schema",
    "ensure_preprocessing_schema_async",
    "load_memory_v2_dataset",
    "propose_memory_mounts",
    "run_mount_consolidation",
    "run_preprocessing",
    "stage_memory_mount_candidates",
    "cluster_summary_from_json",
    "cluster_summary_to_json",
    "write_consolidation_result",
    "write_local_cluster_mounts",
    "write_memory_mounts",
]
