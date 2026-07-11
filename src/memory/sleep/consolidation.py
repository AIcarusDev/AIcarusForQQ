"""Deterministic Memory preprocessing and candidate-storyline consolidation.

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
from collections import defaultdict
from dataclasses import asdict, dataclass
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
CREATE TABLE IF NOT EXISTS MemoryStorylineRuns (
    run_id INTEGER PRIMARY KEY,
    profile TEXT NOT NULL,
    trigger TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER NOT NULL DEFAULT 0,
    min_event_id INTEGER NOT NULL DEFAULT 0,
    max_event_id INTEGER NOT NULL DEFAULT 0,
    params_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS MemoryStorylines (
    storyline_id TEXT PRIMARY KEY,
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
CREATE TABLE IF NOT EXISTS MemoryStorylineMembers (
    storyline_id TEXT NOT NULL,
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
    PRIMARY KEY (storyline_id, event_id)
);
CREATE TABLE IF NOT EXISTS MemoryStorylineMemberRevisions (
    revision_id TEXT PRIMARY KEY,
    storyline_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    revision_event_id INTEGER NOT NULL DEFAULT 0,
    revision_type TEXT NOT NULL,
    status_before TEXT NOT NULL,
    status_after TEXT NOT NULL,
    run_id INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylines_scope
ON MemoryStorylines(scope, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineMembers_event
ON MemoryStorylineMembers(event_id, score DESC);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineMembers_status
ON MemoryStorylineMembers(status, storyline_id);

DROP TABLE IF EXISTS MemoryMounts;
DROP TABLE IF EXISTS MemoryLocalStorylineMounts;

CREATE TABLE IF NOT EXISTS MemoryCandidateStorylines (
    candidate_storyline_id TEXT PRIMARY KEY,
    event_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_MemoryCandidateStorylines_status
ON MemoryCandidateStorylines(status, created_at_ms);

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
CREATE TABLE IF NOT EXISTS MemoryStorylineRelations (
    relation_id TEXT PRIMARY KEY,
    storyline_id TEXT NOT NULL,
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
CREATE TABLE IF NOT EXISTS MemoryStorylineRevisions (
    revision_id TEXT PRIMARY KEY,
    storyline_id TEXT NOT NULL,
    revision_type TEXT NOT NULL,
    before_revision INTEGER NOT NULL DEFAULT 0,
    after_revision INTEGER NOT NULL DEFAULT 0,
    triggered_by_mount_id TEXT NOT NULL DEFAULT '',
    triggered_by_event_id INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineRelations_storyline
ON MemoryStorylineRelations(storyline_id, status, relation_type);

CREATE TABLE IF NOT EXISTS MemoryStorylineSummaryTasks (
    task_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    storyline_id TEXT NOT NULL,
    storyline_revision INTEGER NOT NULL DEFAULT 0,
    input_hash TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    confidence_tier TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS MemoryStorylineSummaryTaskEvents (
    task_id TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    rank INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (task_id, event_id)
);
CREATE TABLE IF NOT EXISTS MemoryStorylineSummaryTaskRelations (
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
    storyline_summary_json TEXT NOT NULL DEFAULT '{}',
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL DEFAULT 0,
    error_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineSummaryTasks_source
ON MemoryStorylineSummaryTasks(storyline_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineSummaryTasks_queue
ON MemoryStorylineSummaryTasks(status, priority DESC, updated_at_ms);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineSummaryTaskEvents_event
ON MemoryStorylineSummaryTaskEvents(event_id, status);
CREATE INDEX IF NOT EXISTS idx_MemoryStorylineSummaryTaskRelations_relation
ON MemoryStorylineSummaryTaskRelations(relation_id, status);
CREATE INDEX IF NOT EXISTS idx_MemorySummaryCache_task
ON MemorySummaryCache(task_id, input_hash, status);
"""

_STORYLINE_DATA_MIGRATION_SQL = """
UPDATE MemoryCandidateStorylines
SET candidate_storyline_id = 'candidate_storyline:' || substr(candidate_storyline_id, 11)
WHERE candidate_storyline_id LIKE 'candidate:%';

UPDATE MemoryStorylineMembers
SET storyline_id = 'candidate_storyline:' || substr(storyline_id, 9)
WHERE storyline_id LIKE 'episode:%';
UPDATE MemoryStorylineMemberRevisions
SET storyline_id = 'candidate_storyline:' || substr(storyline_id, 9)
WHERE storyline_id LIKE 'episode:%';
UPDATE MemoryStorylineRelations
SET storyline_id = 'candidate_storyline:' || substr(storyline_id, 9)
WHERE storyline_id LIKE 'episode:%';
UPDATE MemoryStorylineRevisions
SET storyline_id = 'candidate_storyline:' || substr(storyline_id, 9)
WHERE storyline_id LIKE 'episode:%';
UPDATE MemoryStorylineSummaryTasks
SET storyline_id = 'candidate_storyline:' || substr(storyline_id, 9)
WHERE storyline_id LIKE 'episode:%';
UPDATE MemoryStorylines
SET storyline_id = 'candidate_storyline:' || substr(storyline_id, 9),
    scope = CASE WHEN scope='episode' THEN 'candidate_storyline' ELSE scope END,
    scheme_name = CASE WHEN scheme_name='llm_episode_candidate' THEN 'llm_candidate_storyline' ELSE scheme_name END,
    anchor_key = CASE
        WHEN anchor_key LIKE 'episode:%' THEN 'candidate_storyline:' || substr(anchor_key, 9)
        ELSE anchor_key
    END
WHERE storyline_id LIKE 'episode:%' OR scope='episode' OR scheme_name='llm_episode_candidate';

UPDATE MemoryStorylineSummaryTaskEvents
SET task_id = replace(task_id, 'summary:cluster:', 'summary:storyline:')
WHERE task_id LIKE 'summary:cluster:%';
UPDATE MemoryStorylineSummaryTaskRelations
SET task_id = replace(task_id, 'summary:cluster:', 'summary:storyline:')
WHERE task_id LIKE 'summary:cluster:%';
UPDATE MemoryStorylineSummaryTasks
SET task_id = replace(task_id, 'summary:cluster:', 'summary:storyline:')
WHERE task_id LIKE 'summary:cluster:%';
UPDATE MemorySummaryCache
SET summary_id = replace(summary_id, 'summary:cluster:', 'summary:storyline:'),
    task_id = replace(task_id, 'summary:cluster:', 'summary:storyline:'),
    storyline_summary_json = replace(
        replace(storyline_summary_json, '"source_kind":"cluster"', '"source_kind":"storyline"'),
        'summary:cluster:',
        'summary:storyline:'
    )
WHERE summary_id LIKE 'summary:cluster:%'
   OR task_id LIKE 'summary:cluster:%'
   OR storyline_summary_json LIKE '%"source_kind":"cluster"%';

DROP TABLE IF EXISTS MemoryEpisodeMembers;
DROP TABLE IF EXISTS MemoryEpisodes;
DROP TABLE IF EXISTS MemoryClusterSummaryTaskRelations;
DROP TABLE IF EXISTS MemoryClusterSummaryTaskEvents;
DROP TABLE IF EXISTS MemoryClusterSummaryTasks;
DROP TABLE IF EXISTS MemoryClusterRevisions;
DROP TABLE IF EXISTS MemoryClusterRelations;
DROP TABLE IF EXISTS MemoryEpisodeCandidates;
DROP TABLE IF EXISTS MemoryClusterMemberRevisions;
DROP TABLE IF EXISTS MemoryClusterMembers;
DROP TABLE IF EXISTS MemoryClusters;
DROP TABLE IF EXISTS MemoryClusterRuns;
"""

_LEGACY_SUMMARY_QUEUE_TABLES = (
    "MemorySummaryInputRelations",
    "MemorySummaryInputEvents",
    "MemorySummaryInputs",
)
_STORYLINE_TABLE_RENAMES = (
    ("MemoryClusterRuns", "MemoryStorylineRuns"),
    ("MemoryClusters", "MemoryStorylines"),
    ("MemoryClusterMembers", "MemoryStorylineMembers"),
    ("MemoryClusterMemberRevisions", "MemoryStorylineMemberRevisions"),
    ("MemoryEpisodeCandidates", "MemoryCandidateStorylines"),
    ("MemoryClusterRelations", "MemoryStorylineRelations"),
    ("MemoryClusterRevisions", "MemoryStorylineRevisions"),
    ("MemoryClusterSummaryTasks", "MemoryStorylineSummaryTasks"),
    ("MemoryClusterSummaryTaskEvents", "MemoryStorylineSummaryTaskEvents"),
    ("MemoryClusterSummaryTaskRelations", "MemoryStorylineSummaryTaskRelations"),
)
_STORYLINE_COLUMN_RENAMES = {
    "MemoryStorylines": (("cluster_id", "storyline_id"),),
    "MemoryStorylineMembers": (("cluster_id", "storyline_id"),),
    "MemoryStorylineMemberRevisions": (("cluster_id", "storyline_id"),),
    "MemoryStorylineRelations": (("cluster_id", "storyline_id"),),
    "MemoryStorylineRevisions": (("cluster_id", "storyline_id"),),
    "MemoryStorylineSummaryTasks": (
        ("cluster_id", "storyline_id"),
        ("cluster_revision", "storyline_revision"),
    ),
    "MemorySummaryCache": (("cluster_summary_json", "storyline_summary_json"),),
    "MemoryCandidateStorylines": (("candidate_id", "candidate_storyline_id"),),
}
_STALE_STORYLINE_INDEXES = (
    "idx_MemoryClusters_scope",
    "idx_MemoryClusterMembers_event",
    "idx_MemoryClusterMembers_status",
    "idx_MemoryEpisodeCandidates_status",
    "idx_MemoryClusterRelations_cluster",
    "idx_MemoryClusterSummaryTasks_source",
    "idx_MemoryClusterSummaryTasks_queue",
    "idx_MemoryClusterSummaryTaskEvents_event",
    "idx_MemoryClusterSummaryTaskRelations_relation",
)
_LEGACY_SUMMARY_QUEUE_COLUMNS = {
    "MemorySummaryInputs": {
        "packet_id",
        "packet_type",
        "source_kind",
        "source_id",
        "source_revision",
        "input_hash",
        "priority",
        "confidence_tier",
        "status",
        "created_at_ms",
        "updated_at_ms",
    },
    "MemorySummaryInputEvents": {"packet_id", "event_id", "rank", "role", "status"},
    "MemorySummaryInputRelations": {
        "packet_id",
        "relation_id",
        "source_event_id",
        "target_event_id",
        "relation_type",
        "status",
    },
}
_LEGACY_SUMMARY_QUEUE_MIGRATION_SQL = """
DROP TABLE IF EXISTS _MemorySummaryTaskMigrationMap;
CREATE TEMP TABLE _MemorySummaryTaskMigrationMap (
    packet_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL UNIQUE
);
INSERT OR IGNORE INTO _MemorySummaryTaskMigrationMap (packet_id, task_id)
SELECT
    packet_id,
    CASE
        WHEN packet_type='summary_refresh_input' AND packet_id LIKE 'summary-refresh:summary:%'
            THEN substr(packet_id, 17)
        ELSE packet_id
    END
FROM MemorySummaryInputs
WHERE source_kind='storyline'
  AND status='active'
  AND NOT EXISTS (
      SELECT 1
      FROM MemoryStorylineSummaryTasks current_task
      WHERE current_task.task_id = CASE
          WHEN MemorySummaryInputs.packet_type='summary_refresh_input'
               AND MemorySummaryInputs.packet_id LIKE 'summary-refresh:summary:%'
              THEN substr(MemorySummaryInputs.packet_id, 17)
          ELSE MemorySummaryInputs.packet_id
      END
  )
ORDER BY priority DESC, updated_at_ms DESC, packet_id;

INSERT OR IGNORE INTO MemoryStorylineSummaryTasks (
    task_id, task_type, storyline_id, storyline_revision, input_hash, priority,
    confidence_tier, status, retry_count, last_error, created_at_ms, updated_at_ms
)
SELECT
    migration.task_id,
    'refresh',
    legacy.source_id,
    legacy.source_revision,
    legacy.input_hash,
    legacy.priority,
    legacy.confidence_tier,
    'active',
    0,
    '',
    legacy.created_at_ms,
    legacy.updated_at_ms
FROM MemorySummaryInputs legacy
JOIN _MemorySummaryTaskMigrationMap migration
  ON migration.packet_id=legacy.packet_id;

INSERT OR IGNORE INTO MemoryStorylineSummaryTaskEvents (task_id, event_id, rank, role, status)
SELECT migration.task_id, legacy.event_id, legacy.rank, legacy.role, legacy.status
FROM MemorySummaryInputEvents legacy
JOIN _MemorySummaryTaskMigrationMap migration
  ON migration.packet_id=legacy.packet_id;

INSERT OR IGNORE INTO MemoryStorylineSummaryTaskRelations (
    task_id, relation_id, source_event_id, target_event_id,
    relation_type, status, confidence
)
SELECT
    migration.task_id,
    legacy.relation_id,
    legacy.source_event_id,
    legacy.target_event_id,
    legacy.relation_type,
    legacy.status,
    0.0
FROM MemorySummaryInputRelations legacy
JOIN _MemorySummaryTaskMigrationMap migration
  ON migration.packet_id=legacy.packet_id;

DROP TABLE MemorySummaryInputRelations;
DROP TABLE MemorySummaryInputEvents;
DROP TABLE MemorySummaryInputs;
DROP TABLE _MemorySummaryTaskMigrationMap;
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
class StorylineSummary:
    storyline_id: str
    scope: str
    scheme_name: str
    profile: str
    anchor_key: str
    member_count: int
    score: float
    event_ids: tuple[int, ...]
    evidence_json: str


@dataclass(frozen=True)
class StorylineMember:
    storyline_id: str
    event_id: int
    score: float
    rank: int
    evidence_json: str


@dataclass(frozen=True)
class StorylineSummaryRecord:
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
    source_event_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CandidateStoryline:
    candidate_storyline_id: str
    event_ids: tuple[int, ...]
    status: str = "pending"


@dataclass(frozen=True)
class PreprocessReport:
    events: int
    canonical_entities: int
    entity_mentions: int
    event_relations: int
    algorithmic_storyline_enabled: bool
    algorithmic_storyline_ids: tuple[str, ...] = ()
    algorithmic_storyline_members: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "canonical_entities": self.canonical_entities,
            "entity_mentions": self.entity_mentions,
            "event_relations": self.event_relations,
            "algorithmic_storyline_enabled": self.algorithmic_storyline_enabled,
            "storylines": len(self.algorithmic_storyline_ids),
            "storyline_members": self.algorithmic_storyline_members,
            "algorithmic_storyline_ids": list(self.algorithmic_storyline_ids),
        }


@dataclass(frozen=True)
class CandidateStorylineConsolidationReport:
    pending_candidate_storylines_loaded: int = 0
    dry_run: bool = True
    solidify: bool = False
    candidate_storylines_written: int = 0
    candidate_storyline_members_written: int = 0
    candidate_storyline_status_rows_updated: int = 0
    storyline_ids_written: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending_candidate_storylines_loaded": self.pending_candidate_storylines_loaded,
            "dry_run": self.dry_run,
            "solidify": self.solidify,
            "candidate_storylines_written": self.candidate_storylines_written,
            "candidate_storyline_members_written": self.candidate_storyline_members_written,
            "candidate_storyline_status_rows_updated": self.candidate_storyline_status_rows_updated,
            "storyline_ids_written": list(self.storyline_ids_written),
        }


def ensure_preprocessing_schema(con: sqlite3.Connection) -> None:
    _migrate_storyline_schema(con)
    _migrate_legacy_summary_cache(con)
    con.executescript(PREPROCESSING_SCHEMA_SQL)
    _finish_storyline_schema_migration(con)
    _migrate_legacy_summary_queue(con)


async def ensure_preprocessing_schema_async(db: Any) -> None:
    await _migrate_storyline_schema_async(db)
    await _migrate_legacy_summary_cache_async(db)
    await db.executescript(PREPROCESSING_SCHEMA_SQL)
    await _finish_storyline_schema_migration_async(db)
    await _migrate_legacy_summary_queue_async(db)


def _migrate_storyline_schema(con: sqlite3.Connection) -> None:
    tables = _table_names(con)
    for old_name, new_name in _STORYLINE_TABLE_RENAMES:
        if old_name in tables and new_name not in tables:
            con.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
            tables.remove(old_name)
            tables.add(new_name)
    for table, renames in _STORYLINE_COLUMN_RENAMES.items():
        columns = _table_columns(con, table)
        for old_name, new_name in renames:
            if old_name in columns and new_name not in columns:
                con.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
                columns.remove(old_name)
                columns.add(new_name)
    for index_name in _STALE_STORYLINE_INDEXES:
        con.execute(f"DROP INDEX IF EXISTS {index_name}")


async def _migrate_storyline_schema_async(db: Any) -> None:
    tables = await _table_names_async(db)
    for old_name, new_name in _STORYLINE_TABLE_RENAMES:
        if old_name in tables and new_name not in tables:
            await db.execute(f"ALTER TABLE {old_name} RENAME TO {new_name}")
            tables.remove(old_name)
            tables.add(new_name)
    for table, renames in _STORYLINE_COLUMN_RENAMES.items():
        columns = await _table_columns_async(db, table)
        for old_name, new_name in renames:
            if old_name in columns and new_name not in columns:
                await db.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
                columns.remove(old_name)
                columns.add(new_name)
    for index_name in _STALE_STORYLINE_INDEXES:
        await db.execute(f"DROP INDEX IF EXISTS {index_name}")


def _finish_storyline_schema_migration(con: sqlite3.Connection) -> None:
    con.executescript(_STORYLINE_DATA_MIGRATION_SQL)


async def _finish_storyline_schema_migration_async(db: Any) -> None:
    await db.executescript(_STORYLINE_DATA_MIGRATION_SQL)


def _migrate_legacy_summary_cache(con: sqlite3.Connection) -> None:
    columns = _table_columns(con, "MemorySummaryCache")
    if not columns:
        return
    con.execute("DROP INDEX IF EXISTS idx_MemorySummaryCache_packet")
    if "task_id" not in columns and "packet_id" in columns:
        con.execute("ALTER TABLE MemorySummaryCache RENAME COLUMN packet_id TO task_id")


async def _migrate_legacy_summary_cache_async(db: Any) -> None:
    columns = await _table_columns_async(db, "MemorySummaryCache")
    if not columns:
        return
    await db.execute("DROP INDEX IF EXISTS idx_MemorySummaryCache_packet")
    if "task_id" not in columns and "packet_id" in columns:
        await db.execute("ALTER TABLE MemorySummaryCache RENAME COLUMN packet_id TO task_id")


def _migrate_legacy_summary_queue(con: sqlite3.Connection) -> None:
    tables = _table_names(con)
    legacy_tables = set(_LEGACY_SUMMARY_QUEUE_TABLES)
    if not tables.intersection(legacy_tables):
        return
    if "MemorySummaryInputs" in tables:
        con.execute(
            """
            UPDATE MemorySummaryInputs
            SET source_kind='storyline',
                source_id=replace(source_id, 'cluster:', 'storyline:'),
                packet_id=replace(packet_id, 'summary:cluster:', 'summary:storyline:')
            WHERE source_kind='cluster' OR packet_id LIKE '%summary:cluster:%'
            """
        )
    if legacy_tables <= tables and all(
        required <= _table_columns(con, table)
        for table, required in _LEGACY_SUMMARY_QUEUE_COLUMNS.items()
    ):
        con.executescript(_LEGACY_SUMMARY_QUEUE_MIGRATION_SQL)
        return
    _drop_legacy_summary_queue(con)


async def _migrate_legacy_summary_queue_async(db: Any) -> None:
    tables = await _table_names_async(db)
    legacy_tables = set(_LEGACY_SUMMARY_QUEUE_TABLES)
    if not tables.intersection(legacy_tables):
        return
    if "MemorySummaryInputs" in tables:
        await db.execute(
            """
            UPDATE MemorySummaryInputs
            SET source_kind='storyline',
                source_id=replace(source_id, 'cluster:', 'storyline:'),
                packet_id=replace(packet_id, 'summary:cluster:', 'summary:storyline:')
            WHERE source_kind='cluster' OR packet_id LIKE '%summary:cluster:%'
            """
        )
    columns_are_compatible = legacy_tables <= tables
    if columns_are_compatible:
        for table, required in _LEGACY_SUMMARY_QUEUE_COLUMNS.items():
            if not required <= await _table_columns_async(db, table):
                columns_are_compatible = False
                break
    if columns_are_compatible:
        await db.executescript(_LEGACY_SUMMARY_QUEUE_MIGRATION_SQL)
        return
    await _drop_legacy_summary_queue_async(db)


def _drop_legacy_summary_queue(con: sqlite3.Connection) -> None:
    for table in _LEGACY_SUMMARY_QUEUE_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {table}")


async def _drop_legacy_summary_queue_async(db: Any) -> None:
    for table in _LEGACY_SUMMARY_QUEUE_TABLES:
        await db.execute(f"DROP TABLE IF EXISTS {table}")


def _table_names(con: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


async def _table_names_async(db: Any) -> set[str]:
    async with db.execute("SELECT name FROM sqlite_master WHERE type='table'") as cur:
        return {str(row[0]) for row in await cur.fetchall()}


async def _table_columns_async(db: Any, table: str) -> set[str]:
    async with db.execute(f"PRAGMA table_info({table})") as cur:
        return {str(row[1]) for row in await cur.fetchall()}


def load_memory_dataset(
    con: sqlite3.Connection,
    *,
    limit: int = 2000,
) -> tuple[dict[int, EventRecord], dict[int, list[RoleRecord]]]:
    con.row_factory = sqlite3.Row
    _require_tables(con, {"MemoryEvents", "MemoryParticipants"})
    rows = list(
        con.execute(
            """
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
    algorithmic_storyline_enabled: bool = False,
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
                    "algorithmic_storyline_enabled": bool(algorithmic_storyline_enabled),
                }
            ),
        ),
    )
    run_id = int(cur.lastrowid or 0)
    entity_result = build_entity_resolution(events, roles)
    write_entity_resolution(con, entity_result, now_ms=started)
    working_roles = canonicalize_roles(roles) if canonical_entities else roles
    storylines: list[StorylineSummary] = []
    storyline_members: list[StorylineMember] = []
    if algorithmic_storyline_enabled:
        storylines, storyline_members = materialize_algorithmic_storylines(events, working_roles)
        if storylines:
            storyline_run_id = create_storyline_run(
                con,
                profile="algorithmic",
                trigger=trigger,
                event_ids=(event_id for storyline in storylines for event_id in storyline.event_ids),
                now_ms=started,
                params={"preprocess_run_id": run_id},
            )
            write_storyline_cache(con, storylines, storyline_members, run_id=storyline_run_id, now_ms=started)
    report = PreprocessReport(
        events=len(events),
        canonical_entities=len(entity_result.entities),
        entity_mentions=len(entity_result.mentions),
        event_relations=0,
        algorithmic_storyline_enabled=bool(algorithmic_storyline_enabled),
        algorithmic_storyline_ids=tuple(storyline.storyline_id for storyline in storylines),
        algorithmic_storyline_members=len(storyline_members),
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
def materialize_algorithmic_storylines(
    events: dict[int, EventRecord],
    roles: dict[int, list[RoleRecord]],
) -> tuple[list[StorylineSummary], list[StorylineMember]]:
    summaries: list[StorylineSummary] = []
    members: list[StorylineMember] = []
    for storyline_events, scope, scheme, profile, anchor_key, score in _algorithmic_storyline_groups(events, roles):
        event_ids = tuple(sorted(storyline_events))
        if len(event_ids) < 2:
            continue
        storyline_id = f"{scope}:{_sha1('storyline', scheme, anchor_key, str(min(event_ids)))[:16]}"
        evidence = {
            "generator": "algorithmic_storyline",
            "anchor_key": anchor_key,
            "scheme_name": scheme,
            "profile": profile,
            "event_ids": list(event_ids),
        }
        summaries.append(
            StorylineSummary(
                storyline_id,
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
            members.append(StorylineMember(storyline_id, event_id, score, rank, _json(evidence)))
    return summaries, members


def write_storyline_cache(
    con: sqlite3.Connection,
    summaries: list[StorylineSummary],
    members: list[StorylineMember],
    *,
    run_id: int,
    now_ms: int,
) -> None:
    members_by_storyline: dict[str, set[int]] = defaultdict(set)
    for member in members:
        members_by_storyline[member.storyline_id].add(member.event_id)
    for summary in summaries:
        current_event_ids = sorted(members_by_storyline.get(summary.storyline_id, set()))
        if current_event_ids:
            placeholders = ",".join("?" * len(current_event_ids))
            con.execute(
                f"""
                UPDATE MemoryStorylineMembers
                SET status='inactive', last_seen_at=?, last_seen_run_id=?
                WHERE storyline_id=? AND status='active'
                  AND event_id NOT IN ({placeholders})
                """,
                [now_ms, run_id, summary.storyline_id, *current_event_ids],
            )
    con.executemany(
        """
        INSERT INTO MemoryStorylines (
            storyline_id, scope, scheme_name, anchor_key, profile, status, created_at, updated_at,
            first_seen_run_id, last_seen_run_id, revision, member_count, score, signature_json
        ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, 1, ?, ?, ?)
        ON CONFLICT(storyline_id) DO UPDATE SET
            member_count=excluded.member_count,
            score=excluded.score,
            status='active',
            last_seen_run_id=excluded.last_seen_run_id,
            revision=CASE WHEN MemoryStorylines.member_count != excluded.member_count OR MemoryStorylines.signature_json != excluded.signature_json THEN MemoryStorylines.revision + 1 ELSE MemoryStorylines.revision END,
            updated_at=excluded.updated_at,
            signature_json=excluded.signature_json
        """,
        [(item.storyline_id, item.scope, item.scheme_name, item.anchor_key, item.profile, now_ms, now_ms, run_id, run_id, item.member_count, item.score, item.evidence_json) for item in summaries],
    )
    con.executemany(
        """
        INSERT INTO MemoryStorylineMembers (
            storyline_id, event_id, score, rank, status, revision, corrected_by_event_id,
            first_seen_at, last_seen_at, first_seen_run_id, last_seen_run_id, evidence_json
        ) VALUES (?, ?, ?, ?, 'active', 1, 0, ?, ?, ?, ?, ?)
        ON CONFLICT(storyline_id, event_id) DO UPDATE SET
            score=excluded.score,
            rank=excluded.rank,
            status='active',
            revision=CASE
                WHEN MemoryStorylineMembers.score != excluded.score
                  OR MemoryStorylineMembers.rank != excluded.rank
                  OR MemoryStorylineMembers.status != 'active'
                THEN MemoryStorylineMembers.revision + 1
                ELSE MemoryStorylineMembers.revision
            END,
            last_seen_at=excluded.last_seen_at,
            last_seen_run_id=excluded.last_seen_run_id,
            evidence_json=excluded.evidence_json
        """,
        [(item.storyline_id, item.event_id, item.score, item.rank, now_ms, now_ms, run_id, run_id, item.evidence_json) for item in members],
    )


def create_storyline_run(
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
        INSERT INTO MemoryStorylineRuns (
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
    return int(cur.lastrowid or 0)


def storyline_summary_to_json(card: StorylineSummaryRecord) -> str:
    return _json(asdict(card))


def storyline_summary_from_json(payload: str | dict[str, Any]) -> StorylineSummaryRecord:
    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    return StorylineSummaryRecord(
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
        source_event_ids=tuple(int(x) for x in data.get("source_event_ids") or () if str(x).lstrip("-").isdigit()),
    )


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


def _algorithmic_storyline_groups(events: dict[int, EventRecord], roles: dict[int, list[RoleRecord]]):
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


def load_pending_candidate_storylines(
    con: sqlite3.Connection,
    *,
    max_candidate_storylines: int,
) -> list[CandidateStoryline]:
    previous_row_factory = con.row_factory
    try:
        con.row_factory = sqlite3.Row
        rows = list(
            con.execute(
                """
                SELECT candidate_storyline_id, event_ids_json, status
                FROM MemoryCandidateStorylines
                WHERE status='pending'
                ORDER BY created_at_ms ASC, candidate_storyline_id ASC
                LIMIT ?
                """,
                (max(1, int(max_candidate_storylines)),),
            )
        )
    finally:
        con.row_factory = previous_row_factory
    return [
        CandidateStoryline(
            candidate_storyline_id=str(row["candidate_storyline_id"]),
            event_ids=_json_int_tuple(str(row["event_ids_json"] or "[]")),
            status=str(row["status"] or "pending"),
        )
        for row in rows
    ]


def run_candidate_storyline_consolidation(
    con: sqlite3.Connection,
    *,
    max_candidate_storylines: int = 100,
    dry_run: bool = True,
    solidify: bool = False,
) -> CandidateStorylineConsolidationReport:
    ensure_preprocessing_schema(con)
    candidates = load_pending_candidate_storylines(
        con,
        max_candidate_storylines=max_candidate_storylines,
    )
    should_write = bool(solidify) and not bool(dry_run)
    if not should_write:
        return CandidateStorylineConsolidationReport(
            pending_candidate_storylines_loaded=len(candidates),
            dry_run=True,
            solidify=bool(solidify),
        )

    valid_events = _existing_event_ids(
        con,
        {event_id for candidate in candidates for event_id in candidate.event_ids},
    )
    storylines: list[StorylineSummary] = []
    members: list[StorylineMember] = []
    accepted: list[CandidateStoryline] = []
    rejected: list[CandidateStoryline] = []
    for candidate in candidates:
        event_ids = tuple(sorted({event_id for event_id in candidate.event_ids if event_id in valid_events}))
        if len(event_ids) < 2:
            rejected.append(candidate)
            continue
        storyline_hash = _sha1("candidate-storyline", *(str(event_id) for event_id in event_ids))
        storyline_id = f"candidate_storyline:{storyline_hash[:16]}"
        evidence = {
            "generator": "sleep_candidate_storyline_consolidation",
            "candidate_storyline_id": candidate.candidate_storyline_id,
            "event_ids": list(event_ids),
        }
        storylines.append(
            StorylineSummary(
                storyline_id=storyline_id,
                scope="candidate_storyline",
                scheme_name="llm_candidate_storyline",
                profile="sleep-consolidated",
                anchor_key=f"candidate_storyline:{storyline_hash[:20]}",
                member_count=len(event_ids),
                score=1.0,
                event_ids=event_ids,
                evidence_json=_json(evidence),
            )
        )
        for rank, event_id in enumerate(event_ids, start=1):
            members.append(
                StorylineMember(
                    storyline_id=storyline_id,
                    event_id=event_id,
                    score=1.0,
                    rank=rank,
                    evidence_json=_json(evidence),
                )
            )
        accepted.append(candidate)

    now_ms = _now_ms()
    if storylines:
        run_id = create_storyline_run(
            con,
            profile="sleep-consolidated",
            trigger="candidate_storyline",
            event_ids=(event_id for storyline in storylines for event_id in storyline.event_ids),
            now_ms=now_ms,
            params={"generator": "sleep_candidate_storyline_consolidation"},
        )
        write_storyline_cache(con, storylines, members, run_id=run_id, now_ms=now_ms)

    status_rows = 0
    for status, items in (("accepted", accepted), ("rejected", rejected)):
        if not items:
            continue
        placeholders = ",".join("?" * len(items))
        con.execute(
            f"""
            UPDATE MemoryCandidateStorylines
            SET status=?, updated_at_ms=?
            WHERE candidate_storyline_id IN ({placeholders})
            """,
            [status, now_ms, *(item.candidate_storyline_id for item in items)],
        )
        status_rows += len(items)
    return CandidateStorylineConsolidationReport(
        pending_candidate_storylines_loaded=len(candidates),
        dry_run=False,
        solidify=True,
        candidate_storylines_written=len(storylines),
        candidate_storyline_members_written=len(members),
        candidate_storyline_status_rows_updated=status_rows,
        storyline_ids_written=tuple(storyline.storyline_id for storyline in storylines),
    )


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
    "StorylineSummaryRecord",
    "CandidateStoryline",
    "CandidateStorylineConsolidationReport",
    "EventRecord",
    "PREPROCESSING_SCHEMA_SQL",
    "PreprocessReport",
    "RoleRecord",
    "build_entity_resolution",
    "canonicalize_roles",
    "storyline_summary_from_json",
    "storyline_summary_to_json",
    "ensure_preprocessing_schema",
    "ensure_preprocessing_schema_async",
    "load_memory_dataset",
    "load_pending_candidate_storylines",
    "materialize_algorithmic_storylines",
    "run_candidate_storyline_consolidation",
    "run_preprocessing",
]
