from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest


def _fresh_db(tmp_path: Path, name: str) -> str:
    import database
    from memory.repo import events

    database.DB_PATH = str(tmp_path / f"{name}.sqlite3")
    events._SCHEMA_READY = False
    return database.DB_PATH


def _connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    return con


def _insert_storyline_summary_task(
    con: sqlite3.Connection,
    *,
    task_id: str,
    storyline_id: str,
    task_type: str = "refresh",
    storyline_revision: int = 1,
    input_hash: str = "hash",
    priority: int = 30,
    confidence_tier: str = "medium",
    status: str = "active",
    event_ids: tuple[int, ...] = (),
    relation_rows: tuple[tuple[str, int, int, str, str, float], ...] = (),
    now_ms: int = 1,
) -> None:
    del relation_rows
    con.execute(
        """
        INSERT INTO MemoryStorylineSummaryTasks (
            task_id, task_type, storyline_id, storyline_revision, input_hash,
            priority, confidence_tier, status, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, task_type, storyline_id, storyline_revision, input_hash, priority, confidence_tier, status, now_ms, now_ms),
    )
    con.executemany(
        """
        INSERT INTO MemoryStorylineSummaryTaskEvents (task_id, event_id, rank, role, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        [(task_id, event_id, index, "storyline_member") for index, event_id in enumerate(event_ids, start=1)],
    )


def _create_legacy_summary_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE MemorySummaryInputs (
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
        CREATE TABLE MemorySummaryInputEvents (
            packet_id TEXT NOT NULL,
            event_id INTEGER NOT NULL,
            rank INTEGER NOT NULL DEFAULT 0,
            role TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            PRIMARY KEY (packet_id, event_id)
        );
        CREATE TABLE MemorySummaryInputRelations (
            packet_id TEXT NOT NULL,
            relation_id TEXT NOT NULL,
            source_event_id INTEGER NOT NULL DEFAULT 0,
            target_event_id INTEGER NOT NULL DEFAULT 0,
            relation_type TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            PRIMARY KEY (packet_id, relation_id)
        );
        CREATE TABLE MemorySummaryCache (
            summary_id TEXT PRIMARY KEY,
            packet_id TEXT NOT NULL,
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
        CREATE INDEX idx_MemorySummaryInputs_source
        ON MemorySummaryInputs(source_kind, source_id, status);
        CREATE INDEX idx_MemorySummaryInputs_queue
        ON MemorySummaryInputs(status, priority DESC, updated_at_ms);
        CREATE INDEX idx_MemorySummaryInputEvents_event
        ON MemorySummaryInputEvents(event_id, status);
        CREATE INDEX idx_MemorySummaryInputRelations_relation
        ON MemorySummaryInputRelations(relation_id, status);
        CREATE INDEX idx_MemorySummaryCache_packet
        ON MemorySummaryCache(packet_id, input_hash, status);
        """
    )


def test_preprocessing_schema_has_no_legacy_summary_input_tables(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema

    db_path = tmp_path / "current-summary-schema.sqlite3"
    with sqlite3.connect(db_path) as con:
        ensure_preprocessing_schema(con)

        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols = {row[1] for row in con.execute("PRAGMA table_info(MemorySummaryCache)")}
        assert cols == {
            "summary_id", "task_id", "input_hash", "model", "status",
            "summary", "created_at_ms", "updated_at_ms", "error_json",
        }
        assert "MemorySummaryInputs" not in tables
        assert "MemorySummaryInputEvents" not in tables
        assert "MemorySummaryInputRelations" not in tables


def test_preprocessing_schema_rebuilds_current_summary_cache_and_uses_json_fallback(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema

    db_path = tmp_path / "old-current-summary-cache.sqlite3"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE MemorySummaryCache (
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
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, short_summary,
                storyline_summary_json, created_at_ms, updated_at_ms
            ) VALUES (
                'summary:storyline:json-fallback', 'summary:storyline:json-fallback',
                'hash', 'old-model', 'ready', '',
                '{"short_summary":"从旧 JSON 恢复的故事线。"}', 10, 20
            );
            """
        )

        ensure_preprocessing_schema(con)

        columns = {row[1] for row in con.execute("PRAGMA table_info(MemorySummaryCache)")}
        row = con.execute(
            "SELECT input_hash, model, status, summary, created_at_ms, updated_at_ms FROM MemorySummaryCache"
        ).fetchone()

        assert columns == {
            "summary_id", "task_id", "input_hash", "model", "status",
            "summary", "created_at_ms", "updated_at_ms", "error_json",
        }
        assert row == ("hash", "old-model", "ready", "从旧 JSON 恢复的故事线。", 10, 20)


def test_preprocessing_schema_migrates_cluster_tables_to_storylines(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema

    db_path = tmp_path / "legacy-cluster-schema.sqlite3"
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE MemoryClusters (
                cluster_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                scheme_name TEXT NOT NULL DEFAULT '', anchor_key TEXT NOT NULL DEFAULT '',
                profile TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'active',
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
                first_seen_run_id INTEGER NOT NULL DEFAULT 0,
                last_seen_run_id INTEGER NOT NULL DEFAULT 0,
                revision INTEGER NOT NULL DEFAULT 1,
                member_count INTEGER NOT NULL DEFAULT 0,
                score REAL NOT NULL DEFAULT 0.0,
                signature_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE MemoryClusterMembers (
                cluster_id TEXT NOT NULL, event_id INTEGER NOT NULL, score REAL NOT NULL,
                rank INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'active',
                revision INTEGER NOT NULL DEFAULT 1, corrected_by_event_id INTEGER NOT NULL DEFAULT 0,
                first_seen_at INTEGER NOT NULL, last_seen_at INTEGER NOT NULL,
                first_seen_run_id INTEGER NOT NULL DEFAULT 0,
                last_seen_run_id INTEGER NOT NULL DEFAULT 0,
                evidence_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (cluster_id, event_id)
            );
            CREATE TABLE MemoryEpisodeCandidates (
                candidate_id TEXT PRIMARY KEY,
                event_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at_ms INTEGER NOT NULL DEFAULT 0,
                updated_at_ms INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO MemoryClusters (
                cluster_id, scope, scheme_name, anchor_key, profile,
                created_at, updated_at, member_count, score
            ) VALUES (
                'episode:legacy', 'episode', 'llm_episode_candidate',
                'episode:legacy-anchor', 'sleep-consolidated', 1, 1, 2, 1.0
            );
            INSERT INTO MemoryClusterMembers (
                cluster_id, event_id, score, rank, first_seen_at, last_seen_at
            ) VALUES ('episode:legacy', 41, 1.0, 1, 1, 1);
            INSERT INTO MemoryEpisodeCandidates (
                candidate_id, event_ids_json, status, created_at_ms, updated_at_ms
            ) VALUES ('candidate:legacy', '[41,42]', 'pending', 1, 1);
            """
        )

        ensure_preprocessing_schema(con)
        ensure_preprocessing_schema(con)

        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        storyline = con.execute(
            """
            SELECT storyline_id, scope, scheme_name, anchor_key
            FROM MemoryStorylines
            """
        ).fetchone()
        member = con.execute(
            "SELECT storyline_id, event_id FROM MemoryStorylineMembers"
        ).fetchone()
        candidate = con.execute(
            "SELECT candidate_storyline_id, event_ids_json FROM MemoryCandidateStorylines"
        ).fetchone()

        assert storyline == (
            "candidate_storyline:legacy",
            "candidate_storyline",
            "llm_candidate_storyline",
            "candidate_storyline:legacy-anchor",
        )
        assert member == ("candidate_storyline:legacy", 41)
        assert candidate == ("candidate_storyline:legacy", "[41,42]")
        assert not tables.intersection(
            {
                "MemoryClusters",
                "MemoryClusterMembers",
                "MemoryEpisodeCandidates",
                "MemoryEpisodes",
                "MemoryEpisodeMembers",
            }
        )


def test_preprocessing_schema_migrates_legacy_summary_cache_and_active_storyline_queue(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema

    db_path = tmp_path / "legacy-summary-schema.sqlite3"
    with sqlite3.connect(db_path) as con:
        _create_legacy_summary_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, packet_id, input_hash, model, status, title,
                short_summary, storyline_summary_json, created_at_ms, updated_at_ms
            ) VALUES (
                'summary:storyline:legacy', 'summary:storyline:legacy', 'ready-hash',
                'legacy-model', 'ready', '保留标题', '保留摘要',
                '{"summary_id":"summary:storyline:legacy"}', 1, 2
            )
            """
        )
        con.execute(
            """
            INSERT INTO MemorySummaryInputs (
                packet_id, packet_type, source_kind, source_id, source_revision,
                input_hash, priority, confidence_tier, status, created_at_ms,
                updated_at_ms, packet_json
            ) VALUES (
                'summary-refresh:summary:storyline:legacy', 'summary_refresh_input',
                'storyline', 'storyline:legacy', 3, 'task-hash', 90, 'high',
                'active', 3, 4, '{}'
            )
            """
        )
        con.execute(
            """
            INSERT INTO MemorySummaryInputEvents (packet_id, event_id, rank, role, status)
            VALUES ('summary-refresh:summary:storyline:legacy', 42, 1, 'delta', 'active')
            """
        )
        con.execute(
            """
            INSERT INTO MemorySummaryInputRelations (
                packet_id, relation_id, source_event_id, target_event_id,
                relation_type, status
            ) VALUES (
                'summary-refresh:summary:storyline:legacy', 'relation:legacy',
                41, 42, 'updates_state', 'active'
            )
            """
        )

        ensure_preprocessing_schema(con)
        ensure_preprocessing_schema(con)

        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        columns = {row[1] for row in con.execute("PRAGMA table_info(MemorySummaryCache)")}
        cache_row = con.execute(
            """
            SELECT summary_id, task_id, input_hash, model, status, summary
            FROM MemorySummaryCache
            """
        ).fetchone()
        task_row = con.execute(
            """
            SELECT task_id, task_type, storyline_id, storyline_revision, input_hash,
                   priority, confidence_tier, status
            FROM MemoryStorylineSummaryTasks
            """
        ).fetchone()
        event_row = con.execute(
            "SELECT task_id, event_id, rank, role, status FROM MemoryStorylineSummaryTaskEvents"
        ).fetchone()
        assert columns == {
            "summary_id", "task_id", "input_hash", "model", "status",
            "summary", "created_at_ms", "updated_at_ms", "error_json",
        }
        assert cache_row == (
            "summary:storyline:legacy",
            "summary:storyline:legacy",
            "ready-hash",
            "legacy-model",
            "ready",
            "保留摘要",
        )
        assert task_row == (
            "summary:storyline:legacy",
            "refresh",
            "storyline:legacy",
            3,
            "task-hash",
            90,
            "high",
            "active",
        )
        assert event_row == ("summary:storyline:legacy", 42, 1, "delta", "active")
        assert "MemoryStorylineSummaryTaskRelations" not in tables
        assert not tables.intersection(
            {"MemorySummaryInputs", "MemorySummaryInputEvents", "MemorySummaryInputRelations"}
        )
        assert "idx_MemorySummaryCache_packet" not in indexes
        assert "idx_MemorySummaryCache_task" in indexes


def test_preprocessing_schema_async_migrates_legacy_summary_cache(tmp_path):
    import aiosqlite

    from memory.sleep.consolidation import ensure_preprocessing_schema_async

    db_path = tmp_path / "legacy-summary-schema-async.sqlite3"
    with sqlite3.connect(db_path) as con:
        _create_legacy_summary_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, packet_id, input_hash, model, status, title, short_summary
            ) VALUES (
                'summary:storyline:async', 'summary:storyline:async', 'async-hash',
                'legacy-model', 'ready', '异步标题', '异步摘要'
            )
            """
        )
        con.commit()

    async def scenario() -> None:
        async with aiosqlite.connect(db_path) as db:
            await ensure_preprocessing_schema_async(db)
            await ensure_preprocessing_schema_async(db)
            await db.commit()

    asyncio.run(scenario())

    with sqlite3.connect(db_path) as con:
        columns = {row[1] for row in con.execute("PRAGMA table_info(MemorySummaryCache)")}
        row = con.execute(
            "SELECT summary_id, task_id, status, summary FROM MemorySummaryCache"
        ).fetchone()
        tables = {item[0] for item in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert columns == {
        "summary_id", "task_id", "input_hash", "model", "status",
        "summary", "created_at_ms", "updated_at_ms", "error_json",
    }
    assert row == (
        "summary:storyline:async",
        "summary:storyline:async",
        "ready",
        "异步摘要",
    )
    assert not tables.intersection(
        {"MemorySummaryInputs", "MemorySummaryInputEvents", "MemorySummaryInputRelations"}
    )


def test_summary_worker_keeps_current_task_schema_without_legacy_migration(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    db_path = tmp_path / "current-summary-task-queue.sqlite3"
    with sqlite3.connect(db_path) as con:
        ensure_preprocessing_schema(con)

        stats = process_active_summary_inputs(con, max_inputs=1, now_ms=1).to_dict()

        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert stats["summary_tasks_loaded"] == 0
        assert "MemoryStorylineSummaryTasks" in tables
        assert "MemoryStorylineSummaryTaskEvents" in tables
        assert "MemoryStorylineSummaryTaskRelations" not in tables
        assert "MemorySummaryCache" in tables
        assert "MemoryV2SummaryCache" not in tables


def test_entity_resolution_preserves_explicit_types_and_only_normalizes_unicode():
    from memory.sleep.consolidation import EventRecord, RoleRecord, build_entity_resolution

    events = {
        1: EventRecord(1, "AICQ 测试群出现讨论。", "", "say", "actual", 0.9, 1, "group", "1", 1),
        2: EventRecord(2, "Liklu Loteji 回复 Liklu。", "", "reply", "actual", 0.9, 2, "group", "1", 1),
    }
    roles = {
        1: [
            RoleRecord(1, "location", "Group:AICQ测试群"),
            RoleRecord(1, "platform", "Platform:AICQ 测试群"),
        ],
        2: [
            RoleRecord(2, "agent", "Person:Liklu"),
            RoleRecord(2, "recipient", "Person:Liklu Loteji"),
        ],
    }

    result = build_entity_resolution(events, roles, include_sessions=False)
    aliases_by_raw = {item.raw_entity: item.entity_id for item in result.aliases}

    assert aliases_by_raw["Group:AICQ测试群"] != aliases_by_raw["Platform:AICQ 测试群"]
    assert aliases_by_raw["Person:Liklu"] != aliases_by_raw["Person:Liklu Loteji"]


def test_preprocessing_does_not_infer_relations_from_event_text(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-no-text-relation-inference")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        await write_event(
            event_type="ask",
            summary="A 询问《以撒的结合》角色解锁方法。",
            conv_type="group",
            conv_id="1",
            roles=[{"role": "theme", "entity": "Work:以撒的结合"}],
            occurred_at=1_000,
        )
        await write_event(
            event_type="answer",
            summary="B 回答了角色解锁方法。",
            conv_type="group",
            conv_id="1",
            roles=[{"role": "theme", "entity": "Work:以撒的结合"}],
            occurred_at=2_000,
        )

    asyncio.run(scenario())
    from memory.sleep.consolidation import run_preprocessing

    with _connect(db_path) as con:
        report = run_preprocessing(con, trigger="test.no-text-inference")
        assert report.event_relations == 0
        assert con.execute("SELECT COUNT(*) FROM MemoryEventRelations").fetchone()[0] == 0


def test_summary_worker_consumes_refresh_task_and_writes_ready_summary(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-refresh")
    import app_state

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        old_id = await write_event(
            event_type="start",
            summary="小白开始玩《以撒的结合》。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=1_000,
        )
        new_id = await write_event(
            event_type="complete",
            summary="小白白金《以撒的结合》。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=2_000,
        )
        return old_id, new_id

    old_id, new_id = asyncio.run(scenario())

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    class FakeSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            assert user_content == """<task>
<previous_storyline>
  我正在推进《以撒的结合》。
</previous_storyline>

<events>
  <event occurred_at="1970-01-01T00:00:01+00:00" confidence="0.50">
    小白开始玩《以撒的结合》。
  </event>
  <event occurred_at="1970-01-01T00:00:02+00:00" confidence="0.50">
    小白白金《以撒的结合》。
  </event>
</events>
</task>"""
            return """<analysis>新事件完成了这条故事线。</analysis>
<storyline>我从开始推进《以撒的结合》更新为已经白金。</storyline>"""

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeSummaryAdapter())

    task_id = "summary:storyline:isaac"

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, summary,
                created_at_ms, updated_at_ms
            ) VALUES ('old-summary-row', ?, 'old', 'test', 'stale', ?, 1, 1)
            """,
            (task_id, "我正在推进《以撒的结合》。"),
        )
        _insert_storyline_summary_task(
            con,
            task_id=task_id,
            task_type="refresh",
            storyline_id="storyline:isaac",
            storyline_revision=2,
            input_hash="hash-refresh",
            priority=90,
            confidence_tier="high",
            event_ids=(old_id, new_id),
            relation_rows=(("rel-new", new_id, old_id, "updates_state", "active", 0.9),),
            now_ms=2,
        )

        stats = process_active_summary_inputs(con, now_ms=3).to_dict()
        con.commit()

        row = con.execute(
            """
            SELECT status, summary
            FROM MemorySummaryCache
            WHERE summary_id=?
            """,
            (task_id,),
        ).fetchone()

        assert stats["summaries_ready"] == 1
        assert con.execute("SELECT status FROM MemoryStorylineSummaryTasks WHERE task_id=?", (task_id,)).fetchone()[0] == "done"
        assert row[0] == "ready"
        assert row[1] == "我从开始推进《以撒的结合》更新为已经白金。"


def test_summary_worker_uses_memory_consolidation_llm_for_storyline_summary(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-llm")
    import app_state

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        first_id = await write_event(
            event_type="ask",
            summary="未來星織询问我是否知道游戏TUNIC。",
            conv_type="group",
            conv_id="100",
            roles=[{"role": "theme", "entity": "Game:TUNIC"}],
            occurred_at=1_000,
        )
        second_id = await write_event(
            event_type="say",
            summary="未來星織评价TUNIC为神作，并指出它具有meta元素。",
            conv_type="group",
            conv_id="100",
            roles=[{"role": "theme", "entity": "Game:TUNIC"}],
            occurred_at=2_000,
        )
        return first_id, second_id

    event_ids = asyncio.run(scenario())

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs, summary_id_for_source

    class FakeSummaryAdapter:
        def __init__(self):
            self.calls = 0

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls += 1
            assert log_tag == "memory_consolidation/summary"
            assert user_content.startswith("<task>\n<previous_storyline>\n</previous_storyline>")
            assert "summary_id" not in user_content
            assert "source_id" not in user_content
            assert "roles" not in user_content
            assert "relations" not in user_content
            assert "policy" not in user_content
            return """<analysis>两个事件属于同一条讨论。</analysis>
<storyline>我记得未來星織询问我是否知道 TUNIC，并评价它是带有 meta 元素的神作。</storyline>"""

    adapter = FakeSummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    task_id = summary_id_for_source("storyline", "local:tunic")
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_storyline_summary_task(
            con,
            task_id=task_id,
            storyline_id="local:tunic",
            event_ids=event_ids,
        )
        stats = process_active_summary_inputs(con, now_ms=3).to_dict()
        row = con.execute(
            "SELECT model, summary, status FROM MemorySummaryCache WHERE summary_id=?",
            (task_id,),
        ).fetchone()

        assert adapter.calls == 1
        assert stats["summary_llm_calls"] == 1
        assert stats["summaries_ready"] == 1
        assert row == (
            "memory_consolidation.storyline_summary.v2",
            "我记得未來星織询问我是否知道 TUNIC，并评价它是带有 meta 元素的神作。",
            "ready",
        )
        assert con.execute("SELECT status FROM MemoryStorylineSummaryTasks WHERE task_id=?", (task_id,)).fetchone()[0] == "done"


def test_storyline_summary_xml_contract_escapes_orders_and_parses_only_storyline():
    from memory.sleep.summary_worker import (
        _build_storyline_summary_user_prompt,
        _parse_storyline_summary_response,
    )

    prompt = _build_storyline_summary_user_prompt(
        "我记得 A < B & C。",
        [
            {"event_id": 2, "summary": '后来的 "事件" & 结果', "occurred_at": 2_000, "confidence": 2},
            {"event_id": 1, "summary": "较早的 <事件>", "occurred_at": 1_000, "confidence": -1},
        ],
    )

    assert prompt.index("较早的 &lt;事件&gt;") < prompt.index("后来的 &quot;事件&quot; &amp; 结果")
    assert 'occurred_at="1970-01-01T00:00:01+00:00" confidence="0.00"' in prompt
    assert 'occurred_at="1970-01-01T00:00:02+00:00" confidence="1.00"' in prompt
    assert "我记得 A &lt; B &amp; C。" in prompt
    assert _parse_storyline_summary_response(
        "<analysis>这部分不保存。</analysis><storyline>我记得 A &lt; B。</storyline>"
    ) == "我记得 A < B。"
    assert _parse_storyline_summary_response("<storyline></storyline>") == ""
    assert _parse_storyline_summary_response("<analysis>无需更新。</analysis></storyline>") == ""
    long_summary = "长" * 1_000
    assert _parse_storyline_summary_response(f"<storyline>{long_summary}</storyline>") == long_summary
    with pytest.raises(ValueError, match="missing <storyline>"):
        _parse_storyline_summary_response("普通文本")


def test_summary_worker_empty_storyline_is_successful_no_update(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-empty")
    import app_state

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    class EmptySummaryAdapter:
        def __init__(self):
            self.calls = 0

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls += 1
            return "</storyline>" if self.calls == 1 else "<storyline></storyline>"

    adapter = EmptySummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, summary,
                created_at_ms, updated_at_ms
            ) VALUES ('summary:storyline:existing', 'summary:storyline:existing',
                      'old', 'test', 'stale', '保留旧故事线。', 1, 1)
            """
        )
        _insert_storyline_summary_task(
            con,
            task_id="summary:storyline:existing",
            storyline_id="storyline:existing",
            input_hash="existing-input",
        )
        _insert_storyline_summary_task(
            con,
            task_id="summary:storyline:new",
            storyline_id="storyline:new",
            input_hash="new-input",
        )

        stats = process_active_summary_inputs(con, max_inputs=2, now_ms=3).to_dict()
        rows = con.execute(
            "SELECT summary_id, input_hash, status, summary FROM MemorySummaryCache ORDER BY summary_id"
        ).fetchall()

        assert stats["summaries_ready"] == 2
        assert rows == [
            ("summary:storyline:existing", "existing-input", "ready", "保留旧故事线。"),
            ("summary:storyline:new", "new-input", "ready", ""),
        ]
        assert process_active_summary_inputs(con, max_inputs=2, now_ms=4).summary_tasks_loaded == 0


def test_summary_worker_retries_failed_llm_summary_generation(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-llm-retry")
    import app_state

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    class FlakySummaryAdapter:
        def __init__(self):
            self.calls = 0

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary model error")
            return "<storyline>第二次生成成功。</storyline>"

    adapter = FlakySummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_storyline_summary_task(
            con,
            task_id="summary:storyline:retry",
            storyline_id="local:retry",
        )

        first = process_active_summary_inputs(con, now_ms=2).to_dict()
        assert first["summary_tasks_retrying"] == 1
        assert con.execute("SELECT status FROM MemoryStorylineSummaryTasks WHERE task_id='summary:storyline:retry'").fetchone()[0] == "active"

        second = process_active_summary_inputs(con, now_ms=3).to_dict()
        assert second["summaries_ready"] == 1
        assert con.execute("SELECT status FROM MemoryStorylineSummaryTasks WHERE task_id='summary:storyline:retry'").fetchone()[0] == "done"


def test_summary_worker_does_not_hold_write_lock_during_llm_call(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-no-llm-write-lock")
    import app_state

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    entered_llm = threading.Event()
    release_llm = threading.Event()

    class BlockingSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            entered_llm.set()
            assert release_llm.wait(timeout=2.0)
            return "<storyline>LLM 等待期间数据库仍可写。</storyline>"

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", BlockingSummaryAdapter())

    with sqlite3.connect(db_path, check_same_thread=False) as con:
        con.execute("PRAGMA foreign_keys=ON")
        ensure_preprocessing_schema(con)
        _insert_storyline_summary_task(
            con,
            task_id="summary:storyline:no-lock",
            storyline_id="local:no-lock",
        )
        con.commit()

        errors: list[BaseException] = []

        def run_worker():
            try:
                process_active_summary_inputs(con, now_ms=2)
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_worker)
        worker.start()
        assert entered_llm.wait(timeout=2.0)
        with sqlite3.connect(db_path, timeout=0.1) as other:
            other.execute("PRAGMA busy_timeout=100")
            other.execute(
                """
                INSERT INTO MemoryPreprocessRuns (
                    component, trigger, started_at_ms, status
                ) VALUES ('lock_probe', 'test', 1, 'finished')
                """
            )
            other.commit()
        release_llm.set()
        worker.join(timeout=2.0)

        assert not worker.is_alive()
        assert errors == []
        assert con.execute("SELECT COUNT(*) FROM MemoryPreprocessRuns WHERE component='lock_probe'").fetchone()[0] == 1


def test_summary_worker_finishes_started_request_then_pauses_queue_at_deadline(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-llm-deadline")
    import app_state

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    class SlowSummaryAdapter:
        def __init__(self):
            self.calls = 0

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls += 1
            time.sleep(0.05)
            return f"<storyline>第 {self.calls} 条完成。</storyline>"

    adapter = SlowSummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_storyline_summary_task(con, task_id="summary:storyline:deadline-a", storyline_id="storyline:deadline-a", input_hash="hash-a")
        _insert_storyline_summary_task(con, task_id="summary:storyline:deadline-b", storyline_id="storyline:deadline-b", input_hash="hash-b")

        stats = process_active_summary_inputs(
            con,
            max_inputs=2,
            deadline_ms=int(time.time() * 1000) + 20,
            now_ms=3,
        ).to_dict()

        assert adapter.calls == 1
        assert stats["summaries_ready"] == 1
        assert stats["summary_queue_paused"] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryStorylineSummaryTasks WHERE status='done'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryStorylineSummaryTasks WHERE status='active'").fetchone()[0] == 1


def test_summary_worker_pauses_before_next_input_when_sleep_ends(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-sleep-paused")
    import app_state

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    class OneLineSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            return "<storyline>本轮处理一条后暂停。</storyline>"

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", OneLineSummaryAdapter())

    calls = 0

    def should_continue() -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_storyline_summary_task(con, task_id="summary:storyline:pause-a", storyline_id="storyline:pause-a", input_hash="hash-a")
        _insert_storyline_summary_task(con, task_id="summary:storyline:pause-b", storyline_id="storyline:pause-b", input_hash="hash-b")

        stats = process_active_summary_inputs(
            con,
            max_inputs=2,
            should_continue=should_continue,
            now_ms=3,
        ).to_dict()

        assert calls == 2
        assert stats["summaries_ready"] == 1
        assert stats["summary_queue_paused"] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryStorylineSummaryTasks WHERE status='done'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryStorylineSummaryTasks WHERE status='active'").fetchone()[0] == 1


def test_storyline_cache_retires_removed_members_and_revises_same_size_storyline(tmp_path):
    from memory.sleep.consolidation import (
        StorylineMember,
        StorylineSummary,
        create_storyline_run,
        ensure_preprocessing_schema,
        write_storyline_cache,
    )

    db_path = tmp_path / "storyline-membership-sync.sqlite3"
    with sqlite3.connect(db_path) as con:
        ensure_preprocessing_schema(con)
        first_run = create_storyline_run(
            con,
            profile="algorithmic",
            trigger="test.first",
            event_ids=(1, 2),
            now_ms=1,
        )
        first = StorylineSummary(
            "recurrent-anchor:stable",
            "recurrent-anchor",
            "recurrent_anchor_candidate",
            "topic-strict",
            "role_entity:theme:work",
            2,
            0.45,
            (1, 2),
            json.dumps({"event_ids": [1, 2]}),
        )
        write_storyline_cache(
            con,
            [first],
            [
                StorylineMember(first.storyline_id, 1, 0.45, 1, "{}"),
                StorylineMember(first.storyline_id, 2, 0.45, 2, "{}"),
            ],
            run_id=first_run,
            now_ms=1,
        )
        second_run = create_storyline_run(
            con,
            profile="algorithmic",
            trigger="test.second",
            event_ids=(1, 3),
            now_ms=2,
        )
        second = StorylineSummary(
            first.storyline_id,
            first.scope,
            first.scheme_name,
            first.profile,
            first.anchor_key,
            2,
            0.45,
            (1, 3),
            json.dumps({"event_ids": [1, 3]}),
        )
        write_storyline_cache(
            con,
            [second],
            [
                StorylineMember(second.storyline_id, 1, 0.45, 1, "{}"),
                StorylineMember(second.storyline_id, 3, 0.45, 2, "{}"),
            ],
            run_id=second_run,
            now_ms=2,
        )

        assert con.execute(
            "SELECT revision FROM MemoryStorylines WHERE storyline_id=?",
            (second.storyline_id,),
        ).fetchone()[0] == 2
        assert dict(
            con.execute(
                "SELECT event_id, status FROM MemoryStorylineMembers WHERE storyline_id=? ORDER BY event_id",
                (second.storyline_id,),
            )
        ) == {1: "active", 2: "inactive", 3: "active"}


def test_active_recall_memory_tool_uses_summary_replacement(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-active-summary-recall")
    import app_state
    from types import SimpleNamespace

    app_state.config = {
        "memory": {
            "memory_predicate_similarity_threshold": 0.1,
            "memory_recall_max_results": 8,
            "memory_recall_recent_fallback": True,
            "embedding": {"provider": "hash", "dim": 64},
        }
    }

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        await write_event(
            event_type="observe",
            summary="无关 recent fallback 噪音。",
            conv_type="group",
            conv_id="100",
            roles=[{"role": "agent", "entity": "User:qq_999"}],
            occurred_at=3_000,
        )
        event_id = await write_event(
            event_type="complete",
            summary="小白完成以撒挑战。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "User:qq_42"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=2_000,
        )
        return event_id

    event_id = asyncio.run(scenario())

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import summary_id_for_source
    from tools.core import recall_memory

    storyline_id = "local:active-isaac"
    summary_id = summary_id_for_source("storyline", storyline_id)
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemoryStorylines (
                storyline_id, scope, scheme_name, anchor_key, profile, status,
                created_at, updated_at, member_count, score, signature_json
            ) VALUES (?, 'local', 'llm_local_storyline', ?, 'test', 'active', 1, 3, 1, 0.9, '{}')
            """,
            (storyline_id, storyline_id),
        )
        con.execute(
            """
            INSERT INTO MemoryStorylineMembers (
                storyline_id, event_id, score, rank, status, first_seen_at, last_seen_at, evidence_json
            ) VALUES (?, ?, 0.9, 1, 'active', 1, 3, '{}')
            """,
            (storyline_id, event_id),
        )
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, summary,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, 'hash-ready', 'test', 'ready', ?, 1, 3)
            """,
            (summary_id, summary_id, "小白完成了以撒挑战线。"),
        )
        con.commit()

    monkeypatch.setattr(app_state, "main_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(recall_memory, "run_coroutine_sync", lambda coro, loop, timeout=None: asyncio.run(coro))
    session = SimpleNamespace(
        context_messages=[{"role": "user", "sender_id": "42"}],
        conv_type="group",
        conv_id="100",
    )

    result = recall_memory.make_handler(session)("以撒")

    assert result["found"] >= 1
    summaries = [item["summary"] for item in result["memories"]]
    assert result["memories"][0]["summary"] == "小白完成了以撒挑战线。"
    assert "小白完成了以撒挑战线。" in summaries
    assert all("小白完成以撒挑战" not in item for item in summaries)
    assert any(item["kind"] == "summary" for item in result["memories"])
