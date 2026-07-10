from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path


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


def _insert_cluster_summary_task(
    con: sqlite3.Connection,
    *,
    task_id: str,
    cluster_id: str,
    task_type: str = "refresh",
    cluster_revision: int = 1,
    input_hash: str = "hash",
    priority: int = 30,
    confidence_tier: str = "medium",
    status: str = "active",
    event_ids: tuple[int, ...] = (),
    relation_rows: tuple[tuple[str, int, int, str, str, float], ...] = (),
    now_ms: int = 1,
) -> None:
    con.execute(
        """
        INSERT INTO MemoryClusterSummaryTasks (
            task_id, task_type, cluster_id, cluster_revision, input_hash,
            priority, confidence_tier, status, created_at_ms, updated_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, task_type, cluster_id, cluster_revision, input_hash, priority, confidence_tier, status, now_ms, now_ms),
    )
    con.executemany(
        """
        INSERT INTO MemoryClusterSummaryTaskEvents (task_id, event_id, rank, role, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        [(task_id, event_id, index, "cluster_member") for index, event_id in enumerate(event_ids, start=1)],
    )
    con.executemany(
        """
        INSERT INTO MemoryClusterSummaryTaskRelations (
            task_id, relation_id, source_event_id, target_event_id, relation_type, status, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [(task_id, *row) for row in relation_rows],
    )


def test_preprocessing_schema_is_additive_and_keeps_memory_write_path(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-schema")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        return await write_event(
            event_type="prefer",
            summary="用户喜欢简洁回答。",
            conv_type="group",
            conv_id="100",
            roles=[{"role": "experiencer", "entity": "Person:用户"}],
        )

    event_id = asyncio.run(scenario())
    with _connect(db_path) as con:
        tables = {
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert event_id > 0
        assert {
            "MemoryMounts",
            "MemoryThreadStates",
            "MemoryClusterRelations",
            "MemoryClusterSummaryTasks",
            "MemoryClusterSummaryTaskEvents",
            "MemoryClusterSummaryTaskRelations",
            "MemorySummaryCache",
            "MemoryCanonicalEntities",
        } <= tables
        summary_cache_cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(MemorySummaryCache)")
        }
        assert "cluster_summary_json" in summary_cache_cols
        assert "task_id" in summary_cache_cols
        assert "packet_id" not in summary_cache_cols
        assert con.execute("SELECT COUNT(*) FROM MemoryEvents").fetchone()[0] == 1


def test_preprocessing_schema_has_no_legacy_summary_input_tables(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema

    db_path = tmp_path / "current-summary-schema.sqlite3"
    with sqlite3.connect(db_path) as con:
        ensure_preprocessing_schema(con)

        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols = {row[1] for row in con.execute("PRAGMA table_info(MemorySummaryCache)")}
        assert "cluster_summary_json" in cols
        assert "task_id" in cols
        assert "MemorySummaryInputs" not in tables
        assert "MemorySummaryInputEvents" not in tables
        assert "MemorySummaryInputRelations" not in tables


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
        assert "MemoryClusterSummaryTasks" in tables
        assert "MemoryClusterSummaryTaskEvents" in tables
        assert "MemoryClusterSummaryTaskRelations" in tables
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
        assert report.episodes == 0
        assert con.execute("SELECT COUNT(*) FROM MemoryEventRelations").fetchone()[0] == 0


def test_mount_consolidation_preserves_model_relation_and_rejects_obsolete_anchor():
    from memory.sleep.consolidation import (
        MemoryMount,
        ClusterSummaryRecord,
        consolidate_memory_mounts,
    )

    card = ClusterSummaryRecord(
        summary_id="cluster:badge:summary",
        source_kind="cluster",
        source_id="cluster:badge",
        revision=2,
        title="A 的徽章",
        short_summary="A 买了新徽章。",
        core_entities=("Person:A", "Item:徽章"),
        open_slots=("correction",),
        source_event_ids=(10,),
    )
    mounts = [
        MemoryMount("m-background", 11, card.summary_id, "cluster", "cluster:badge", 2, "background_only", 0.55, "B 说徽章很贵"),
        MemoryMount("m-obsolete", 12, card.summary_id, "cluster", "cluster:badge", 1, "updates_state", 0.8, "过期锚点"),
    ]

    result = consolidate_memory_mounts([card], mounts)

    assert {item.status for item in result.mount_status_updates} == {"accepted", "obsolete"}
    assert len(result.new_relations) == 1
    assert result.new_relations[0].relation_type == "background_only"
    assert result.stale_summary_ids == (card.summary_id,)


def test_correction_relation_is_preserved_for_summary_llm_without_local_rewrite(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-correction")
    from memory.sleep.consolidation import (
        ClusterRelation,
        MemoryMount,
        ClusterSummaryRecord,
        ensure_preprocessing_schema,
        cluster_summary_to_json,
        write_cluster_relations,
        write_consolidation_result,
        write_memory_mounts,
        consolidate_memory_mounts,
    )

    card = ClusterSummaryRecord(
        summary_id="cluster:badge:summary",
        source_kind="cluster",
        source_id="cluster:badge",
        revision=1,
        title="A 的徽章",
        short_summary="A 买了新徽章。",
        core_entities=("Person:A", "Item:徽章"),
        open_slots=("identity_correction",),
        source_event_ids=(100,),
    )
    mount = MemoryMount(
        "m-correct",
        102,
        card.summary_id,
        "cluster",
        card.source_id,
        1,
        "corrects_identity",
        0.86,
        "B 纠正说徽章很贵不是指 A 的徽章。",
    )
    old_relation = ClusterRelation(
        "rel-old",
        "cluster:badge",
        101,
        100,
        "same_object",
        0.72,
        "active",
        1,
        {"note": "intentional wrong relation"},
    )

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('summary-row', ?, 'hash1', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        write_cluster_relations(con, [old_relation], now_ms=1)
        write_memory_mounts(con, [mount], now_ms=2)
        result = consolidate_memory_mounts([card], [mount])
        stats = write_consolidation_result(con, result, now_ms=3)
        con.commit()

        assert stats["summary_cache_rows_stale"] == 1
        assert con.execute("SELECT status FROM MemoryClusterRelations WHERE relation_id='rel-old'").fetchone()[0] == "active"
        assert con.execute("SELECT status FROM MemoryMounts WHERE mount_id='m-correct'").fetchone()[0] == "accepted"
        assert con.execute("SELECT status FROM MemorySummaryCache WHERE task_id=?", (card.summary_id,)).fetchone()[0] == "stale"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterRevisions").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterSummaryTasks WHERE task_type='refresh'").fetchone()[0] == 1


def test_updates_state_mount_updates_thread_state_and_summary_refresh_input(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-thread")
    from memory.sleep.consolidation import (
        MemoryMount,
        ClusterSummaryRecord,
        ensure_preprocessing_schema,
        cluster_summary_to_json,
        write_consolidation_result,
        write_memory_mounts,
        consolidate_memory_mounts,
    )

    card = ClusterSummaryRecord(
        summary_id="thread:person-game:xiaobai:isaac:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=1,
        title="小白玩以撒",
        short_summary="小白开始玩《以撒的结合》。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        open_slots=("progress_update", "completion", "correction"),
        source_event_ids=(200,),
    )
    mount = MemoryMount(
        "m-progress",
        201,
        card.summary_id,
        "thread",
        card.source_id,
        1,
        "updates_state",
        0.84,
        "小白终于白金《以撒的结合》了。",
        evidence={"entity_overlap": ["person:小白", "work:以撒的结合"]},
    )

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('thread-summary-row', ?, 'hash1', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        write_memory_mounts(con, [mount], now_ms=1)
        result = consolidate_memory_mounts([card], [mount])
        stats = write_consolidation_result(con, result, now_ms=2)
        con.commit()

        state_json = con.execute(
            "SELECT state_json FROM MemoryThreadStates WHERE thread_id=?",
            (card.source_id,),
        ).fetchone()[0]
        state = json.loads(state_json)
        assert stats["thread_state_rows_updated"] == 1
        assert state["state"] == "updates_state"
        assert "open_slots" not in state
        assert state["milestones"][-1]["event_id"] == 201
        assert con.execute("SELECT COUNT(*) FROM MemoryThreadStateRevisions").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterSummaryTasks WHERE cluster_id=?", (card.source_id,)).fetchone()[0] == 1


def test_mount_consolidation_requires_explicit_solidify_before_writing(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-solidify-gate")
    from memory.sleep.consolidation import (
        MemoryMount,
        ClusterSummaryRecord,
        ensure_preprocessing_schema,
        run_mount_consolidation,
        cluster_summary_to_json,
        write_memory_mounts,
    )

    card = ClusterSummaryRecord(
        summary_id="cluster:badge:summary",
        source_kind="cluster",
        source_id="cluster:badge",
        revision=1,
        title="A 的徽章",
        short_summary="A 买了新徽章。",
        core_entities=("Person:A", "Item:徽章"),
        source_event_ids=(100,),
    )
    mount = MemoryMount(
        "m-progress",
        101,
        card.summary_id,
        "cluster",
        card.source_id,
        1,
        "updates_state",
        0.83,
        "A 后续确认徽章已经到货。",
    )

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('summary-row', ?, 'hash1', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        write_memory_mounts(con, [mount], now_ms=1)

        preview = run_mount_consolidation(con, dry_run=False, solidify=False).to_dict()
        assert preview["dry_run"] is True
        assert con.execute("SELECT status FROM MemoryMounts WHERE mount_id=?", (mount.mount_id,)).fetchone()[0] == "pending"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterRelations").fetchone()[0] == 0

        written = run_mount_consolidation(con, dry_run=False, solidify=True).to_dict()
        assert written["dry_run"] is False
        assert con.execute("SELECT status FROM MemoryMounts WHERE mount_id=?", (mount.mount_id,)).fetchone()[0] == "accepted"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterRelations").fetchone()[0] == 1


def test_mount_consolidation_does_not_accept_stale_summary_anchor(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-stale-anchor")
    from memory.sleep.consolidation import (
        MemoryMount,
        ClusterSummaryRecord,
        ensure_preprocessing_schema,
        run_mount_consolidation,
        cluster_summary_to_json,
        write_memory_mounts,
    )

    card = ClusterSummaryRecord(
        summary_id="cluster:stale:summary",
        source_kind="cluster",
        source_id="cluster:stale",
        revision=1,
        title="旧簇",
        short_summary="这是旧 summary。",
        core_entities=("Person:A",),
        source_event_ids=(100,),
    )
    mount = MemoryMount(
        "m-stale-anchor",
        101,
        card.summary_id,
        card.source_kind,
        card.source_id,
        card.revision,
        "updates_state",
        0.9,
        "新事件不能挂到 stale anchor。",
    )

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('stale-row', ?, 'old', 'test', 'stale', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        write_memory_mounts(con, [mount], now_ms=2)

        stats = run_mount_consolidation(con, dry_run=False, solidify=True).to_dict()
        con.commit()

        assert stats["mount_status_counts"] == {"obsolete": 1}
        assert con.execute("SELECT status FROM MemoryMounts WHERE mount_id=?", (mount.mount_id,)).fetchone()[0] == "obsolete"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterRelations").fetchone()[0] == 0


def test_stage_atom_to_cluster_mounts_validates_batch_local_id_and_anchor(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-stage-mounts")
    from memory.sleep.consolidation import (
        ClusterSummaryRecord,
        ensure_preprocessing_schema,
        stage_atom_to_cluster_mounts,
        cluster_summary_to_json,
    )

    card = ClusterSummaryRecord(
        summary_id="thread:isaac:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=2,
        title="小白玩以撒",
        short_summary="小白在推进《以撒的结合》。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        open_slots=("progress_update", "completion"),
        source_event_ids=(10,),
    )
    candidates = [
        {
            "new_atom_local_id": "n1",
            "anchor_summary_id": card.summary_id,
            "anchor_revision": 2,
            "relation_type": "updates_state",
            "confidence": 0.84,
            "evidence_text": "小白终于白金《以撒的结合》了。",
            "uncertainty_reason": "",
            "_raw_mount_json": '{"new_atom_local_id":"n1"}',
        },
        {
            "new_atom_local_id": "missing",
            "anchor_summary_id": card.summary_id,
            "anchor_revision": 2,
            "relation_type": "updates_state",
            "confidence": 0.84,
            "evidence_text": "未知本批事件。",
        },
        {
            "new_atom_local_id": "n1",
            "anchor_summary_id": card.summary_id,
            "anchor_revision": 1,
            "relation_type": "updates_state",
            "confidence": 0.84,
            "evidence_text": "过期 revision。",
        },
        {
            "new_atom_local_id": "n1",
            "anchor_summary_id": "hallucinated:summary",
            "anchor_revision": 1,
            "relation_type": "updates_state",
            "confidence": 0.84,
            "evidence_text": "不存在的 anchor。",
        },
        {
            "new_atom_local_id": "n1",
            "anchor_summary_id": card.summary_id,
            "anchor_revision": 2,
            "relation_type": "",
            "confidence": 0.84,
            "evidence_text": "空关系不允许 staging。",
        },
    ]

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('thread-summary-row', ?, 'hash1', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        result = stage_atom_to_cluster_mounts(
            con,
            candidates,
            local_event_ids={"n1": 20},
            now_ms=2,
        )
        stats = result.to_dict()
        con.commit()

        row = con.execute(
            """
            SELECT new_event_id, anchor_source_kind, anchor_source_id, anchor_revision,
                   relation_type, status, evidence_text
            FROM MemoryMounts
            """
        ).fetchone()
        assert stats["mount_candidates"] == 5
        assert stats["mounts_staged"] == 1
        assert len(stats["mount_errors"]) == 4
        assert row == (
            20,
            "thread",
            "thread:Person:小白:Work:以撒的结合",
            2,
            "updates_state",
            "pending",
            "小白终于白金《以撒的结合》了。",
        )


def test_post_archive_mount_workflow_llm_mount_uses_recent_ready_cards_without_candidates(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-consolidation-llm-post-archive-mounts")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        old_id = await write_event(
            event_type="start",
            summary="小白开始推进《以撒的结合》。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
        )
        new_id = await write_event(
            event_type="complete",
            summary="小白完成了《以撒的结合》的白金进度。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
        )
        return old_id, new_id

    old_id, new_id = asyncio.run(scenario())

    import app_state
    from memory.sleep.consolidation import (
        ClusterSummaryRecord,
        ensure_preprocessing_schema,
        cluster_summary_to_json,
    )
    from memory.post_archive.mount_workflow import run_post_archive_mount_workflow

    card = ClusterSummaryRecord(
        summary_id="thread:isaac:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=1,
        title="小白玩以撒",
        short_summary="小白在推进《以撒的结合》。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        open_slots=("progress_update", "completion"),
        source_event_ids=(old_id,),
    )

    class FakeMountAdapter:
        def __init__(self):
            self.calls = []

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "gen": gen,
                    "log_tag": log_tag,
                }
            )
            return json.dumps(
                {
                    "mounts": [
                        {
                            "new_atom_local_id": "N1",
                            "anchor_summary_id": card.summary_id,
                            "anchor_revision": 1,
                            "relation_type": "updates_state",
                            "confidence": 0.82,
                            "evidence_text": "新事件完成了 anchor 中的小白以撒进度。",
                        }
                    ]
                },
                ensure_ascii=False,
            )

    fake_adapter = FakeMountAdapter()
    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {
            "enabled": True,
            "llm_mount_enabled": True,
            "algorithmic_clustering_enabled": True,
            "generation": {"temperature": 0.2, "max_output_tokens": 4000},
        },
    )
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", fake_adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, 'hash1', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (
                card.summary_id,
                card.summary_id,
                card.title,
                card.short_summary,
                cluster_summary_to_json(card),
            ),
        )

        stats = run_post_archive_mount_workflow(
            con,
            new_event_ids=[new_id],
            candidate_event_ids=[],
            now_ms=3,
        )
        con.commit()

        row = con.execute(
            """
            SELECT new_event_id, anchor_summary_id, anchor_revision, relation_type,
                   confidence, status, evidence_text, evidence_json
            FROM MemoryMounts
            """
        ).fetchone()
        evidence = json.loads(row[7])
        assert stats["mount_mode"] == "llm"
        assert stats["candidate_event_ids"] == 0
        assert stats["cluster_summaries_loaded"] == 1
        assert stats["mounts_proposed"] == 1
        assert stats["mounts_staged"] == 1
        assert stats["model_errors"] == []
        assert stats["mount_errors"] == []
        assert fake_adapter.calls[0]["log_tag"] == "memory_consolidation/mount"
        payload = json.loads(fake_adapter.calls[0]["user_content"])
        assert payload["new_atoms"][0]["local_id"] == "N1"
        assert payload["anchors"][0]["summary_id"] == card.summary_id
        assert row[:7] == (
            new_id,
            card.summary_id,
            1,
            "updates_state",
            0.82,
            "pending",
            "新事件完成了 anchor 中的小白以撒进度。",
        )
        assert evidence["generator"] == "post_archive_mount_workflow.model_candidate"
        assert evidence["new_atom_local_id"] == "N1"


def test_post_archive_mount_workflow_llm_atom_links_history_atoms_without_summary_anchor(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-consolidation-llm-history-atom-links")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        old_id = await write_event(
            event_type="tell",
            summary="未來星織告知我华风的本名是公孙车。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:未來星織"},
                {"role": "theme", "entity": "Person:华风"},
            ],
        )
        new_id = await write_event(
            event_type="claim",
            summary="未來星織声称公孙车现在叫华车。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:未來星織"},
                {"role": "theme", "entity": "Person:华风"},
            ],
        )
        return old_id, new_id

    old_id, new_id = asyncio.run(scenario())

    import app_state
    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.post_archive.mount_workflow import run_post_archive_mount_workflow

    class FakeAtomLinkAdapter:
        def __init__(self):
            self.calls = []

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "gen": gen,
                    "log_tag": log_tag,
                }
            )
            payload = json.loads(user_content)
            assert payload["anchors"] == []
            assert payload["new_atoms"][0]["local_id"] == "N1"
            assert payload["historical_atoms"][0]["local_id"] == "H1"
            assert payload["historical_atoms"][0]["event_id"] == old_id
            return json.dumps(
                {
                    "mounts": [],
                    "atom_links": [
                        {
                            "new_atom_local_id": "N1",
                            "historical_atom_local_id": "H1",
                            "relation_type": "updates_state",
                            "confidence": 0.84,
                            "evidence_text": "新事件延续并更新了华风/公孙车身份信息。",
                        }
                    ],
                    "local_clusters": [],
                },
                ensure_ascii=False,
            )

    fake_adapter = FakeAtomLinkAdapter()
    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {
            "enabled": True,
            "llm_mount_enabled": True,
            "generation": {"temperature": 0.2, "max_output_tokens": 4000},
        },
    )
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", fake_adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        stats = run_post_archive_mount_workflow(
            con,
            new_event_ids=[new_id],
            candidate_event_ids=[old_id],
            now_ms=3,
        )
        con.commit()

        proposal = con.execute(
            """
            SELECT event_ids_json, title, confidence, status, evidence_json
            FROM MemoryLocalClusterMounts
            """
        ).fetchone()
        evidence = json.loads(proposal[4])
        assert stats["mount_mode"] == "llm"
        assert stats["candidate_event_ids"] == 1
        assert stats["historical_atoms_loaded"] == 1
        assert stats["cluster_summaries_loaded"] == 0
        assert stats["mounts_staged"] == 0
        assert stats["atom_links_proposed"] == 1
        assert stats["atom_links_staged"] == 1
        assert stats["atom_link_errors"] == []
        assert fake_adapter.calls[0]["log_tag"] == "memory_consolidation/mount"
        assert json.loads(proposal[0]) == [old_id, new_id]
        assert proposal[2:4] == (0.84, "pending")
        assert "华风" in proposal[1]
        assert evidence["generator"] == "post_archive_mount_workflow.atom_link_candidate"
        assert evidence["new_atom_local_id"] == "N1"
        assert evidence["historical_atom_local_id"] == "H1"
        assert evidence["relation_type"] == "updates_state"
        assert con.execute("SELECT COUNT(*) FROM MemoryMounts").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM MemoryClusters").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterRelations").fetchone()[0] == 0


def test_post_archive_mount_workflow_does_not_hold_write_lock_during_llm_call(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-consolidation-llm-mount-no-lock")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        return await write_event(
            event_type="observe",
            summary="小白继续推进以撒。",
            conv_type="group",
            conv_id="100",
            roles=[{"role": "agent", "entity": "Person:小白"}],
        )

    new_id = asyncio.run(scenario())

    import app_state
    from memory.post_archive.mount_workflow import run_post_archive_mount_workflow

    entered_llm = threading.Event()
    release_llm = threading.Event()

    class BlockingMountAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            entered_llm.set()
            assert release_llm.wait(timeout=2.0)
            return json.dumps({"mounts": [], "local_clusters": []}, ensure_ascii=False)

    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {"enabled": True, "llm_mount_enabled": True},
    )
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", BlockingMountAdapter())

    errors: list[BaseException] = []

    def run_worker():
        try:
            run_post_archive_mount_workflow(
                db_path,
                new_event_ids=[new_id],
                candidate_event_ids=[],
                now_ms=3,
            )
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
            ) VALUES ('mount_lock_probe', 'test', 1, 'finished')
            """
        )
        other.commit()
    release_llm.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    with _connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM MemoryPreprocessRuns WHERE component='mount_lock_probe'").fetchone()[0] == 1


def test_post_archive_mount_workflow_llm_local_cluster_stages_until_sleep_solidify(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-consolidation-llm-local-cluster")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        ids = []
        for summary, event_type, roles in [
            (
                "未來星織询问我是否知道游戏TUNIC。",
                "ask",
                [
                    {"role": "agent", "entity": "Person:未來星織"},
                    {"role": "recipient", "entity": "self"},
                    {"role": "theme", "entity": "Game:TUNIC"},
                ],
            ),
            (
                "TUNIC是一款独立游戏，主角是一只小狐狸，风格类似塞尔达传说。",
                "be",
                [{"role": "theme", "entity": "Game:TUNIC"}],
            ),
            (
                "未來星織评价TUNIC为神作，并指出它具有meta元素。",
                "say",
                [
                    {"role": "agent", "entity": "Person:未來星織"},
                    {"role": "theme", "entity": "Game:TUNIC"},
                ],
            ),
        ]:
            ids.append(
                await write_event(
                    event_type=event_type,
                    summary=summary,
                    conv_type="group",
                    conv_id="100",
                    roles=roles,
                )
            )
        return ids

    event_ids = asyncio.run(scenario())

    import app_state
    from memory.sleep.consolidation import ensure_preprocessing_schema, run_mount_consolidation, cluster_summary_from_json
    from memory.post_archive.mount_workflow import run_post_archive_mount_workflow
    from memory.sleep.summary_worker import run_summary_refresh_worker

    class FakeLocalClusterAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            if log_tag == "memory_consolidation/summary":
                return json.dumps(
                    {
                        "title": "TUNIC 讨论",
                        "summary": "未來星織围绕 TUNIC 询问认知、补充游戏介绍，并评价其为带有 meta 元素的神作。",
                        "core_entities": ["Game:TUNIC", "Person:未來星織"],
                        "confirmed_claims": ["TUNIC 被描述为独立游戏。", "未來星織评价 TUNIC 具有 meta 元素。"],
                        "uncertain_claims": [],
                        "disputed_claims": [],
                        "current_state": "observed",
                        "open_slots": ["new_evidence"],
                        "boundary_notes": [],
                    },
                    ensure_ascii=False,
                )
            payload = json.loads(user_content)
            assert payload["anchors"] == []
            assert [item["local_id"] for item in payload["new_atoms"]] == ["N1", "N2", "N3"]
            return json.dumps(
                {
                    "mounts": [],
                    "local_clusters": [
                        {
                            "new_atom_local_ids": ["N1", "N2", "N3"],
                            "title": "TUNIC 讨论",
                            "confidence": 0.86,
                            "evidence_text": "三条新事件都围绕TUNIC的认知、评价和meta特征。",
                        }
                    ],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {
            "enabled": True,
            "llm_mount_enabled": True,
            "generation": {"temperature": 0.2, "max_output_tokens": 4000},
        },
    )
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeLocalClusterAdapter())

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        stats = run_post_archive_mount_workflow(
            con,
            new_event_ids=event_ids,
            candidate_event_ids=[],
            now_ms=3,
        )
        con.commit()

        proposal = con.execute(
            """
            SELECT proposal_id, event_ids_json, title, confidence, status, evidence_json
            FROM MemoryLocalClusterMounts
            """
        ).fetchone()
        assert stats["mount_mode"] == "llm"
        assert stats["cluster_summaries_loaded"] == 0
        assert stats["mounts_staged"] == 0
        assert stats["local_clusters_proposed"] == 1
        assert stats["local_clusters_staged"] == 1
        assert stats["summaries_ready"] == 0
        assert json.loads(proposal[1]) == event_ids
        assert proposal[2:5] == ("TUNIC 讨论", 0.86, "pending")
        assert json.loads(proposal[5])["generator"] == "post_archive_mount_workflow.local_cluster_candidate"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusters WHERE scope='local'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM MemorySummaryCache").fetchone()[0] == 0

        dry_run_stats = run_mount_consolidation(con, dry_run=True, solidify=False).to_dict()
        assert dry_run_stats["pending_local_cluster_mounts_loaded"] == 1
        assert dry_run_stats["local_cluster_rows_written"] == 0
        assert con.execute("SELECT status FROM MemoryLocalClusterMounts").fetchone()[0] == "pending"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusters WHERE scope='local'").fetchone()[0] == 0

        solidify_stats = run_mount_consolidation(con, dry_run=False, solidify=True).to_dict()
        assert solidify_stats["local_cluster_rows_written"] == 1
        assert solidify_stats["local_cluster_member_rows_written"] == 3
        assert con.execute("SELECT status FROM MemoryLocalClusterMounts").fetchone()[0] == "accepted"

        summary_stats = run_summary_refresh_worker(
            con,
            max_inputs=8,
            priority_cluster_ids=solidify_stats["local_cluster_ids_written"],
            now_ms=4,
        ).to_dict()
        assert summary_stats["summaries_ready"] == 1

        cluster = con.execute(
            """
            SELECT cluster_id, scope, scheme_name, profile, member_count, score, signature_json
            FROM MemoryClusters
            WHERE scope='local'
            """
        ).fetchone()
        member_ids = [
            row[0]
            for row in con.execute(
                "SELECT event_id FROM MemoryClusterMembers WHERE cluster_id=? ORDER BY rank",
                (cluster[0],),
            )
        ]
        summary_row = con.execute(
            """
            SELECT cluster_summary_json
            FROM MemorySummaryCache
            WHERE status='ready' AND cluster_summary_json <> '{}'
            """
        ).fetchone()
        card = cluster_summary_from_json(summary_row[0])
        evidence = json.loads(cluster[6])

        assert cluster[:6] == (
            cluster[0],
            "local",
            "llm_local_cluster",
            "sleep-consolidated",
            3,
            0.86,
        )
        assert evidence["generator"] == "sleep_mount_consolidation.local_cluster"
        assert member_ids == event_ids
        assert set(card.source_event_ids) == set(event_ids)
        assert "TUNIC" in card.short_summary
        assert con.execute("SELECT COUNT(*) FROM MemoryMounts").fetchone()[0] == 0


def test_post_archive_mount_workflow_stages_runtime_flow_local_cluster_without_solidifying(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-consolidation-llm-local-cluster-runtime-flow")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        ids = []
        for index, event_type in enumerate(
            [
                "say",
                "decide",
                "say",
                "decide",
                "wake_up",
                "decide",
                "search",
                "decide",
                "wake_up",
                "observe",
                "decide",
            ],
            start=1,
        ):
            ids.append(
                await write_event(
                    event_type=event_type,
                    summary=f"运行流水事件 {index}: idle/search/window 切换。",
                    conv_type="group",
                    conv_id="100",
                    roles=[
                        {"role": "agent", "entity": "self"},
                        {"role": "recipient", "entity": "Person:未來星織"},
                    ],
                )
            )
        return ids

    event_ids = asyncio.run(scenario())

    import app_state
    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.post_archive.mount_workflow import run_post_archive_mount_workflow

    class FakeRuntimeFlowAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            assert "local_clusters" in system_prompt
            return json.dumps(
                {
                    "mounts": [],
                    "local_clusters": [
                        {
                            "new_atom_local_ids": [f"N{i}" for i in range(1, 12)],
                            "title": "idle/search/window 连续行为链",
                            "confidence": 0.92,
                            "evidence_text": "这些事件在时间和主体上连续。",
                        }
                    ],
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {
            "enabled": True,
            "llm_mount_enabled": True,
            "generation": {"temperature": 0.2, "max_output_tokens": 4000},
        },
    )
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeRuntimeFlowAdapter())

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        stats = run_post_archive_mount_workflow(
            con,
            new_event_ids=event_ids,
            candidate_event_ids=[],
            now_ms=3,
        )

        assert stats["local_clusters_proposed"] == 1
        assert stats["local_clusters_staged"] == 1
        assert stats["local_cluster_errors"] == []
        assert con.execute("SELECT COUNT(*) FROM MemoryLocalClusterMounts WHERE status='pending'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryClusters WHERE scope='local'").fetchone()[0] == 0
        assert con.execute("SELECT COUNT(*) FROM MemorySummaryCache").fetchone()[0] == 0


def test_candidate_cluster_summaries_resolve_refreshed_summary_links_outside_recency_scan(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-candidate-refreshed-summary-link")

    from memory.sleep.consolidation import ClusterSummaryRecord, ensure_preprocessing_schema, cluster_summary_to_json
    from memory.post_archive.mount_workflow import load_candidate_cluster_summaries

    target = ClusterSummaryRecord(
        summary_id="thread:refreshed:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=2,
        title="小白玩以撒",
        short_summary="小白完成了以撒挑战线。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        source_event_ids=(42,),
    )
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_cluster_summary_task(
            con,
            task_id=target.summary_id,
            task_type="refresh",
            cluster_id=target.source_id,
            cluster_revision=target.revision,
            input_hash="hash-refresh",
            priority=90,
            confidence_tier="high",
            status="done",
            event_ids=(42,),
            now_ms=2,
        )
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, 'hash-ready', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (
                target.summary_id,
                target.summary_id,
                target.title,
                target.short_summary,
                cluster_summary_to_json(target),
            ),
        )
        for index in range(200):
            filler = ClusterSummaryRecord(
                summary_id=f"filler:{index}",
                source_kind="thread",
                source_id=f"thread:filler:{index}",
                revision=1,
                title=f"filler {index}",
                short_summary=f"filler summary {index}",
                source_event_ids=(10_000 + index,),
            )
            con.execute(
                """
                INSERT INTO MemorySummaryCache (
                    summary_id, task_id, input_hash, model, status, title, short_summary,
                    cluster_summary_json, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, 'test', 'ready', ?, ?, ?, ?, ?)
                """,
                (
                    filler.summary_id,
                    filler.summary_id,
                    f"hash-filler-{index}",
                    filler.title,
                    filler.short_summary,
                    cluster_summary_to_json(filler),
                    10 + index,
                    10 + index,
                ),
            )

        cards = load_candidate_cluster_summaries(con, [42], max_cluster_summaries=32)

    assert [card.summary_id for card in cards] == [target.summary_id]


def test_summary_refresh_window_selects_delta_and_activated_old_events_then_orders_old_to_new(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-consolidation-summary-window")

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        low_old = await write_event(
            event_type="start",
            summary="小白最早开始玩《以撒的结合》。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=1_000,
        )
        hot_old = await write_event(
            event_type="progress",
            summary="小白反复提到《以撒的结合》的解锁进度。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=2_000,
        )
        new_delta = await write_event(
            event_type="complete",
            summary="小白终于白金《以撒的结合》了。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=3_000,
        )
        return low_old, hot_old, new_delta

    low_old, hot_old, new_delta = asyncio.run(scenario())

    from memory.sleep.consolidation import ClusterSummaryRecord, _build_summary_refresh_event_window

    card = ClusterSummaryRecord(
        summary_id="thread:isaac:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=1,
        title="小白玩以撒",
        short_summary="小白在推进《以撒的结合》。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        open_slots=("progress_update", "completion"),
        source_event_ids=(low_old, hot_old),
    )
    relations = [
        {
            "relation_id": "rel-complete",
            "source_event_id": new_delta,
            "target_event_id": hot_old,
            "relation_type": "updates_state",
            "status": "active",
            "confidence": 0.9,
        }
    ]

    with _connect(db_path) as con:
        con.execute(
            """
            UPDATE MemoryEvents
            SET access_count=12, last_accessed=9500
            WHERE event_id=?
            """,
            (hot_old,),
        )
        con.commit()

        window = _build_summary_refresh_event_window(
            con,
            card,
            relations,
            now_ms=10_000,
            max_events=2,
        )

        assert [item["event_id"] for item in window] == [hot_old, new_delta]
        assert [item["window_role"] for item in window] == [
            "previous_summary_source",
            "delta_new_evidence",
        ]
        assert window[0]["access_count"] == 12
        assert window[0]["activation_score"] > 0
        assert window[1]["occurred_at"] > window[0]["occurred_at"]


def test_disabling_llm_mount_does_not_fall_back_to_rule_mounting(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-no-rule-mount-fallback")
    import app_state

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        return await write_event(
            event_type="complete",
            summary="小白终于白金《以撒的结合》了。",
            conv_type="group",
            conv_id="100",
            roles=[{"role": "theme", "entity": "Work:以撒的结合"}],
            occurred_at=2_000,
        )

    event_id = asyncio.run(scenario())
    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {"enabled": True, "llm_mount_enabled": False, "algorithmic_clustering_enabled": True},
    )
    from memory.post_archive.mount_workflow import run_post_archive_mount_workflow

    stats = run_post_archive_mount_workflow(db_path, new_event_ids=[event_id])

    assert stats["mount_mode"] == "disabled"
    assert stats["mounts_proposed"] == 0
    assert stats["mounts_staged"] == 0
    with _connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM MemoryMounts").fetchone()[0] == 0


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

    from memory.sleep.consolidation import ClusterSummaryRecord, ensure_preprocessing_schema, cluster_summary_from_json, cluster_summary_to_json
    from memory.sleep.summary_worker import process_active_summary_inputs

    class FakeSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            payload = json.loads(user_content)
            assert payload["summary_id"] == card.summary_id
            assert [item["event_id"] for item in payload["events"]] == [old_id, new_id]
            return json.dumps(
                {
                    "title": "小白玩以撒",
                    "summary": "小白从开始推进《以撒的结合》更新为已经白金。",
                    "core_entities": ["Person:小白", "Work:以撒的结合"],
                    "confirmed_claims": ["小白白金《以撒的结合》。"],
                    "current_state": "completed",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeSummaryAdapter())

    card = ClusterSummaryRecord(
        summary_id="thread:isaac:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=1,
        title="小白玩以撒",
        short_summary="小白在推进《以撒的结合》。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        open_slots=("progress_update", "completion"),
        source_event_ids=(old_id,),
    )
    task_id = card.summary_id

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('old-summary-row', ?, 'old', 'test', 'stale', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        _insert_cluster_summary_task(
            con,
            task_id=task_id,
            task_type="refresh",
            cluster_id=card.source_id,
            cluster_revision=card.revision,
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
            SELECT status, cluster_summary_json
            FROM MemorySummaryCache
            WHERE summary_id=?
            """,
            (card.summary_id,),
        ).fetchone()
        refreshed = cluster_summary_from_json(row[1])

        assert stats["summaries_ready"] == 1
        assert con.execute("SELECT status FROM MemoryClusterSummaryTasks WHERE task_id=?", (task_id,)).fetchone()[0] == "done"
        assert row[0] == "ready"
        assert refreshed.revision == 2
        assert refreshed.source_event_ids == (old_id, new_id)
        assert "白金" in refreshed.short_summary


def test_summary_worker_uses_memory_consolidation_llm_for_cluster_summary(tmp_path, monkeypatch):
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
            payload = json.loads(user_content)
            assert payload["source_id"] == "local:tunic"
            assert [item["event_id"] for item in payload["events"]] == list(event_ids)
            return json.dumps(
                {
                    "title": "TUNIC 讨论",
                    "summary": "未來星織围绕 TUNIC 询问认知并评价其为带有 meta 元素的神作。",
                    "core_entities": ["Game:TUNIC", "Person:未來星織"],
                    "confirmed_claims": ["未來星織评价 TUNIC 具有 meta 元素。"],
                    "uncertain_claims": [],
                    "disputed_claims": [],
                    "current_state": "observed",
                    "open_slots": ["new_evidence"],
                    "boundary_notes": [],
                },
                ensure_ascii=False,
            )

    adapter = FakeSummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    task_id = summary_id_for_source("cluster", "local:tunic")
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_cluster_summary_task(
            con,
            task_id=task_id,
            cluster_id="local:tunic",
            event_ids=event_ids,
        )
        stats = process_active_summary_inputs(con, now_ms=3).to_dict()
        row = con.execute(
            "SELECT model, title, short_summary, status FROM MemorySummaryCache WHERE summary_id=?",
            (task_id,),
        ).fetchone()

        assert adapter.calls == 1
        assert stats["summary_llm_calls"] == 1
        assert stats["summaries_ready"] == 1
        assert row == (
            "memory_consolidation.cluster_summary.v1",
            "TUNIC 讨论",
            "未來星織围绕 TUNIC 询问认知并评价其为带有 meta 元素的神作。",
            "ready",
        )
        assert con.execute("SELECT status FROM MemoryClusterSummaryTasks WHERE task_id=?", (task_id,)).fetchone()[0] == "done"


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
            return json.dumps({"title": "重试成功", "summary": "第二次生成成功。"}, ensure_ascii=False)

    adapter = FlakySummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_cluster_summary_task(
            con,
            task_id="summary:cluster:retry",
            cluster_id="local:retry",
        )

        first = process_active_summary_inputs(con, now_ms=2).to_dict()
        assert first["summary_tasks_retrying"] == 1
        assert con.execute("SELECT status FROM MemoryClusterSummaryTasks WHERE task_id='summary:cluster:retry'").fetchone()[0] == "active"

        second = process_active_summary_inputs(con, now_ms=3).to_dict()
        assert second["summaries_ready"] == 1
        assert con.execute("SELECT status FROM MemoryClusterSummaryTasks WHERE task_id='summary:cluster:retry'").fetchone()[0] == "done"


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
            return json.dumps({"title": "无锁", "summary": "LLM 等待期间数据库仍可写。"}, ensure_ascii=False)

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", BlockingSummaryAdapter())

    with sqlite3.connect(db_path, check_same_thread=False) as con:
        con.execute("PRAGMA foreign_keys=ON")
        ensure_preprocessing_schema(con)
        _insert_cluster_summary_task(
            con,
            task_id="summary:cluster:no-lock",
            cluster_id="local:no-lock",
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
            return json.dumps({"title": f"完成 {self.calls}", "summary": f"第 {self.calls} 条完成。"}, ensure_ascii=False)

    adapter = SlowSummaryAdapter()
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_cluster_summary_task(con, task_id="summary:cluster:deadline-a", cluster_id="cluster:deadline-a", input_hash="hash-a")
        _insert_cluster_summary_task(con, task_id="summary:cluster:deadline-b", cluster_id="cluster:deadline-b", input_hash="hash-b")

        stats = process_active_summary_inputs(
            con,
            max_inputs=2,
            deadline_ms=int(time.time() * 1000) + 20,
            now_ms=3,
        ).to_dict()

        assert adapter.calls == 1
        assert stats["summaries_ready"] == 1
        assert stats["summary_queue_paused"] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterSummaryTasks WHERE status='done'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterSummaryTasks WHERE status='active'").fetchone()[0] == 1


def test_summary_worker_pauses_before_next_input_when_sleep_ends(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-summary-worker-sleep-paused")
    import app_state

    from memory.sleep.consolidation import ensure_preprocessing_schema
    from memory.sleep.summary_worker import process_active_summary_inputs

    class OneLineSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            return json.dumps({"title": "暂停", "summary": "本轮处理一条后暂停。"}, ensure_ascii=False)

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", OneLineSummaryAdapter())

    calls = 0

    def should_continue() -> bool:
        nonlocal calls
        calls += 1
        return calls == 1

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        _insert_cluster_summary_task(con, task_id="summary:cluster:pause-a", cluster_id="cluster:pause-a", input_hash="hash-a")
        _insert_cluster_summary_task(con, task_id="summary:cluster:pause-b", cluster_id="cluster:pause-b", input_hash="hash-b")

        stats = process_active_summary_inputs(
            con,
            max_inputs=2,
            should_continue=should_continue,
            now_ms=3,
        ).to_dict()

        assert calls == 2
        assert stats["summaries_ready"] == 1
        assert stats["summary_queue_paused"] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterSummaryTasks WHERE status='done'").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterSummaryTasks WHERE status='active'").fetchone()[0] == 1


def test_sleep_memory_maintenance_solidifies_mount_and_refreshes_summary(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-sleep-maintenance")
    import app_state

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        old_id = await write_event(
            event_type="start",
            summary="小白开始推进《以撒的结合》。",
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
            summary="小白终于白金《以撒的结合》了。",
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

    from memory.sleep.consolidation import MemoryMount, ClusterSummaryRecord, ensure_preprocessing_schema, cluster_summary_from_json, cluster_summary_to_json, write_memory_mounts
    from memory.sleep.sleep_maintenance import run_sleep_memory_maintenance

    class FakeSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            payload = json.loads(user_content)
            summaries = [str(item.get("summary") or "") for item in payload["events"]]
            return json.dumps(
                {
                    "title": "小白玩以撒",
                    "summary": "；".join(summaries),
                    "core_entities": ["Person:小白", "Work:以撒的结合"],
                    "current_state": "completed",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeSummaryAdapter())

    card = ClusterSummaryRecord(
        summary_id="thread:isaac:summary",
        source_kind="thread",
        source_id="thread:Person:小白:Work:以撒的结合",
        revision=1,
        title="小白玩以撒",
        short_summary="小白在推进《以撒的结合》。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        open_slots=("progress_update", "completion"),
        source_event_ids=(old_id,),
    )
    mount = MemoryMount(
        "m-complete",
        new_id,
        card.summary_id,
        card.source_kind,
        card.source_id,
        card.revision,
        "updates_state",
        0.9,
        "小白终于白金《以撒的结合》了。",
    )
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES ('old-thread-row', ?, 'old', 'test', 'ready', ?, ?, ?, 1, 1)
            """,
            (card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        write_memory_mounts(con, [mount], now_ms=2)
        con.commit()

    stats = run_sleep_memory_maintenance(
        db_path,
        trigger="test.sleep",
        config={"memory": {"consolidation": {"dry_run": False, "solidify": True, "max_mounts_per_sleep": 10}}},
    )

    with _connect(db_path) as con:
        refreshed_json = con.execute(
            "SELECT cluster_summary_json FROM MemorySummaryCache WHERE summary_id=? AND status='ready'",
            (card.summary_id,),
        ).fetchone()[0]
        refreshed = cluster_summary_from_json(refreshed_json)

        assert stats["ok"] is True
        assert con.execute("SELECT status FROM MemoryMounts WHERE mount_id=?", (mount.mount_id,)).fetchone()[0] == "accepted"
        assert con.execute("SELECT COUNT(*) FROM MemoryClusterRelations WHERE cluster_id=?", (card.source_id,)).fetchone()[0] == 1
        assert con.execute("SELECT status FROM MemoryClusterSummaryTasks WHERE task_id=?", (card.summary_id,)).fetchone()[0] == "done"
        assert refreshed.revision == 2
        assert "白金" in refreshed.short_summary


def test_sleep_memory_maintenance_zero_timeout_disables_time_deadline(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-sleep-zero-timeout")

    import memory.sleep.sleep_maintenance as sleep_maintenance
    from memory.sleep.consolidation import ConsolidationResult, MountConsolidationReport, PreprocessReport
    from memory.sleep.summary_worker import SummaryRefreshReport

    observed: dict[str, object] = {}

    monkeypatch.setattr(
        sleep_maintenance,
        "run_preprocessing",
        lambda con, *, limit, trigger, algorithmic_clustering_enabled: PreprocessReport(
            events=0,
            canonical_entities=0,
            entity_mentions=0,
            event_relations=0,
            episodes=0,
            algorithmic_clustering_enabled=algorithmic_clustering_enabled,
        ),
    )
    monkeypatch.setattr(
        sleep_maintenance,
        "run_mount_consolidation",
        lambda con, **kwargs: MountConsolidationReport(
            result=ConsolidationResult((), (), (), (), ()),
        ),
    )

    def fake_summary_worker(con, **kwargs):
        observed.update(kwargs)
        return SummaryRefreshReport(summary_tasks_loaded=0)

    monkeypatch.setattr(sleep_maintenance, "run_summary_refresh_worker", fake_summary_worker)

    stats = sleep_maintenance.run_sleep_memory_maintenance(
        db_path,
        trigger="test.sleep",
        config={
            "memory": {
                "consolidation": {
                    "sleep_maintenance_timeout_seconds": 0,
                }
            }
        },
    )

    assert stats["ok"] is True
    assert observed["deadline_ms"] is None


def test_llm_mount_and_algorithmic_clustering_switches_are_independent(monkeypatch):
    import app_state
    from memory.post_archive.mount_workflow import _llm_mount_enabled
    from memory.sleep.sleep_maintenance import SleepMaintenanceConfig

    for llm_mount_enabled in (False, True):
        for algorithmic_clustering_enabled in (False, True):
            raw = {
                "enabled": True,
                "llm_mount_enabled": llm_mount_enabled,
                "algorithmic_clustering_enabled": algorithmic_clustering_enabled,
            }
            monkeypatch.setattr(app_state, "memory_consolidation_cfg", raw)

            assert _llm_mount_enabled() is llm_mount_enabled
            assert (
                SleepMaintenanceConfig.from_raw(raw).algorithmic_clustering_enabled
                is algorithmic_clustering_enabled
            )


def test_cluster_cache_retires_removed_members_and_revises_same_size_cluster(tmp_path):
    from memory.sleep.consolidation import (
        ClusterMember,
        ClusterSummary,
        create_cluster_run,
        ensure_preprocessing_schema,
        write_cluster_cache,
    )

    db_path = tmp_path / "cluster-membership-sync.sqlite3"
    with sqlite3.connect(db_path) as con:
        ensure_preprocessing_schema(con)
        first_run = create_cluster_run(
            con,
            profile="algorithmic",
            trigger="test.first",
            event_ids=(1, 2),
            now_ms=1,
        )
        first = ClusterSummary(
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
        write_cluster_cache(
            con,
            [first],
            [
                ClusterMember(first.cluster_id, 1, 0.45, 1, "{}"),
                ClusterMember(first.cluster_id, 2, 0.45, 2, "{}"),
            ],
            run_id=first_run,
            now_ms=1,
        )
        second_run = create_cluster_run(
            con,
            profile="algorithmic",
            trigger="test.second",
            event_ids=(1, 3),
            now_ms=2,
        )
        second = ClusterSummary(
            first.cluster_id,
            first.scope,
            first.scheme_name,
            first.profile,
            first.anchor_key,
            2,
            0.45,
            (1, 3),
            json.dumps({"event_ids": [1, 3]}),
        )
        write_cluster_cache(
            con,
            [second],
            [
                ClusterMember(second.cluster_id, 1, 0.45, 1, "{}"),
                ClusterMember(second.cluster_id, 3, 0.45, 2, "{}"),
            ],
            run_id=second_run,
            now_ms=2,
        )

        assert con.execute(
            "SELECT revision FROM MemoryClusters WHERE cluster_id=?",
            (second.cluster_id,),
        ).fetchone()[0] == 2
        assert dict(
            con.execute(
                "SELECT event_id, status FROM MemoryClusterMembers WHERE cluster_id=? ORDER BY event_id",
                (second.cluster_id,),
            )
        ) == {1: "active", 2: "inactive", 3: "active"}


def test_sleep_runs_algorithmic_and_llm_mount_clusters_together(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-parallel-cluster-producers")
    import app_state

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        first_id = await write_event(
            event_type="ask",
            summary="小白询问《以撒的结合》角色解锁方法。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=1_000,
        )
        second_id = await write_event(
            event_type="answer",
            summary="小白得到《以撒的结合》角色解锁答案。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:小白"},
                {"role": "theme", "entity": "Work:以撒的结合"},
            ],
            occurred_at=2_000,
        )
        return first_id, second_id

    event_ids = asyncio.run(scenario())

    from memory.sleep.consolidation import LocalClusterMount, ensure_preprocessing_schema, write_local_cluster_mounts
    from memory.sleep.sleep_maintenance import run_sleep_memory_maintenance

    class FakeSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            payload = json.loads(user_content)
            return json.dumps(
                {
                    "title": str(payload.get("cluster", {}).get("anchor_key") or "事件簇"),
                    "summary": "；".join(str(item.get("summary") or "") for item in payload["events"]),
                    "current_state": "observed",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {
            "enabled": True,
            "llm_mount_enabled": True,
            "algorithmic_clustering_enabled": True,
            "summary_max_retries": 3,
        },
    )
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeSummaryAdapter())

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        write_local_cluster_mounts(
            con,
            [
                LocalClusterMount(
                    proposal_id="llm-local-parallel",
                    event_ids=event_ids,
                    title="LLM 挂载候选簇",
                    confidence=0.9,
                    evidence_text="模拟归档后 LLM 挂载产生的 pending local cluster。",
                )
            ],
            now_ms=3,
        )
        con.commit()

    stats = run_sleep_memory_maintenance(
        db_path,
        trigger="test.sleep.parallel",
        config={
            "memory": {
                "consolidation": {
                    "algorithmic_clustering_enabled": True,
                    "dry_run": False,
                    "solidify": True,
                    "summary_max_inputs_per_sleep": 32,
                }
            }
        },
    )

    algorithmic_ids = set(stats["preprocess"]["algorithmic_cluster_ids"])
    local_ids = set(stats["mount_consolidation"]["local_cluster_ids_written"])
    assert algorithmic_ids
    assert len(local_ids) == 1
    assert algorithmic_ids.isdisjoint(local_ids)
    assert stats["summary_worker"]["summaries_ready"] == len(algorithmic_ids | local_ids)
    with _connect(db_path) as con:
        cluster_runs = list(
            con.execute(
                "SELECT profile, trigger FROM MemoryClusterRuns ORDER BY run_id"
            )
        )
        ready_cluster_ids = {
            row[0]
            for row in con.execute(
                """
                SELECT t.cluster_id
                FROM MemoryClusterSummaryTasks t
                JOIN MemorySummaryCache c ON c.task_id=t.task_id
                WHERE t.status='done' AND c.status='ready'
                """
            )
        }
        assert ("algorithmic", "test.sleep.parallel") in cluster_runs
        assert ("sleep-consolidated", "local_cluster_mount") in cluster_runs
        assert algorithmic_ids | local_ids <= ready_cluster_ids


def test_sleep_memory_maintenance_refreshes_new_local_cluster_summary_before_backlog(tmp_path, monkeypatch):
    db_path = _fresh_db(tmp_path, "memory-sleep-local-cluster-summary-priority")
    import app_state

    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        first_id = await write_event(
            event_type="ask",
            summary="未來星織询问我是否知道游戏TUNIC。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:未來星織"},
                {"role": "theme", "entity": "Game:TUNIC"},
            ],
            occurred_at=1_000,
        )
        second_id = await write_event(
            event_type="say",
            summary="未來星織评价TUNIC为神作，并指出它具有meta元素。",
            conv_type="group",
            conv_id="100",
            roles=[
                {"role": "agent", "entity": "Person:未來星織"},
                {"role": "theme", "entity": "Game:TUNIC"},
            ],
            occurred_at=2_000,
        )
        return first_id, second_id

    event_ids = asyncio.run(scenario())

    from memory.sleep.consolidation import LocalClusterMount, ensure_preprocessing_schema, cluster_summary_from_json, write_local_cluster_mounts
    from memory.sleep.sleep_maintenance import run_sleep_memory_maintenance
    from memory.sleep.summary_worker import summary_id_for_source

    class FakeSummaryAdapter:
        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            payload = json.loads(user_content)
            summaries = [str(item.get("summary") or "") for item in payload["events"]]
            return json.dumps(
                {
                    "title": "TUNIC 讨论",
                    "summary": "；".join(summaries),
                    "core_entities": ["Game:TUNIC", "Person:未來星織"],
                    "current_state": "observed",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "summary_max_retries": 3})
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", FakeSummaryAdapter())

    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        for index in range(33):
            task_id = f"old-refresh:{index}"
            _insert_cluster_summary_task(
                con,
                task_id=task_id,
                task_type="refresh",
                cluster_id=f"old-thread:{index}",
                input_hash=f"old-hash:{index}",
                priority=90,
                confidence_tier="high",
            )
        write_local_cluster_mounts(
            con,
            [
                LocalClusterMount(
                    proposal_id="local-priority",
                    event_ids=event_ids,
                    title="TUNIC 讨论",
                    confidence=0.86,
                    evidence_text="两条新事件围绕TUNIC讨论构成同一话题。",
                )
            ],
            now_ms=2,
        )
        con.commit()

    stats = run_sleep_memory_maintenance(
        db_path,
        trigger="test.sleep",
        config={
            "memory": {
                "consolidation": {
                    "dry_run": False,
                    "solidify": True,
                    "max_mounts_per_sleep": 10,
                    "summary_max_inputs_per_sleep": 1,
                }
            }
        },
    )

    with _connect(db_path) as con:
        cluster_id = con.execute(
            "SELECT cluster_id FROM MemoryClusters WHERE scope='local'"
        ).fetchone()[0]
        summary_id = summary_id_for_source("cluster", cluster_id)
        row = con.execute(
            """
            SELECT cluster_summary_json
            FROM MemorySummaryCache
            WHERE summary_id=? AND status='ready'
            """,
            (summary_id,),
        ).fetchone()
        card = cluster_summary_from_json(row[0])

        assert stats["preprocess"]["algorithmic_clustering_enabled"] is False
        assert stats["preprocess"]["algorithmic_cluster_ids"] == []
        assert stats["mount_consolidation"]["local_cluster_ids_written"] == [cluster_id]
        assert stats["summary_worker"]["summary_tasks_loaded"] == 1
        assert stats["summary_worker"]["summaries_ready"] == 1
        assert con.execute(
            "SELECT status FROM MemoryLocalClusterMounts WHERE proposal_id='local-priority'"
        ).fetchone()[0] == "accepted"
        assert con.execute(
            "SELECT status FROM MemoryClusterSummaryTasks WHERE task_id=?",
            (summary_id,),
        ).fetchone()[0] == "done"
        assert set(card.source_event_ids) == set(event_ids)
        assert "TUNIC" in card.short_summary


def test_recall_includes_ready_summary_and_excludes_pending_mount(tmp_path):
    db_path = _fresh_db(tmp_path, "memory-summary-recall")
    import app_state

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
            summary="raw-only-secret-token 小白完成挑战。",
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

    from memory.sleep.consolidation import MemoryMount, ClusterSummaryRecord, ensure_preprocessing_schema, cluster_summary_to_json, write_memory_mounts
    from memory.recall.recall_query import build_recall_query_facets, recall_events_from_facets
    from memory.sleep.summary_worker import summary_id_for_source

    cluster_id = "local:isaac-complete"
    card = ClusterSummaryRecord(
        summary_id=summary_id_for_source("cluster", cluster_id),
        source_kind="cluster",
        source_id=cluster_id,
        revision=2,
        title="小白玩以撒",
        short_summary="小白完成了以撒挑战线。",
        core_entities=("Person:小白", "Work:以撒的结合"),
        current_state="completed",
    )
    pending = MemoryMount(
        "m-pending-secret",
        new_id,
        card.summary_id,
        card.source_kind,
        card.source_id,
        card.revision,
        "updates_state",
        0.9,
        "pending-only-secret",
    )
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemoryClusters (
                cluster_id, scope, scheme_name, anchor_key, profile, status,
                created_at, updated_at, member_count, score, signature_json
            ) VALUES (?, 'local', 'llm_local_cluster', ?, 'test', 'active', 1, 3, 2, 0.9, '{}')
            """,
            (cluster_id, cluster_id),
        )
        con.executemany(
            """
            INSERT INTO MemoryClusterMembers (
                cluster_id, event_id, score, rank, status, first_seen_at, last_seen_at, evidence_json
            ) VALUES (?, ?, 0.9, ?, 'active', 1, 3, '{}')
            """,
            [(cluster_id, old_id, 1), (cluster_id, new_id, 2)],
        )
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, 'hash-ready', 'test', 'ready', ?, ?, ?, 1, 3)
            """,
            (card.summary_id, card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
        )
        write_memory_mounts(con, [pending], now_ms=4)
        con.commit()

    facets = build_recall_query_facets(latest_user_text="raw-only-secret-token")
    recalled = asyncio.run(
        recall_events_from_facets(
            sender_entity="Person:小白",
            context_scope="group:qq_100",
            limit=3,
            facets=facets,
        )
    )

    assert any(item.get("memory_kind") == "summary" for item in recalled)
    rendered_text = "\n".join(str(item.get("summary") or "") for item in recalled)
    assert "小白完成了以撒挑战线" in rendered_text
    assert "raw-only-secret-token" not in rendered_text
    assert "pending-only-secret" not in rendered_text
    assert all(int(item.get("event_id")) != new_id for item in recalled if str(item.get("event_id", "")).isdigit())


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

    from memory.sleep.consolidation import ClusterSummaryRecord, ensure_preprocessing_schema, cluster_summary_to_json
    from memory.sleep.summary_worker import summary_id_for_source
    from tools.core import recall_memory

    cluster_id = "local:active-isaac"
    card = ClusterSummaryRecord(
        summary_id=summary_id_for_source("cluster", cluster_id),
        source_kind="cluster",
        source_id=cluster_id,
        revision=1,
        title="小白玩以撒",
        short_summary="小白完成了以撒挑战线。",
        core_entities=("User:qq_42", "Work:以撒的结合"),
        current_state="completed",
    )
    with _connect(db_path) as con:
        ensure_preprocessing_schema(con)
        con.execute(
            """
            INSERT INTO MemoryClusters (
                cluster_id, scope, scheme_name, anchor_key, profile, status,
                created_at, updated_at, member_count, score, signature_json
            ) VALUES (?, 'local', 'llm_local_cluster', ?, 'test', 'active', 1, 3, 1, 0.9, '{}')
            """,
            (cluster_id, cluster_id),
        )
        con.execute(
            """
            INSERT INTO MemoryClusterMembers (
                cluster_id, event_id, score, rank, status, first_seen_at, last_seen_at, evidence_json
            ) VALUES (?, ?, 0.9, 1, 'active', 1, 3, '{}')
            """,
            (cluster_id, event_id),
        )
        con.execute(
            """
            INSERT INTO MemorySummaryCache (
                summary_id, task_id, input_hash, model, status, title, short_summary,
                cluster_summary_json, created_at_ms, updated_at_ms
            ) VALUES (?, ?, 'hash-ready', 'test', 'ready', ?, ?, ?, 1, 3)
            """,
            (card.summary_id, card.summary_id, card.title, card.short_summary, cluster_summary_to_json(card)),
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
