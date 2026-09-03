from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest


def _fresh_db(tmp_path, name="event-structuring"):
    import database
    from memory.repo import events

    database.DB_PATH = str(tmp_path / f"{name}.sqlite3")
    events._SCHEMA_READY = False
    return database.DB_PATH


def _write_events(db_path):
    async def scenario():
        from memory.repo.events import ensure_schema, write_event

        await ensure_schema()
        old_id = await write_event(
            event_type="question",
            summary="吹雪问我像不像人。",
            roles=[{"role": "agent", "entity": "Person:吹雪"}],
            conv_type="qq",
            conv_id="1",
            occurred_at=1,
        )
        new_ids = []
        for index, summary in enumerate(("我决定保持安静等待。", "我继续等待回复。"), start=2):
            new_ids.append(
                await write_event(
                    event_type="wait",
                    summary=summary,
                    roles=[{"role": "agent", "entity": "self"}],
                    conv_type="qq",
                    conv_id="1",
                    occurred_at=index,
                )
            )
        return old_id, new_ids

    return asyncio.run(scenario())


def test_event_structuring_user_payload_exposes_only_local_id_summary_and_entities():
    from memory.event_structuring.workflow import EventAtom, build_event_structuring_user_payload

    payload = json.loads(
        build_event_structuring_user_payload(
            [EventAtom(101, "new", ("self",))],
            [EventAtom(202, "old", ("Person:吹雪",))],
        )
    )

    assert payload == {
        "new_events": [{"id": "N1", "summary": "new", "entities": ["self"]}],
        "existing_events": [{"id": "E1", "summary": "old", "entities": ["Person:吹雪"]}],
    }
    serialized = json.dumps(payload)
    for forbidden in ("local_id", "event_id", "event_type", "status", "occurred_at", "anchors"):
        assert forbidden not in serialized


def test_parse_event_structuring_response_accepts_empty_arrays_and_deduplicates():
    from memory.event_structuring.workflow import parse_event_structuring_response

    empty = parse_event_structuring_response(
        '{"links":[],"candidate_storylines":[]}',
        new_ids={"N1", "N2"},
        historical_ids={"E1"},
    )
    assert empty.links == ()
    assert empty.candidate_storylines == ()
    assert empty.errors == ()

    result = parse_event_structuring_response(
        """{
        "links":[{"new_event":"N1","existing_event":"E1"},{"new_event":"N3","existing_event":"E2"},{"new_event":"N1","existing_event":"E1"}],
        "candidate_storylines":[["N2","N1","N1"],["N4","N3"],["N1","N2"]]
        }""",
        new_ids={"N1", "N2", "N3", "N4"},
        historical_ids={"E1", "E2"},
    )
    assert result.links == (("N1", "E1"), ("N3", "E2"))
    assert result.candidate_storylines == (("N1", "N2"), ("N3", "N4"))
    assert result.errors == ()


@pytest.mark.parametrize(
    "raw",
    [
        '[{"links":[],"candidate_storylines":[]}]',
        '{"links":[]}',
        '{"links":{},"candidate_storylines":[]}',
        '{"links":[,"candidate_storylines":[]}',
        '{"links":[{"new_event":"E1","existing_event":"E1"}],"candidate_storylines":[]}',
        '{"links":[{"new_event":"N1","existing_event":"N1"}],"candidate_storylines":[]}',
        '{"links":[{"new_event":"N1","existing_event":"E1","extra":1}],"candidate_storylines":[]}',
        '{"links":[],"candidate_storylines":[["N1"]]}',
        '{"links":[],"candidate_storylines":[["N1","E1"]]}',
    ],
)
def test_parse_event_structuring_response_rejects_invalid_contract(raw):
    from memory.event_structuring.workflow import parse_event_structuring_response

    result = parse_event_structuring_response(raw, new_ids={"N1", "N2"}, historical_ids={"E1"})
    assert result.errors


def test_event_structuring_writes_idempotent_relation_and_candidate_storyline(tmp_path, monkeypatch):
    import app_state
    from memory.event_structuring.workflow import run_event_structuring
    from memory.maintenance.preprocessing import run_candidate_storyline_consolidation

    db_path = _fresh_db(tmp_path)
    old_id, new_ids = _write_events(db_path)

    class Adapter:
        calls = []

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls.append((system_prompt, user_content, gen, log_tag))
            return '{"links":[{"new_event":"N1","existing_event":"E1"}],"candidate_storylines":[["N1","N2"]]}'

    adapter = Adapter()
    monkeypatch.setattr(app_state, "memory_processing_adapter", adapter)
    monkeypatch.setattr(
        app_state,
        "memory_processing_cfg",
        {"enabled": True, "event_structuring_enabled": True, "generation": {}},
    )

    first = run_event_structuring(
        db_path, new_event_ids=new_ids, candidate_event_ids=[old_id], now_ms=10
    )
    second = run_event_structuring(
        db_path, new_event_ids=new_ids, candidate_event_ids=[old_id], now_ms=11
    )
    assert first["links_written"] == 1
    assert second["links_written"] == 0
    assert first["candidate_storylines_staged"] == 1
    assert adapter.calls[0][3] == "memory/event_structuring"

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT src_event_id, dst_event_id, relation_type, reason FROM MemoryRelations"
        ).fetchall() == [(new_ids[0], old_id, "related", "event_structuring")]
        assert json.loads(
            con.execute("SELECT event_ids_json FROM MemoryCandidateStorylines").fetchone()[0]
        ) == new_ids

        report = run_candidate_storyline_consolidation(
            con, max_candidate_storylines=10, dry_run=False, solidify=True
        )
        con.commit()
        assert report.candidate_storylines_written == 1
        assert report.candidate_storyline_members_written == 2
        assert con.execute("SELECT status FROM MemoryCandidateStorylines").fetchone()[0] == "accepted"
        storyline_id, scope, origin_type = con.execute(
            "SELECT storyline_id, scope, origin_type FROM MemoryStorylines"
        ).fetchone()
        assert storyline_id.startswith("candidate_storyline:")
        assert scope == "candidate_storyline"
        assert origin_type == "llm_candidate_storyline"
        member_ids = {
            row[0]
            for row in con.execute(
                "SELECT event_id FROM MemoryStorylineMembers WHERE storyline_id=?",
                (report.storyline_ids_written[0],),
            )
        }
        assert member_ids == set(new_ids)


def test_event_structuring_does_not_hold_write_lock_during_model_call(tmp_path, monkeypatch):
    import app_state
    from memory.event_structuring.workflow import run_event_structuring

    db_path = _fresh_db(tmp_path, "structuring-no-lock")
    old_id, new_ids = _write_events(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE lock_probe (value INTEGER)")

    class Adapter:
        def call_simple_text(self, *_args, **_kwargs):
            with sqlite3.connect(db_path, timeout=0.2) as other:
                other.execute("INSERT INTO lock_probe(value) VALUES (1)")
            return '{"links":[],"candidate_storylines":[]}'

    monkeypatch.setattr(app_state, "memory_processing_adapter", Adapter())
    monkeypatch.setattr(app_state, "memory_processing_cfg", {"enabled": True, "event_structuring_enabled": True})
    result = run_event_structuring(
        db_path, new_event_ids=new_ids, candidate_event_ids=[old_id]
    )
    assert result["model_errors"] == []
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM lock_probe").fetchone()[0] == 1


def test_schema_replaces_legacy_pending_mount_tables(tmp_path):
    from memory.maintenance.preprocessing import ensure_preprocessing_schema

    with sqlite3.connect(tmp_path / "legacy-pending.sqlite3") as con:
        con.executescript(
            """
            CREATE TABLE MemoryMounts (mount_id TEXT PRIMARY KEY);
            CREATE TABLE MemoryLocalStorylineMounts (proposal_id TEXT PRIMARY KEY);
            INSERT INTO MemoryMounts VALUES ('old');
            INSERT INTO MemoryLocalStorylineMounts VALUES ('old');
            """
        )
        ensure_preprocessing_schema(con)
        ensure_preprocessing_schema(con)
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "MemoryMounts" not in tables
        assert "MemoryLocalStorylineMounts" not in tables
        assert "MemoryCandidateStorylines" in tables
