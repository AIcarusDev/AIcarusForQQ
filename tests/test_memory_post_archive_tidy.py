from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest


def _fresh_db(tmp_path, name="post-archive-tidy"):
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


def test_tidy_user_payload_exposes_only_local_id_summary_and_entities():
    from memory.post_archive.tidy_workflow import EventAtom, build_tidy_user_payload

    payload = json.loads(
        build_tidy_user_payload(
            [EventAtom(101, "new", ("self",))],
            [EventAtom(202, "old", ("Person:吹雪",))],
        )
    )

    assert payload == {
        "new_events": [{"id": "N1", "summary": "new", "entities": ["self"]}],
        "existing_events": [{"id": "H1", "summary": "old", "entities": ["Person:吹雪"]}],
    }
    serialized = json.dumps(payload)
    for forbidden in ("local_id", "event_id", "event_type", "status", "occurred_at", "anchors"):
        assert forbidden not in serialized


def test_parse_tidy_response_accepts_empty_blocks_and_deduplicates():
    from memory.post_archive.tidy_workflow import parse_tidy_response

    empty = parse_tidy_response(
        "<analysis>none</analysis><tidy><link></link><candidate></candidate></tidy>",
        new_ids={"N1", "N2"},
        historical_ids={"H1"},
    )
    assert empty.links == ()
    assert empty.candidates == ()
    assert empty.errors == ()

    result = parse_tidy_response(
        """<analysis>plan</analysis><tidy>
        <link>[{"new_event":"N1","existing_event":"H1"},{"new_event":"N3","existing_event":"H2"},{"new_event":"N1","existing_event":"H1"}]</link>
        <candidate>[["N2","N1","N1"],["N4","N3"],["N1","N2"]]</candidate>
        </tidy>""",
        new_ids={"N1", "N2", "N3", "N4"},
        historical_ids={"H1", "H2"},
    )
    assert result.links == (("N1", "H1"), ("N3", "H2"))
    assert result.candidates == (("N1", "N2"), ("N3", "N4"))
    assert result.errors == ()


@pytest.mark.parametrize(
    ("raw", "error_fragment"),
    [
        ("<tidy><link></link><candidate></candidate></tidy>", "exactly match"),
        ("<analysis>x</analysis><tidy><candidate></candidate><link></link></tidy>", "exactly match"),
        ("<analysis>x</analysis><tidy><link>{}</link><candidate></candidate></tidy>", "JSON array"),
        ("<analysis>x</analysis><tidy><link>[</link><candidate></candidate></tidy>", "invalid JSON"),
        (
            '<analysis>x</analysis><tidy><link>[{"new_event":"H1","existing_event":"H1"}]</link><candidate></candidate></tidy>',
            "unknown new_event",
        ),
        (
            '<analysis>x</analysis><tidy><link>[{"new_event":"N1","existing_event":"N1"}]</link><candidate></candidate></tidy>',
            "unknown existing_event",
        ),
        (
            '<analysis>x</analysis><tidy><link>[{"new_event":"N1","existing_event":"H1","extra":1}]</link><candidate></candidate></tidy>',
            "exactly new_event",
        ),
        (
            '<analysis>x</analysis><tidy><link></link><candidate>[["N1"]]</candidate></tidy>',
            "at least two",
        ),
        (
            '<analysis>x</analysis><tidy><link></link><candidate>[["N1","H1"]]</candidate></tidy>',
            "unknown new event id",
        ),
    ],
)
def test_parse_tidy_response_rejects_invalid_contract(raw, error_fragment):
    from memory.post_archive.tidy_workflow import parse_tidy_response

    result = parse_tidy_response(raw, new_ids={"N1", "N2"}, historical_ids={"H1"})
    assert any(error_fragment in error for error in result.errors)


def test_tidy_workflow_writes_idempotent_relation_and_episode_candidate(tmp_path, monkeypatch):
    import app_state
    from memory.post_archive.tidy_workflow import run_post_archive_tidy_workflow
    from memory.sleep.consolidation import run_episode_candidate_consolidation

    db_path = _fresh_db(tmp_path)
    old_id, new_ids = _write_events(db_path)

    class Adapter:
        calls = []

        def call_simple_text(self, system_prompt, user_content, gen, log_tag):
            self.calls.append((system_prompt, user_content, gen, log_tag))
            return (
                '<analysis>plan</analysis><tidy>'
                '<link>[{"new_event":"N1","existing_event":"H1"}]</link>'
                '<candidate>[["N1","N2"]]</candidate></tidy>'
            )

    adapter = Adapter()
    monkeypatch.setattr(app_state, "memory_consolidation_adapter", adapter)
    monkeypatch.setattr(
        app_state,
        "memory_consolidation_cfg",
        {"enabled": True, "llm_tidy_enabled": True, "generation": {}},
    )

    first = run_post_archive_tidy_workflow(
        db_path, new_event_ids=new_ids, candidate_event_ids=[old_id], now_ms=10
    )
    second = run_post_archive_tidy_workflow(
        db_path, new_event_ids=new_ids, candidate_event_ids=[old_id], now_ms=11
    )
    assert first["links_written"] == 1
    assert second["links_written"] == 0
    assert first["episode_candidates_staged"] == 1
    assert adapter.calls[0][3] == "memory_consolidation/tidy"

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT src_event_id, dst_event_id, relation_type FROM MemoryRelations"
        ).fetchall() == [(new_ids[0], old_id, "related")]
        assert json.loads(
            con.execute("SELECT event_ids_json FROM MemoryEpisodeCandidates").fetchone()[0]
        ) == new_ids

        report = run_episode_candidate_consolidation(
            con, max_candidates=10, dry_run=False, solidify=True
        )
        con.commit()
        assert report.clusters_written == 1
        assert report.cluster_members_written == 2
        assert con.execute("SELECT status FROM MemoryEpisodeCandidates").fetchone()[0] == "accepted"
        member_ids = {
            row[0]
            for row in con.execute(
                "SELECT event_id FROM MemoryClusterMembers WHERE cluster_id=?",
                (report.cluster_ids_written[0],),
            )
        }
        assert member_ids == set(new_ids)


def test_tidy_workflow_does_not_hold_write_lock_during_model_call(tmp_path, monkeypatch):
    import app_state
    from memory.post_archive.tidy_workflow import run_post_archive_tidy_workflow

    db_path = _fresh_db(tmp_path, "tidy-no-lock")
    old_id, new_ids = _write_events(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE lock_probe (value INTEGER)")

    class Adapter:
        def call_simple_text(self, *_args, **_kwargs):
            with sqlite3.connect(db_path, timeout=0.2) as other:
                other.execute("INSERT INTO lock_probe(value) VALUES (1)")
            return "<analysis>none</analysis><tidy><link></link><candidate></candidate></tidy>"

    monkeypatch.setattr(app_state, "memory_consolidation_adapter", Adapter())
    monkeypatch.setattr(app_state, "memory_consolidation_cfg", {"enabled": True, "llm_tidy_enabled": True})
    result = run_post_archive_tidy_workflow(
        db_path, new_event_ids=new_ids, candidate_event_ids=[old_id]
    )
    assert result["model_errors"] == []
    with sqlite3.connect(db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM lock_probe").fetchone()[0] == 1


def test_schema_replaces_legacy_pending_mount_tables(tmp_path):
    from memory.sleep.consolidation import ensure_preprocessing_schema

    with sqlite3.connect(tmp_path / "legacy-pending.sqlite3") as con:
        con.executescript(
            """
            CREATE TABLE MemoryMounts (mount_id TEXT PRIMARY KEY);
            CREATE TABLE MemoryLocalClusterMounts (proposal_id TEXT PRIMARY KEY);
            INSERT INTO MemoryMounts VALUES ('old');
            INSERT INTO MemoryLocalClusterMounts VALUES ('old');
            """
        )
        ensure_preprocessing_schema(con)
        ensure_preprocessing_schema(con)
        tables = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "MemoryMounts" not in tables
        assert "MemoryLocalClusterMounts" not in tables
        assert "MemoryEpisodeCandidates" in tables
