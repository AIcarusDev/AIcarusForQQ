import asyncio
import os
import sqlite3
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_memory_v2_parser_contract():
    from memory.parser_v2 import ArchiveParseFatalError, parse_archive_output

    parsed = parse_archive_output(
        '<analysis>ignore</analysis><extract><event>{"summary":"A likes tea","source_id":"1","event_type":"likes","roles":[]}</event>'
        '<event>```json\n{"summary":"bad","source_id":"1","event_type":"x","roles":[]}\n```</event></extract>'
    )
    assert len(parsed.events) == 1
    assert parsed.errors
    missing_source = parse_archive_output(
        '<extract><event>{"summary":"A likes tea","event_type":"likes","roles":[]}</event></extract>'
    )
    assert not missing_source.events
    assert any("source_id" in err for err in missing_source.errors)
    empty_source = parse_archive_output(
        '<extract><event>{"summary":"A likes tea","source_id":"","event_type":"likes","roles":[]}</event></extract>'
    )
    assert len(empty_source.events) == 1

    try:
        parse_archive_output("<extract></extract><extract></extract>")
    except ArchiveParseFatalError:
        pass
    else:
        raise AssertionError("duplicated extract should be fatal")

    try:
        parse_archive_output("<analysis><extract></extract></analysis>")
    except ArchiveParseFatalError:
        pass
    else:
        raise AssertionError("nested extract should not count as top-level")


def test_cognition_sources_are_core_not_v2():
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-v2-test-{uuid.uuid4().hex}", "memory_v2.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    async def scenario():
        from consciousness.sources import upsert_cognition_sources

        first = await upsert_cognition_sources(
            {"1": {"timestamp": "2026-06-20T00:00:00+08:00", "text": "Core cognition source"}},
            origin_type="test",
            origin_id="core",
        )
        second = await upsert_cognition_sources(
            {"9": {"timestamp": "2026-06-20T00:00:00+08:00", "text": "Core cognition source"}},
            origin_type="test",
            origin_id="core",
        )
        return first, second

    source_meta, repeated_source_meta = asyncio.run(scenario())
    with sqlite3.connect(database.DB_PATH) as conn:
        core_count = conn.execute("SELECT COUNT(*) FROM CognitionSources").fetchone()[0]
        old_v2_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='MemoryV2CognitionSources'"
        ).fetchone()
    assert core_count == 1
    assert old_v2_table is None
    assert source_meta["1"]["source_uid"].startswith("cog_")
    assert repeated_source_meta["9"]["source_uid"] == source_meta["1"]["source_uid"]


def test_memory_v2_storage_recall_and_render():
    import app_state
    import database

    tmp_dir = ROOT / "tmp" / f"memory-v2-test-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    database.DB_PATH = os.path.join(tmp_dir, "memory_v2.sqlite3")
    app_state.config = {
        "memory": {
            "v2": {
                "memory_predicate_similarity_threshold": 0.1,
                "memory_recall_max_results": 8,
                "memory_recall_recent_fallback": True,
                "embedding": {"provider": "hash", "dim": 64},
            }
        }
    }

    from memory.render import build_memory_debug_xml, build_memory_xml
    from memory.repo import events_v2
    from consciousness.sources import upsert_cognition_sources

    events_v2._SCHEMA_READY = False
    events_v2._EMBED_CLIENT_KEY = None

    async def scenario():
        await events_v2.ensure_schema()
        source_meta_7 = await upsert_cognition_sources(
            {"7": {"timestamp": "2026-06-20T00:00:00+08:00", "text": "Alice likes green tea"}},
            origin_type="test",
            origin_id="storage",
        )
        source_meta_8 = await upsert_cognition_sources(
            {"8": {"timestamp": "2026-06-20T00:01:00+08:00", "text": "Alice still likes green tea"}},
            origin_type="test",
            origin_id="storage",
        )
        first = {
            "summary": "Alice likes green tea",
            "source_id": "7",
            "event_type": "likes",
            "roles": [
                {"role": "subject", "entity": "Alice"},
                {"role": "object", "value": "green tea"},
            ],
        }
        first_repeat = dict(first)
        first_repeat["source_id"] = "8"
        second = {
            "summary": "Alice prefers matcha",
            "source_id": "9",
            "event_type": "prefers",
            "roles": [
                {"role": "subject", "entity": "Alice"},
                {"role": "object", "value": "matcha"},
            ],
        }
        first_id = await events_v2.write_prompt_event(
            first,
            conv_type="qq",
            conv_id="1",
            occurred_at=1710000000000,
            source_meta=source_meta_7,
        )
        duplicate_id = await events_v2.write_prompt_event(
            first_repeat,
            conv_type="qq",
            conv_id="1",
            occurred_at=1710000000000,
            source_meta=source_meta_8,
        )
        await events_v2.write_prompt_event(
            second, conv_type="qq", conv_id="1", occurred_at=1710000100000
        )

        backfill = await events_v2.run_embedding_backfill(limit=20)
        recalled = await events_v2.load_events_for_recall(
            sender_entity="Alice", context_scope="qq:1", query="likes tea", limit=5
        )
        recent_only = await events_v2.load_events_for_recall(
            sender_entity="", context_scope="qq:1", query="", limit=5
        )
        return first_id, duplicate_id, backfill, recalled, recent_only

    first_id, duplicate_id, backfill, recalled, recent_only = asyncio.run(scenario())

    assert first_id == duplicate_id
    with sqlite3.connect(database.DB_PATH) as conn:
        vector_count = conn.execute("SELECT COUNT(*) FROM MemoryV2Vectors").fetchone()[0]
        source_rows = conn.execute(
            """
            SELECT es.prompt_source_id, es.source_id, cs.source_uid
            FROM MemoryV2EventSources es
            JOIN CognitionSources cs ON cs.source_uid=es.source_uid
            WHERE es.event_id=?
            ORDER BY es.prompt_source_id
            """,
            (first_id,),
        ).fetchall()
    assert vector_count >= 2
    assert [row[0] for row in source_rows] == ["7", "8"]
    assert all(row[1] == row[2] and str(row[1]).startswith("cog_") for row in source_rows)
    assert {"queued", "processed", "ready", "failed"} <= set(backfill)
    assert recalled
    assert recent_only
    assert any("recall_path" in event for event in recalled)

    normal_xml = build_memory_xml(recalled_events=recalled)
    debug_xml = build_memory_debug_xml(recalled_events=recalled)
    assert "Alice" in normal_xml
    assert "confidence=" in normal_xml
    assert "event_type" not in normal_xml
    assert "recall_score" not in normal_xml
    assert "<memory_debug" in debug_xml
    assert "score=" in debug_xml


def test_memory_v2_supersedes_and_embedding_failure(tmp_path=None):
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-v2-test-{uuid.uuid4().hex}", "memory_v2.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {"memory": {"v2": {"embedding": {"provider": "hash", "dim": 32}}}}

    from memory.repo import events_v2

    class FailingClient:
        model = "failing-test"
        model_version = "v1"

        def embed_texts(self, texts):
            raise RuntimeError("forced embedding failure")

    events_v2._SCHEMA_READY = False
    events_v2._EMBED_CLIENT_KEY = None
    original_embedding_client = events_v2._embedding_client

    async def scenario():
        await events_v2.ensure_schema()
        old_id = await events_v2.write_prompt_event(
            {"summary": "Old value", "event_type": "state", "roles": []},
            conv_type="qq",
            conv_id="2",
            occurred_at=1710000000000,
        )
        events_v2._embedding_client = lambda: FailingClient()
        new_id = await events_v2.write_prompt_event(
            {"summary": "New value", "event_type": "state", "roles": []},
            conv_type="qq",
            conv_id="2",
            occurred_at=1710000100000,
            supersedes=old_id,
        )
        return old_id, new_id

    try:
        old_id, new_id = asyncio.run(scenario())
    finally:
        events_v2._embedding_client = original_embedding_client
        events_v2._EMBED_CLIENT_KEY = None

    with sqlite3.connect(database.DB_PATH) as conn:
        relation = conn.execute(
            "SELECT relation_type FROM MemoryV2Relations WHERE src_event_id=? AND dst_event_id=?",
            (new_id, old_id),
        ).fetchone()
        failed_jobs = conn.execute(
            "SELECT COUNT(*) FROM MemoryV2EmbeddingJobs WHERE status='failed' AND last_error<>''"
        ).fetchone()[0]
    assert relation == ("supersedes",)
    assert failed_jobs >= 1


def test_memory_v2_hypothetical_not_recent_fallback():
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-v2-test-{uuid.uuid4().hex}", "memory_v2.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {
        "memory": {
            "v2": {
                "memory_recall_recent_fallback": True,
                "memory_recall_max_results": 8,
                "embedding": {"provider": "hash", "dim": 32},
            }
        }
    }

    from memory.repo import events_v2

    events_v2._SCHEMA_READY = False
    events_v2._EMBED_CLIENT_KEY = None

    async def scenario():
        await events_v2.ensure_schema()
        await events_v2.write_prompt_event(
            {"summary": "Actual stable fact", "event_type": "state", "roles": [], "status": "actual"},
            conv_type="qq",
            conv_id="3",
            occurred_at=1710000000000,
        )
        await events_v2.write_prompt_event(
            {"summary": "Hypothetical hidden branch", "event_type": "state", "roles": [], "status": "hypothetical"},
            conv_type="qq",
            conv_id="3",
            occurred_at=1710000100000,
        )
        return await events_v2.load_events_for_recall(context_scope="qq:3", query="", limit=8)

    recalled = asyncio.run(scenario())
    summaries = {event["summary"] for event in recalled}
    assert "Actual stable fact" in summaries
    assert "Hypothetical hidden branch" not in summaries


def test_memory_v2_web_graph_has_no_preset_account_or_group_nodes():
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-v2-test-{uuid.uuid4().hex}", "memory_v2.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {"memory": {"v2": {"embedding": {"provider": "hash", "dim": 32}}}}

    from quart import Quart
    from memory.repo import events_v2
    from web.routes_memory import memory_graph

    events_v2._SCHEMA_READY = False
    events_v2._EMBED_CLIENT_KEY = None

    async def scenario():
        await events_v2.ensure_schema()
        await events_v2.write_prompt_event(
            {
                "summary": "Alice likes tea",
                "event_type": "likes",
                "roles": [{"role": "subject", "entity": "User:qq_1"}],
            },
            conv_type="qq",
            conv_id="1",
            occurred_at=1710000000000,
        )
        app = Quart(__name__)
        async with app.test_request_context("/memory/graph"):
            response = await memory_graph()
            return await response.get_json()

    data = asyncio.run(scenario())
    groups = {node["group"] for node in data["nodes"]}
    assert "event" in groups
    assert "predicate" in groups
    assert "participant" in groups
    assert "account" not in groups
    assert "group" not in groups
    assert "session" not in groups
    assert "self" not in groups


def test_cognition_flow_range_archive_uses_raw_range_not_summary():
    from memory import archiver

    class Call:
        name = "notes.write"
        call_id = "c1"
        args = {"text": "Alice prefers jasmine tea"}

    class Response:
        name = "notes.write"
        call_id = "c1"
        response = {"ok": True}

    class Round:
        seq = 7
        timestamp = 123.0
        cognition = "Alice said she prefers jasmine tea."
        raw_response = "<cognition>Alice prefers jasmine tea.</cognition>"
        calls = [Call()]
        responses = [Response()]

    task_xml = archiver._format_cognition_flow_task_xml(
        [Round()],
        coverage_start_seq=7,
        coverage_end_seq=7,
    )

    assert '<cognition id="1"' in task_xml
    assert "Alice said she prefers jasmine tea." in task_xml
    assert "notes.write" not in task_xml
    assert "<summary" not in task_xml
    assert archiver._extract_cognition_source_map(task_xml) == {
        "1": {
            "timestamp": "1970-01-01T00:02:03+00:00",
            "text": "Alice said she prefers jasmine tea.",
        }
    }
    assert archiver.schedule_archive(None, "", []) is None
    assert archiver.schedule_compression_archive("legacy summary", 7) is None


def test_cognition_flow_range_archive_job_writes_valid_events():
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-v2-test-{uuid.uuid4().hex}", "memory_v2.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {
        "memory": {
            "auto_archive": {"enabled": True, "max_per_turn": 3},
            "v2": {"embedding": {"provider": "hash", "dim": 32}},
        }
    }

    class FakeAdapter:
        def call_simple_text(self, *args, **kwargs):
            return """
<extract>
<event>{"summary":"Alice likes jasmine tea.","source_id":"1/2/35","event_type":"like","status":"occurred","roles":[{"role":"agent","entity":"User:qq_1"},{"role":"theme","value_text":"jasmine tea"}]}</event>
<event>{"summary":"Bob likes oolong tea.","source_id":"35","event_type":"like","status":"occurred","roles":[{"role":"agent","entity":"User:qq_2"},{"role":"theme","value_text":"oolong tea"}]}</event>
</extract>
"""

    async def scenario():
        from memory.repo import events_v2

        events_v2._SCHEMA_READY = False
        events_v2._EMBED_CLIENT_KEY = None
        await database.init_db()
        app_state.archiver_adapter = FakeAdapter()
        app_state.archive_tasks = set()
        from memory import archiver

        await archiver._run_archive_job(
            {
                "job_id": 999,
                "conv_type": "flow",
                "conv_id": "cognition_flow_range:1-2",
                "conv_name": "Cognition flow range 1-2",
                "sender_id": "",
                "dialogue": (
                    '<task><cognition id="1" timestamp="2026-06-20T00:00:00+08:00">Alice</cognition>'
                    '<cognition id="2" timestamp="2026-06-20T00:01:00+08:00">Bob</cognition></task>'
                ),
                "signature": "test-sig",
                "prev_signature": "",
                "valid_candidate_ids": [],
                "archive_mode": "cognition_flow_range",
            }
        )
        import sqlite3

        with sqlite3.connect(database.DB_PATH) as conn:
            events = conn.execute(
                "SELECT event_id, summary, event_type, status, source, conv_id FROM MemoryV2Events ORDER BY summary"
            ).fetchall()
            sources = conn.execute(
                """
                SELECT e.summary, s.prompt_source_id, s.source_id, s.source_uid,
                       c.source_uid, c.cognition_text
                FROM MemoryV2EventSources s
                JOIN MemoryV2Events e ON e.event_id=s.event_id
                JOIN CognitionSources c ON c.source_uid=s.source_uid
                ORDER BY e.summary, s.prompt_source_id
                """
            ).fetchall()
            cognition_sources = conn.execute(
                "SELECT prompt_source_id, source_uid, cognition_text FROM CognitionSources ORDER BY prompt_source_id"
            ).fetchall()
            return events, sources, cognition_sources

    rows, sources, cognition_sources = asyncio.run(scenario())
    assert [row[1:] for row in rows] == [
        ("Alice likes jasmine tea.", "like", "occurred", "cognition_flow_range", "cognition_flow_range:1-2"),
        ("Bob likes oolong tea.", "like", "occurred", "cognition_flow_range", "cognition_flow_range:1-2"),
    ]
    assert [(row[0], row[1]) for row in sources] == [
        ("Alice likes jasmine tea.", "1"),
        ("Alice likes jasmine tea.", "2"),
    ]
    assert all(row[2] == row[3] == row[4] and str(row[2]).startswith("cog_") for row in sources)
    assert [row[5] for row in sources] == ["Alice", "Bob"]
    assert [(row[0], row[2]) for row in cognition_sources] == [("1", "Alice"), ("2", "Bob")]
    assert all(str(row[1]).startswith("cog_") for row in cognition_sources)
