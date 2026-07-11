import asyncio
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_memory_parser_contract():
    from memory.archive.parser import ArchiveParseFatalError, parse_archive_output

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


def test_cognition_sources_are_core_runtime_data():
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
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
        old_memory_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='MemoryCognitionSources'"
        ).fetchone()
    assert core_count == 1
    assert old_memory_table is None
    assert source_meta["1"]["source_uid"].startswith("cog_")
    assert repeated_source_meta["9"]["source_uid"] == source_meta["1"]["source_uid"]


def test_archiver_treats_empty_generation_as_provider_failure(caplog):
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {
        "memory": {
            "auto_archive": {"enabled": True},
            "embedding": {"provider": "hash", "dim": 32},
        }
    }

    class EmptyAdapter:
        def call_simple_text(self, *args, **kwargs):
            return None

    async def scenario():
        await database.init_db()
        app_state.archiver_adapter = EmptyAdapter()
        app_state.archive_tasks = set()
        from memory.archive import archiver

        sess_key = ("flow", f"empty_generation:{uuid.uuid4().hex}")
        await database.save_archive_signature(*sess_key, "old-sig")
        job_id = await database.enqueue_archive_job(
            conv_type=sess_key[0],
            conv_id=sess_key[1],
            conv_name="Empty generation",
            sender_id="",
            dialogue="<task></task>",
            signature="new-sig",
            prev_signature="old-sig",
            valid_candidate_ids=[],
        )
        await archiver._run_archive_job(
            {
                "job_id": job_id,
                "conv_type": sess_key[0],
                "conv_id": sess_key[1],
                "conv_name": "Empty generation",
                "sender_id": "",
                "dialogue": "<task></task>",
                "signature": "new-sig",
                "prev_signature": "old-sig",
                "valid_candidate_ids": [],
                "archive_mode": "cognition_flow_range",
            }
        )
        signatures = await database.load_archive_signatures()
        return await database.load_pending_archive_jobs(), signatures[sess_key], archiver._LAST_ARCHIVED_SIG[sess_key]

    with caplog.at_level(logging.WARNING, logger="AICQ.memory.archive.archiver"):
        pending_jobs, persisted_sig, cached_sig = asyncio.run(scenario())

    assert pending_jobs == []
    assert persisted_sig == "old-sig"
    assert cached_sig == "old-sig"
    assert not any("prompt 输出结构无效" in record.getMessage() for record in caplog.records)


def test_memory_event_sources_old_schema_migrates_before_uid_index():
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database.DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE MemoryEventSources (
                event_source_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                source_kind TEXT NOT NULL DEFAULT 'cognition',
                source_id TEXT NOT NULL,
                source_seq INTEGER,
                source_timestamp TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                UNIQUE(event_id, source_kind, source_id)
            );
            CREATE INDEX idx_memory_sources_event
                ON MemoryEventSources(event_id);
            CREATE INDEX idx_memory_sources_source
                ON MemoryEventSources(source_kind, source_id);
            """
        )

    from memory.repo import events

    events._SCHEMA_READY = False

    async def scenario():
        await events.ensure_schema()

    asyncio.run(scenario())

    with sqlite3.connect(database.DB_PATH) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(MemoryEventSources)").fetchall()
        }
        indexes = {
            row[1]
            for row in conn.execute("PRAGMA index_list(MemoryEventSources)").fetchall()
        }
    assert {"source_uid", "prompt_source_id"} <= columns
    assert "idx_memory_sources_uid" in indexes


def test_memory_storage_recall_and_render():
    import app_state
    import database

    tmp_dir = ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    database.DB_PATH = os.path.join(tmp_dir, "memory.sqlite3")
    app_state.config = {
        "memory": {
            "memory_predicate_similarity_threshold": 0.1,
            "memory_recall_max_results": 8,
            "memory_recall_recent_fallback": True,
            "embedding": {"provider": "hash", "dim": 64},
        }
    }

    from memory.recall.render import build_memory_debug_xml, build_memory_xml
    from memory.repo import events
    from consciousness.sources import upsert_cognition_sources

    events._SCHEMA_READY = False
    events._EMBED_CLIENT_KEY = None

    async def scenario():
        await events.ensure_schema()
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
        first_id = await events.write_prompt_event(
            first,
            conv_type="qq",
            conv_id="1",
            occurred_at=1710000000000,
            source_meta=source_meta_7,
        )
        duplicate_id = await events.write_prompt_event(
            first_repeat,
            conv_type="qq",
            conv_id="1",
            occurred_at=1710000000000,
            source_meta=source_meta_8,
        )
        await events.write_prompt_event(
            second, conv_type="qq", conv_id="1", occurred_at=1710000100000
        )

        backfill = await events.run_embedding_backfill(limit=20)
        recalled = await events.load_events_for_recall(
            sender_entity="Alice", context_scope="qq:1", query="likes tea", limit=5
        )
        recent_only = await events.load_events_for_recall(
            sender_entity="", context_scope="qq:1", query="", limit=5
        )
        return first_id, duplicate_id, backfill, recalled, recent_only

    first_id, duplicate_id, backfill, recalled, recent_only = asyncio.run(scenario())

    assert first_id == duplicate_id
    with sqlite3.connect(database.DB_PATH) as conn:
        vector_count = conn.execute("SELECT COUNT(*) FROM MemoryVectors").fetchone()[0]
        source_rows = conn.execute(
            """
            SELECT es.prompt_source_id, es.source_id, cs.source_uid
            FROM MemoryEventSources es
            JOIN CognitionSources cs ON cs.source_uid=es.source_uid
            WHERE es.event_id=?
            ORDER BY es.prompt_source_id
            """,
            (first_id,),
        ).fetchall()
    assert vector_count >= 2
    assert [row[0] for row in source_rows] == ["7", "8"]
    assert all(row[1] == row[2] and str(row[1]).startswith("cog_") for row in source_rows)
    with sqlite3.connect(database.DB_PATH) as conn:
        access_count, last_accessed = conn.execute(
            "SELECT access_count, last_accessed FROM MemoryEvents WHERE event_id=?",
            (first_id,),
        ).fetchone()
    assert access_count >= 1
    assert last_accessed > 0
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


def test_memory_supersedes_and_embedding_failure(tmp_path=None):
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {"memory": {"embedding": {"provider": "hash", "dim": 32}}}

    from memory.repo import events

    class FailingClient:
        model = "failing-test"
        model_version = "v1"

        def embed_texts(self, texts):
            raise RuntimeError("forced embedding failure")

    events._SCHEMA_READY = False
    events._EMBED_CLIENT_KEY = None
    original_embedding_client = events._embedding_client

    async def scenario():
        await events.ensure_schema()
        old_id = await events.write_prompt_event(
            {"summary": "Old value", "event_type": "state", "roles": []},
            conv_type="qq",
            conv_id="2",
            occurred_at=1710000000000,
        )
        events._embedding_client = lambda: FailingClient()
        new_id = await events.write_prompt_event(
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
        events._embedding_client = original_embedding_client
        events._EMBED_CLIENT_KEY = None

    with sqlite3.connect(database.DB_PATH) as conn:
        relation = conn.execute(
            "SELECT relation_type FROM MemoryRelations WHERE src_event_id=? AND dst_event_id=?",
            (new_id, old_id),
        ).fetchone()
        failed_jobs = conn.execute(
            "SELECT COUNT(*) FROM MemoryEmbeddingJobs WHERE status='failed' AND last_error<>''"
        ).fetchone()[0]
    assert relation == ("supersedes",)
    assert failed_jobs >= 1


def test_memory_hypothetical_not_recent_fallback():
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {
        "memory": {
            "memory_recall_recent_fallback": True,
            "memory_recall_max_results": 8,
            "embedding": {"provider": "hash", "dim": 32},
        }
    }

    from memory.repo import events

    events._SCHEMA_READY = False
    events._EMBED_CLIENT_KEY = None

    async def scenario():
        await events.ensure_schema()
        await events.write_prompt_event(
            {"summary": "Actual stable fact", "event_type": "state", "roles": [], "status": "actual"},
            conv_type="qq",
            conv_id="3",
            occurred_at=1710000000000,
        )
        await events.write_prompt_event(
            {"summary": "Hypothetical hidden branch", "event_type": "state", "roles": [], "status": "hypothetical"},
            conv_type="qq",
            conv_id="3",
            occurred_at=1710000100000,
        )
        return await events.load_events_for_recall(context_scope="qq:3", query="", limit=8)

    recalled = asyncio.run(scenario())
    summaries = {event["summary"] for event in recalled}
    assert "Actual stable fact" in summaries
    assert "Hypothetical hidden branch" not in summaries


def test_memory_web_graph_has_no_preset_account_or_group_nodes():
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {"memory": {"embedding": {"provider": "hash", "dim": 32}}}

    from quart import Quart
    from memory.repo import events
    from web.routes_memory import memory_graph

    events._SCHEMA_READY = False
    events._EMBED_CLIENT_KEY = None

    async def scenario():
        await events.ensure_schema()
        await events.write_prompt_event(
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
    from memory.archive import archiver

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


def test_cognition_flow_range_archive_job_writes_valid_events():
    import app_state
    import database

    database.DB_PATH = os.path.join(ROOT / "tmp" / f"memory-test-{uuid.uuid4().hex}", "memory.sqlite3")
    Path(database.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    app_state.config = {
        "memory": {
            "auto_archive": {"enabled": True},
            "embedding": {"provider": "hash", "dim": 32},
        }
    }

    class FakeAdapter:
        def call_simple_text(self, *args, **kwargs):
            return """
<extract>
<event>{"summary":"Alice likes jasmine tea.","source_id":"1/2/35","event_type":"like","status":"occurred","roles":[{"role":"agent","entity":"User:qq_1"},{"role":"theme","value_text":"jasmine tea"}]}</event>
<event>{"summary":"Bob likes oolong tea.","source_id":"35","event_type":"like","status":"occurred","roles":[{"role":"agent","entity":"User:qq_2"},{"role":"theme","value_text":"oolong tea"}]}</event>
<event>{"summary":"Carol likes puer tea.","source_id":"1","event_type":"like","status":"occurred","roles":[{"role":"agent","entity":"User:qq_3"},{"role":"theme","value_text":"puer tea"}]}</event>
<event>{"summary":"Dana likes sencha tea.","source_id":"2","event_type":"like","status":"occurred","roles":[{"role":"agent","entity":"User:qq_4"},{"role":"theme","value_text":"sencha tea"}]}</event>
</extract>
"""

    async def scenario():
        from memory.repo import events

        events._SCHEMA_READY = False
        events._EMBED_CLIENT_KEY = None
        await database.init_db()
        app_state.archiver_adapter = FakeAdapter()
        app_state.archive_tasks = set()
        from memory.archive import archiver

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
                "SELECT event_id, summary, event_type, status, source, conv_id FROM MemoryEvents ORDER BY summary"
            ).fetchall()
            sources = conn.execute(
                """
                SELECT e.summary, s.prompt_source_id, s.source_id, s.source_uid,
                       c.source_uid, c.cognition_text
                FROM MemoryEventSources s
                JOIN MemoryEvents e ON e.event_id=s.event_id
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
        ("Carol likes puer tea.", "like", "occurred", "cognition_flow_range", "cognition_flow_range:1-2"),
        ("Dana likes sencha tea.", "like", "occurred", "cognition_flow_range", "cognition_flow_range:1-2"),
    ]
    assert [(row[0], row[1]) for row in sources] == [
        ("Alice likes jasmine tea.", "1"),
        ("Alice likes jasmine tea.", "2"),
        ("Carol likes puer tea.", "1"),
        ("Dana likes sencha tea.", "2"),
    ]
    assert all(row[2] == row[3] == row[4] and str(row[2]).startswith("cog_") for row in sources)
    assert [row[5] for row in sources] == ["Alice", "Bob", "Alice", "Bob"]
    assert [(row[0], row[2]) for row in cognition_sources] == [("1", "Alice"), ("2", "Bob")]
    assert all(str(row[1]).startswith("cog_") for row in cognition_sources)


def test_archiver_existing_candidates_use_recalled_events_and_summary_sources():
    from memory.archive.archiver import _candidate_event_ids, _format_existing_candidates, _merge_existing_candidates

    recalled = [
        {
            "memory_kind": "summary",
            "summary_id": "local:abc",
            "summary": "华风身份信息故事线。",
            "source_event_ids": [11, 12],
            "core_entities": ["Person:华风"],
        },
        {
            "event_id": 13,
            "summary": "未來星織告知我华风的本名是公孙车。",
            "roles": [{"role": "agent", "entity": "Person:未來星織"}],
        },
    ]
    prefetch = [
        {
            "event_id": 13,
            "summary": "重复候选。",
            "roles": [],
        },
        {
            "event_id": 14,
            "summary": "华风问我我是谁。",
            "roles": [{"role": "agent", "entity": "Person:华风"}],
        },
    ]

    candidates = _merge_existing_candidates(recalled, prefetch)
    rendered = _format_existing_candidates(candidates)

    assert _candidate_event_ids(candidates) == [11, 12, 13, 14]
    assert rendered.count("#13") == 1
    assert "#local:abc" in rendered
    assert "source_events=11,12" in rendered
    assert "Person:华风" in rendered


def test_archive_turn_memories_passes_recalled_events_to_mount_candidates(monkeypatch):
    import app_state
    from memory.archive import archiver

    class FakeSession:
        conv_type = "group"
        conv_id = "100"
        conv_name = "Test Group"
        recalled_events = [
            {
                "memory_kind": "summary",
                "summary_id": "local:abc",
                "summary": "华风身份信息故事线。",
                "source_event_ids": [21, 22],
                "core_entities": ["Person:华风"],
            },
            {
                "event_id": 23,
                "summary": "未來星織告知我华风的本名是公孙车。",
                "roles": [{"role": "agent", "entity": "Person:未來星織"}],
            },
        ]

        def __init__(self):
            self.context_messages = [
                {"role": "user", "content": "华风现在叫华车", "message_id": "m1"},
            ]

        def get_chat_log_display(self):
            return "未來星織: 华风现在叫华车"

    captured: dict[str, object] = {}

    async def fake_prefetch(*args, **kwargs):
        return []

    async def fake_enqueue_archive_job(**kwargs):
        captured["enqueue"] = kwargs
        return 777

    async def fake_run_archive_job(payload):
        captured["payload"] = payload

    async def fake_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(app_state, "config", {"memory": {"auto_archive": {"enabled": True}}})
    monkeypatch.setattr(app_state, "archiver_adapter", object())
    monkeypatch.setattr(archiver, "_ensure_sig_loaded", fake_noop)
    monkeypatch.setattr(archiver, "_persist_signature", fake_noop)
    monkeypatch.setattr(archiver, "_run_archive_job", fake_run_archive_job)
    monkeypatch.setattr(archiver, "_LAST_ARCHIVED_SIG", {})
    monkeypatch.setattr("memory.repo.events.prefetch_candidates_for_archiver", fake_prefetch)
    monkeypatch.setattr("database.enqueue_archive_job", fake_enqueue_archive_job)

    asyncio.run(archiver.archive_turn_memories(FakeSession(), "42", []))

    enqueue = captured["enqueue"]
    payload = captured["payload"]
    assert enqueue["valid_candidate_ids"] == [21, 22, 23]
    assert payload["valid_candidate_ids"] == [21, 22, 23]
    assert "<existing_candidates>" in enqueue["dialogue"]
    assert "source_events=21,22" in enqueue["dialogue"]


def test_cognition_flow_range_archive_passes_round_memory_candidates(monkeypatch):
    import app_state
    from memory.archive import archiver

    class Round:
        seq = 9
        timestamp = 123.0
        cognition = "华风现在叫华车这件事需要和旧身份记忆挂载。"
        raw_response = "<cognition>华风现在叫华车。</cognition>"
        calls = []
        responses = []
        memory_candidates = [
            {
                "memory_kind": "summary",
                "summary_id": "local:abc",
                "summary": "华风身份信息故事线。",
                "source_event_ids": [31, 32],
                "core_entities": ["Person:华风"],
            },
            {
                "event_id": 33,
                "summary": "未來星織告知我华风的本名是公孙车。",
                "roles": [{"role": "agent", "entity": "Person:未來星織"}],
            },
        ]

    captured: dict[str, object] = {}

    async def fake_enqueue_archive_job(**kwargs):
        captured["enqueue"] = kwargs
        return 778

    async def fake_run_archive_job(payload):
        captured["payload"] = payload

    async def fake_noop(*args, **kwargs):
        return None

    monkeypatch.setattr(app_state, "config", {"memory": {"auto_archive": {"enabled": True}}})
    monkeypatch.setattr(app_state, "archiver_adapter", object())
    monkeypatch.setattr(archiver, "_ensure_sig_loaded", fake_noop)
    monkeypatch.setattr(archiver, "_persist_signature", fake_noop)
    monkeypatch.setattr(archiver, "_run_archive_job", fake_run_archive_job)
    monkeypatch.setattr(archiver, "_LAST_ARCHIVED_SIG", {})
    monkeypatch.setattr("database.enqueue_archive_job", fake_enqueue_archive_job)

    asyncio.run(
        archiver.archive_cognition_flow_range(
            [Round()],
            coverage_start_seq=9,
            coverage_end_seq=9,
        )
    )

    enqueue = captured["enqueue"]
    payload = captured["payload"]
    assert enqueue["valid_candidate_ids"] == [31, 32, 33]
    assert payload["valid_candidate_ids"] == [31, 32, 33]
    assert "<existing_candidates>" in enqueue["dialogue"]
    assert "source_events=31,32" in enqueue["dialogue"]
    assert "华风身份信息故事线" in enqueue["dialogue"]
