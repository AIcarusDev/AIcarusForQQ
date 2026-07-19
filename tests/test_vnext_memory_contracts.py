from __future__ import annotations

import asyncio
import hashlib
import sqlite3

import pytest
from quart import Quart

from memory.maintenance.preprocessing import ensure_preprocessing_schema
from memory.semantic_query import (
    LANGUAGE_VERSION,
    MemoryQLValidationError,
    MemoryQueryTimeout,
    MemoryQueryUnavailable,
    SemanticMemoryService,
)
from web import routes_ui_v1


def _create_memory_database(path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE MemoryEvents (
                event_id INTEGER PRIMARY KEY,
                summary TEXT NOT NULL,
                event_type_norm TEXT NOT NULL,
                occurred_at INTEGER NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                is_deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE MemoryCanonicalEntities (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE MemoryEntityMentions (
                event_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                raw_entity TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE MemoryStorylines (
                storyline_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                origin_type TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL,
                member_count INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );
            CREATE TABLE MemoryStorylineMembers (
                storyline_id TEXT NOT NULL,
                event_id INTEGER NOT NULL,
                score REAL NOT NULL,
                rank INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE MemoryEventSources (
                event_source_id INTEGER PRIMARY KEY,
                event_id INTEGER NOT NULL,
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_seq INTEGER,
                source_timestamp TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE MemoryEventRelations (
                relation_id TEXT PRIMARY KEY,
                source_event_id INTEGER NOT NULL,
                target_event_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                revision INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );

            INSERT INTO MemoryEvents VALUES
                (1, '管理员完成配置热重载', 'configuration', 1784421000000, 0.96, 'actual', 0),
                (2, 'AIcarus 完成记忆维护', 'maintenance', 1784422000000, 0.93, 'actual', 0),
                (3, '管理员检查记忆关系', 'inspection', 1784423000000, 0.88, 'actual', 0),
                (4, '已删除的事件', 'deleted', 1784424000000, 0.99, 'actual', 1);
            INSERT INTO MemoryCanonicalEntities VALUES
                ('entity-admin', 'admin', 'person', 0.95, 'active', 1784423000000),
                ('entity-aicarus', 'AIcarus', 'system', 0.98, 'active', 1784423000000),
                ('entity-retired', 'retired', 'person', 0.80, 'retired', 1784423000000);
            INSERT INTO MemoryEntityMentions VALUES
                (1, 'actor', '管理员', 'entity-admin', 0.94, '{}'),
                (2, 'subject', 'AIcarus', 'entity-aicarus', 0.98, '{}'),
                (3, 'actor', '管理员', 'entity-admin', 0.91, '{}'),
                (4, 'subject', 'AIcarus', 'entity-aicarus', 0.99, '{}');
            INSERT INTO MemoryStorylines VALUES
                ('story-vnext', 'WebUI vNext', 'algorithmic_storyline', 'active', 0.91, 2, 1784423000000);
            INSERT INTO MemoryStorylineMembers VALUES
                ('story-vnext', 1, 0.92, 1, 'active'),
                ('story-vnext', 2, 0.89, 2, 'active');
            INSERT INTO MemoryEventSources VALUES
                (1, 1, 'core_chat', 'session-1', 1, '2026-07-19T08:00:00Z', 1784421000000),
                (2, 2, 'cognition', 'flow-2', 2, '2026-07-19T08:10:00Z', 1784422000000);
            INSERT INTO MemoryEventRelations VALUES
                ('rel-1', 1, 2, 'causes', 0.86, 'active', 1, 1784423000000),
                ('rel-2', 2, 3, 'followed_by', 0.83, 'active', 1, 1784424000000);
            """
        )


@pytest.fixture()
def memory_db(tmp_path):
    path = tmp_path / "memory.sqlite3"
    _create_memory_database(path)
    return path


def _involves_query(*, node_limit: int = 80, edge_limit: int = 120) -> str:
    return f'''MATCH
  $event ISA Event
  $entity ISA CanonicalEntity
  ($event)-[INVOLVES]->($entity)
WHERE $entity.confidence >= 0.90 AND $event.status = "actual"
RETURN GRAPH
LIMIT NODES {node_limit} EDGES {edge_limit}'''


def test_semantic_schema_reports_counts_and_availability(memory_db) -> None:
    schema = SemanticMemoryService(memory_db).schema()
    types = {item["name"]: item for item in schema["types"]}
    relations = {item["name"]: item for item in schema["relations"]}

    assert schema["language"] == {
        "name": "MemoryQL",
        "version": LANGUAGE_VERSION,
        "read_only": True,
        "clauses": ["MATCH", "WHERE", "EXPAND", "RETURN", "LIMIT"],
    }
    assert schema["compatibility"]["status"] == "compatible"
    assert types["Event"]["count"] == 3
    assert types["CanonicalEntity"]["count"] == 2
    storyline_summary = next(
        prop for prop in types["Storyline"]["properties"] if prop["name"] == "summary"
    )
    assert storyline_summary == {
        "name": "summary",
        "type": "string",
        "operators": [],
        "projected_only": True,
    }
    assert relations["INVOLVES"]["count"] == 3
    assert all(item["available"] for item in [*types.values(), *relations.values()])


def test_semantic_relation_query_is_bounded_parameterized_and_isolated(memory_db) -> None:
    service = SemanticMemoryService(memory_db)
    first = service.query(_involves_query(), language_version=LANGUAGE_VERSION)
    second = service.query(_involves_query(), language_version=LANGUAGE_VERSION)

    assert first["query_id"] != second["query_id"]
    assert first["provenance"]["read_only"] is True
    assert first["provenance"]["isolation"] == "per_query"
    assert first["budget"]["consumed"]["nodes"] == 5
    assert first["budget"]["consumed"]["edges"] == 3
    assert {node["id"] for node in first["nodes"]} == {
        "event:1",
        "event:2",
        "event:3",
        "entity:entity-admin",
        "entity:entity-aicarus",
    }
    assert len(first["table"]["rows"]) == 3


def test_storyline_projection_tolerates_missing_optional_summary_cache(memory_db) -> None:
    result = SemanticMemoryService(memory_db).query(
        """MATCH
  $storyline ISA Storyline
RETURN GRAPH
LIMIT NODES 10 EDGES 10""",
        language_version=LANGUAGE_VERSION,
    )

    assert result["nodes"][0]["properties"]["summary"] == ""


def test_storyline_projection_includes_latest_ready_summary(memory_db) -> None:
    with sqlite3.connect(memory_db) as connection:
        connection.executescript(
            """
            CREATE TABLE MemoryStorylineSummaryTasks (
                task_id TEXT PRIMARY KEY,
                storyline_id TEXT NOT NULL
            );
            CREATE TABLE MemorySummaryCache (
                summary_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );
            INSERT INTO MemoryStorylineSummaryTasks VALUES
                ('task-story-vnext', 'story-vnext');
            INSERT INTO MemorySummaryCache VALUES
                ('summary-old', 'task-story-vnext', 'stale', '旧摘要', 10),
                ('summary-ready', 'task-story-vnext', 'ready', '配置热重载后，系统完成记忆维护与关系检查。', 20);
            """
        )

    result = SemanticMemoryService(memory_db).query(
        """MATCH
  $storyline ISA Storyline
RETURN GRAPH
LIMIT NODES 10 EDGES 10""",
        language_version=LANGUAGE_VERSION,
    )

    assert result["nodes"][0]["properties"]["summary"] == "配置热重载后，系统完成记忆维护与关系检查。"


def test_recent_relation_index_is_part_of_the_idempotent_schema(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "schema.sqlite3") as connection:
        ensure_preprocessing_schema(connection)
        ensure_preprocessing_schema(connection)
        index_columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info('idx_MemoryEventRelations_recent')"
            )
        ]

    assert index_columns == ["status", "updated_at_ms", "relation_id"]


def test_recent_relation_query_bounds_ids_before_projecting_event_payloads(
    memory_db,
    monkeypatch,
) -> None:
    from memory import semantic_query

    captured_sql: list[str] = []
    original_query_rows = semantic_query._query_rows

    def capture_query(connection, sql, params):
        captured_sql.append(sql)
        return original_query_rows(connection, sql, params)

    monkeypatch.setattr(semantic_query, "_query_rows", capture_query)
    SemanticMemoryService(memory_db).query(
        '''MATCH
  $source ISA Event
  $target ISA Event
  ($source)-[RELATES_TO]->($target)
RETURN GRAPH
LIMIT NODES 20 EDGES 10''',
        language_version=LANGUAGE_VERSION,
    )

    relation_sql = next(
        sql for sql in captured_sql if "FROM MemoryEventRelations" in sql
    )
    bounded_relation_end = relation_sql.index("LIMIT ?")
    payload_projection_start = relation_sql.index(
        "JOIN MemoryEvents a ON a.event_id=r.source_event_id"
    )
    assert bounded_relation_end < payload_projection_start


def test_memoryql_rejects_writes_without_changing_database(memory_db) -> None:
    service = SemanticMemoryService(memory_db)
    before = hashlib.sha256(memory_db.read_bytes()).digest()

    with pytest.raises(MemoryQLValidationError):
        service.query(
            "DELETE MemoryEvents\nMATCH\n  $event ISA Event\nRETURN TABLE\nLIMIT ROWS 10",
            language_version=LANGUAGE_VERSION,
        )

    assert hashlib.sha256(memory_db.read_bytes()).digest() == before
    with sqlite3.connect(memory_db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM MemoryEvents").fetchone()[0] == 4


def test_string_identifiers_are_queryable_without_exposing_storage_columns(memory_db) -> None:
    result = SemanticMemoryService(memory_db).query(
        '''MATCH
  $entity ISA CanonicalEntity
WHERE $entity.id = "entity-admin"
RETURN TABLE
LIMIT ROWS 10''',
        language_version=LANGUAGE_VERSION,
    )

    assert [node["label"] for node in result["nodes"]] == ["admin"]
    assert result["table"]["rows"][0]["id"] == "entity-admin"


def test_service_clamps_requested_budget_and_marks_truncation(memory_db) -> None:
    result = SemanticMemoryService(memory_db).query(
        _involves_query(node_limit=999, edge_limit=999),
        language_version=LANGUAGE_VERSION,
        node_limit=2,
        edge_limit=1,
        row_limit=1,
    )

    assert result["budget"]["effective"]["nodes"] == 2
    assert result["budget"]["effective"]["edges"] == 1
    assert result["budget"]["effective"]["rows"] == 1
    assert result["budget"]["clamped"] is True
    assert result["truncated"] is True
    consumed = result["budget"]["consumed"]
    assert consumed["nodes"] == 2
    assert consumed["edges"] == 1
    assert consumed["rows"] == 1
    assert consumed["elapsed_ms"] >= 0


def test_missing_processing_table_degrades_schema_and_rejects_only_affected_query(memory_db) -> None:
    with sqlite3.connect(memory_db) as connection:
        connection.execute("DROP TABLE MemoryEntityMentions")

    service = SemanticMemoryService(memory_db)
    schema = service.schema()
    relations = {item["name"]: item for item in schema["relations"]}

    assert schema["compatibility"]["status"] == "degraded"
    assert relations["INVOLVES"]["available"] is False
    assert "INVOLVES" in schema["compatibility"]["missing"]
    assert "MemoryEntityMentions" in schema["compatibility"]["missing_tables"]
    with pytest.raises(MemoryQueryUnavailable):
        service.query(_involves_query(), language_version=LANGUAGE_VERSION)


def test_v1_memory_routes_expose_contract_and_error_models(memory_db, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(
            routes_ui_v1,
            "semantic_memory_service",
            SemanticMemoryService(memory_db),
        )
        app = Quart(__name__)
        app.register_blueprint(routes_ui_v1.ui_v1_bp)
        client = app.test_client()

        schema_response = await client.get("/api/ui/v1/memory/schema")
        assert schema_response.status_code == 200
        assert (await schema_response.get_json())["data"]["schema_version"]

        query_response = await client.post(
            "/api/ui/v1/memory/query",
            json={
                "query": _involves_query(),
                "language_version": LANGUAGE_VERSION,
                "node_limit": 10,
                "edge_limit": 10,
                "row_limit": 10,
                "max_depth": 2,
            },
        )
        assert query_response.status_code == 200
        assert (await query_response.get_json())["data"]["nodes"]

        invalid_version = await client.post(
            "/api/ui/v1/memory/query",
            json={"query": _involves_query(), "language_version": "9.9"},
        )
        invalid_budget = await client.post(
            "/api/ui/v1/memory/query",
            json={
                "query": _involves_query(),
                "language_version": LANGUAGE_VERSION,
                "node_limit": True,
            },
        )

        assert invalid_version.status_code == 422
        assert (await invalid_version.get_json())["error"]["code"] == "memoryql_validation_error"
        assert invalid_budget.status_code == 400
        assert (await invalid_budget.get_json())["error"]["code"] == "invalid_request"

    asyncio.run(scenario())


def test_v1_memory_route_maps_timeout_without_treating_it_as_bad_input(monkeypatch) -> None:
    class TimeoutService:
        def query(self, *_args, **_kwargs):
            raise MemoryQueryTimeout("查询超过服务端时间预算")

    async def scenario() -> None:
        monkeypatch.setattr(routes_ui_v1, "semantic_memory_service", TimeoutService())
        app = Quart(__name__)
        app.register_blueprint(routes_ui_v1.ui_v1_bp)
        response = await app.test_client().post(
            "/api/ui/v1/memory/query",
            json={"query": _involves_query(), "language_version": LANGUAGE_VERSION},
        )

        assert response.status_code == 408
        assert (await response.get_json())["error"]["code"] == "memory_query_budget_exceeded"

    asyncio.run(scenario())
