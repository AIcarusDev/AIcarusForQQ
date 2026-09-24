"""Contracts: durable batches, FIFO, atomic revisions, and host-independent access."""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

import app_state
import database
from memory.active import store, workflow


@pytest.fixture
def db(monkeypatch):
    monkeypatch.setattr(app_state, "config", {})
    monkeypatch.setattr(workflow, "_service", None)
    store.initialize(database.DB_PATH)
    return database.DB_PATH


def response(*changes):
    return json.dumps({"changes": list(changes), "note": "processed"})


def add(content, item_id=None):
    return {"action": "upsert", "id": item_id, "content": content}


def seed(db, content):
    batch = store.enqueue(db, "fixture cognition", [content])
    number = store.parse_id(batch["batch_id"], "B")
    assert store.begin(db, number)
    store.complete(db, number, [add(content)], "created", [])
    return store.read(db, batch["batch_id"])["result"]["memory_ids"][0]


def test_batch_is_durable_and_consolidator_receives_whole_input(db):
    old_id = seed(db, "technical answers should include reasons")
    proposals = ["technical answers: conclusion first", "cooking preference: less salt"]
    submitted = store.enqueue(db, "current cognition sentinel", proposals)
    payloads = []

    def generate(payload):
        payloads.append(payload)
        return response(add("technical answers: conclusion then reasons", old_id), add("cooking: less salt"))

    # Fresh service object reads the persisted batch, not an in-memory payload.
    service = workflow.ActiveMemoryService(db, generate)
    assert asyncio.run(service.process_one())
    payload = payloads[0]
    assert payload["cognition"] == "current cognition sentinel"
    assert payload["memories"] == proposals
    assert any(item["id"] == old_id for item in payload["related"])
    result = store.read(db, submitted["batch_id"])
    assert result["status"] == "completed"
    assert len(result["result"]["memory_ids"]) == 2
    assert store.read(db, old_id)["revision"] == 2


def test_failed_head_blocks_later_batches_then_retries_in_order(db):
    first = store.enqueue(db, "cognition 1", ["first batch"])
    second = store.enqueue(db, "cognition 2", ["second batch"])
    calls = []

    def generate(payload):
        calls.append(payload["memories"])
        if len(calls) == 1:
            return "not JSON"
        return response(add(payload["memories"][0]))

    service = workflow.ActiveMemoryService(db, generate)

    async def run():
        assert await service.process_one()
        assert not await service.process_one()
        assert store.read(db, first["batch_id"])["status"] == "retry"
        assert store.read(db, second["batch_id"])["attempts"] == 0
        with store.connection(db) as con:
            con.execute("UPDATE ActiveMemoryBatches SET retry_at=0")
        assert await service.process_one()
        assert await service.process_one()

    asyncio.run(run())
    assert calls == [["first batch"], ["first batch"], ["second batch"]]


def test_batch_rolls_back_all_changes_if_a_target_is_invalid(db):
    old_id = seed(db, "retained")
    batch = store.enqueue(db, "cognition", ["proposals"])
    batch_id = store.parse_id(batch["batch_id"], "B")
    assert store.begin(db, batch_id)
    with pytest.raises(ValueError):
        store.complete(db, batch_id, [add("must roll back"), add("wrong", "M999999")], "note", [store.read(db, old_id)])
    assert store.search(db, "must roll back") == []
    assert store.read(db, batch["batch_id"])["status"] == "running"


def test_completed_batch_cannot_be_applied_twice(db):
    item_id = seed(db, "once only")
    with pytest.raises(ValueError):
        store.complete(db, 1, [add("duplicate")], "note", [])
    assert store.read(db, item_id)["revision"] == 1
    assert store.search(db, "duplicate") == []


def test_stale_version_is_rejected_without_overwriting(db):
    item_id = seed(db, "original")
    snapshot = store.read(db, item_id)
    first = store.enqueue(db, "cognition", ["change"])
    assert store.begin(db, 2)
    store.complete(db, 2, [add("newer", item_id)], "note", [snapshot])
    store.enqueue(db, "cognition", ["old worker"])
    assert store.begin(db, 3)
    with pytest.raises(ValueError):
        store.complete(db, 3, [add("stale", item_id)], "note", [snapshot])
    assert store.read(db, item_id)["content"] == "newer"
    assert store.read(db, first["batch_id"])["status"] == "completed"


def test_cancelled_model_cannot_publish_and_restart_recovers(db):
    batch = store.enqueue(db, "cognition", ["pending"])
    entered, release = threading.Event(), threading.Event()

    def blocking(_payload):
        entered.set()
        release.wait(timeout=5)
        return response(add("discarded model output"))

    async def run():
        service = workflow.ActiveMemoryService(db, blocking)
        task = asyncio.create_task(service.process_one())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            store.recover(db)
            resumed = workflow.ActiveMemoryService(db, lambda _: response(add("recovered")))
            assert await resumed.process_one()
        finally:
            release.set()

    asyncio.run(run())
    assert store.read(db, batch["batch_id"])["status"] == "completed"
    assert store.search(db, "discarded", literal=True) == []
    assert len(store.search(db, "recovered", literal=True)) == 1


def test_read_search_and_retraction_keep_current_state_and_history(db, tmp_path):
    item_id = seed(db, "中文测试 喜欢茶 100%_literal")
    assert store.search(db, "喜欢茶")[0]["id"] == item_id
    assert store.search(db, "100%_literal", literal=True)[0]["id"] == item_id
    assert store.search(db, "%unmatched", literal=True) == []
    store.publish(db)
    file = tmp_path / "memory" / "published" / "entries" / f"{item_id}.md"
    assert file.exists()
    snapshot = store.read(db, item_id)
    store.enqueue(db, "correction", [f"retract {item_id}"])
    assert store.begin(db, 2)
    store.complete(db, 2, [{"action": "retract", "id": item_id}], "withdrawn", [snapshot])
    assert store.search(db, "喜欢茶") == []
    history = store.read(db, item_id, history=True)
    assert history["status"] == "retracted"
    assert [item["status"] for item in history["history"]] == ["active", "retracted"]
    store.publish(db)
    assert not file.exists()


def test_publication_failure_does_not_replay_model_or_lose_memory(db, monkeypatch):
    item_id = seed(db, "durable despite mirror failure")
    original = store._atomic_text
    monkeypatch.setattr(store, "_atomic_text", lambda *_: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(OSError):
        store.publish(db)
    assert store.read(db, item_id)["content"] == "durable despite mirror failure"
    assert store.read(db, "B000001")["published"] == 0
    assert store.head(db) is None
    monkeypatch.setattr(store, "_atomic_text", original)
    store.publish(db)
    assert store.read(db, "B000001")["published"] == 1


def test_empty_valid_result_completes_and_malformed_result_retries(db):
    first = store.enqueue(db, "cognition", ["no lasting value"])
    service = workflow.ActiveMemoryService(db, lambda _: response())
    assert asyncio.run(service.process_one())
    assert store.read(db, first["batch_id"])["result"]["memory_ids"] == []
    second = store.enqueue(db, "cognition", ["valuable"])
    service.generate = lambda _: '{"changes": [{"action":"upsert", "content":"valid"}, {"action":"unknown"}], "note":"n"}'
    asyncio.run(service.process_one())
    assert store.read(db, second["batch_id"])["status"] == "retry"
    assert store.search(db, "valid") == []


def test_write_tool_captures_executor_cognition_without_external_anchors(db, monkeypatch):
    from tools import build_tools
    from llm.core.tool_executor import ToolExecutor
    from llm.core.round_context import get_current_inner_state

    captured = []

    async def submit(cognition, memories):
        captured.append((cognition, memories))
        return store.enqueue(db, cognition, memories)

    monkeypatch.setattr(workflow, "submit", submit)

    async def run():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        collection = build_tools(config={})
        call = SimpleNamespace(id="test", function=SimpleNamespace(
            name="memory_write", namespace="memory_manage", arguments=json.dumps({"memories": ["one", "two"]}),
        ))
        executor = ToolExecutor(provider_name="fixture", tool_collection=collection)
        result = await asyncio.to_thread(executor.execute, [call], inner_state={"cognition": "round cognition"})
        assert result.round_responses[0].response["saved"] is True

    asyncio.run(run())
    assert captured == [("round cognition", ["one", "two"])]
    assert get_current_inner_state() == {}
    from tools.memory_manage.memory_write import execute
    result = execute(memories=["one"], source_refs=["external-message"])
    assert result.get("error")
    assert store.read(db, "B000001")["memories"] == ["one", "two"]


def test_read_and_search_tools_work_without_computer_or_main_loop(db, monkeypatch):
    from tools.memory_manage.memory_search import execute as search
    from tools.memory_manage.memory_read import execute as read
    monkeypatch.setattr(app_state, "main_loop", None)
    monkeypatch.setattr(app_state, "workspace_service", None)
    item_id = seed(db, "standalone query")
    assert search(query="standalone")["items"][0]["id"] == item_id
    assert read(id=item_id)["item"]["content"] == "standalone query"
    assert read(id="B000001")["item"]["status"] == "completed"


def test_active_recall_reaches_shared_facade_even_when_old_recall_fails(db, monkeypatch):
    from memory.recall import recall_query
    from memory.recall.render import build_memory_xml
    import xml.etree.ElementTree as ET

    item_id = seed(db, 'cobalt <special> & text')

    async def unavailable(**_kwargs):
        raise RuntimeError("legacy store unavailable")

    monkeypatch.setattr(recall_query, "_recall_event_facets", unavailable)
    result = asyncio.run(recall_query.recall_events_from_facets(
        sender_entity="", context_scope="unrelated:scope", limit=3,
        facets=recall_query.build_recall_query_facets(latest_user_text="cobalt"),
    ))
    assert result[0]["memory_id"] == item_id
    element = ET.fromstring(build_memory_xml(recalled_events=result))
    assert element.attrib["id"] == item_id
    assert element.text == 'cobalt <special> & text'
    assert asyncio.run(workflow.recall("no-such-keyword", 3)) == []


def test_disabled_submission_changes_nothing_and_disabled_recall_is_empty(db, monkeypatch):
    seed(db, "known")
    monkeypatch.setattr(app_state, "config", {"memory": {"active": {"enabled": False}}})
    service = workflow.ActiveMemoryService(db)
    assert service.submit("cognition", ["new"])["saved"] is False
    assert store.head(db) is None
    assert asyncio.run(workflow.recall("known", 3)) == []


def test_schema_initialization_preserves_existing_data(db):
    item_id = seed(db, "persisted")
    store.initialize(db)
    assert store.read(db, item_id)["content"] == "persisted"


def test_one_worker_drains_concurrent_submissions_in_fifo_order(db):
    calls = []

    def generate(payload):
        calls.append(payload["memories"])
        return response(add(payload["memories"][0]))

    async def run():
        service = workflow.ActiveMemoryService(db, generate)
        try:
            first = service.submit("cognition one", ["first"])
            task = service.task
            second = service.submit("cognition two", ["second"])
            assert service.task is task

            async def drained():
                while store.read(db, second["batch_id"])["status"] != "completed":
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(drained(), 3)
            assert store.read(db, first["batch_id"])["status"] == "completed"
        finally:
            await service.stop()
        assert service.submit("late cognition", ["late write"])["saved"] is False

    asyncio.run(run())
    assert calls == [["first"], ["second"]]


def test_unconfigured_model_keeps_batch_and_later_configuration_recovers(db, monkeypatch):
    batch = store.enqueue(db, "cognition", ["remember"])
    monkeypatch.setattr(app_state, "memory_processing_adapter", None)
    service = workflow.ActiveMemoryService(db)
    asyncio.run(service.process_one())
    assert store.read(db, batch["batch_id"])["error"] == "model_unavailable"
    received = []

    def model(_system, user, generation, log_tag):
        received.append((json.loads(user), generation, log_tag))
        return response(add("remember"))

    monkeypatch.setattr(app_state, "memory_processing_adapter", SimpleNamespace(call_simple_text=model))
    monkeypatch.setattr(app_state, "memory_processing_cfg", {"generation": {"temperature": 0.42}})
    with store.connection(db) as con:
        con.execute("UPDATE ActiveMemoryBatches SET retry_at=0")
    asyncio.run(service.process_one())
    assert received[0][0]["cognition"] == "cognition"
    assert received[0][1]["temperature"] == 0.42
    assert received[0][2] == "memory/active_consolidation"
    assert store.read(db, batch["batch_id"])["status"] == "completed"


def test_maintenance_clears_active_queue_and_mirrors_without_late_model_writes(db, monkeypatch, tmp_path):
    from runtime import maintenance
    monkeypatch.setattr(maintenance, "DB_PATH", db)
    item_id = seed(db, "must disappear")
    store.publish(db)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def generate(_payload):
        entered.set()
        release.wait(timeout=5)
        finished.set()
        return response(add("late result"))

    async def run():
        service = workflow.ActiveMemoryService(db, generate)
        monkeypatch.setattr(workflow, "_service", service)
        maint = maintenance.MaintenanceService()
        try:
            service.submit("cognition", ["pending"])
            assert await asyncio.to_thread(entered.wait, 3)
            await maint._cancel_archive_tasks()
            assert service.submit("late cognition", ["rejected"])["saved"] is False
            deleted = await maint._delete_long_term_memory_rows()
            maint._publish_active_memory_views()
            assert deleted["ActiveMemoryEntries"] == 1
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            assert store.search(db, "late result") == []
            assert store.head(db) is None
        finally:
            release.set()
            await service.stop()

    asyncio.run(run())
    assert store.read(db, item_id) is None
    assert not (tmp_path / "memory" / "published" / "entries" / f"{item_id}.md").exists()
