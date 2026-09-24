"""Explicit access and context deduplication are independent of recall policy."""
import asyncio
import xml.etree.ElementTree as ET

import pytest

import app_state
import database
from consciousness.flow import ConsciousnessFlow, ToolResponse
from memory.active import store
from memory.recall.render import build_memory_xml, prepare_memory_context
from memory.preview import MEMORY_PREVIEW_CHARS
from llm.prompt.sections import build_memory_block
from tools.memory_manage.memory_search import execute as search
from tools.memory_manage.memory_read import execute as read


@pytest.fixture
def memories(monkeypatch):
    from memory.repo import events
    monkeypatch.setattr(events, "_SCHEMA_READY", False)
    asyncio.run(events.ensure_schema())
    store.initialize(database.DB_PATH)
    batch = store.enqueue(database.DB_PATH, "fixture", ["shared active"])
    number = store.parse_id(batch["batch_id"], "B")
    store.begin(database.DB_PATH, number)
    store.complete(database.DB_PATH, number, [
        {"action": "upsert", "id": None, "content": f"shared active {i}"}
        for i in range(5)
    ], "fixture", [])
    with store.connection(database.DB_PATH) as con:
        con.execute("INSERT INTO MemoryEvents "
                    "(summary, event_type, event_type_norm, occurred_at, created_at, raw_event_json) "
                    "VALUES ('shared event', 'observe', 'observe', 1, 1, '{}')")
        con.execute("INSERT INTO MemorySummaryCache "
                    "(summary_id, task_id, input_hash, status, summary) "
                    "VALUES ('summary:storyline:test', 'test', 'hash', 'ready', 'shared storyline')")
    return batch


def test_search_all_stores_with_real_pagination_and_readable_ids(memories, monkeypatch):
    from memory.recall import recall_query
    def forbidden(*args, **kwargs):
        raise AssertionError("explicit search must not invoke automatic recall")
    monkeypatch.setattr(recall_query, "recall_events_from_facets", forbidden)
    monkeypatch.setattr(app_state, "main_loop", None)
    monkeypatch.setattr(app_state, "config", {"memory": {"active": {"enabled": False}}})
    items = []
    for offset in (0, 3, 6):
        page = search(query="shared", limit=3, offset=offset)
        items.extend(page["items"])
        assert page["has_more"] is (offset < 6)
    assert len(items) == len({item["id"] for item in items}) == 7
    assert sum(item["kind"] == "active" for item in items) == 5
    assert {item["kind"] for item in items} == {"active", "event", "summary"}
    for item in items:
        assert read(id=item["id"])["item"]["content"] == item["content"]
        assert search(query=item["id"])["items"][0]["id"] == item["id"]
    assert search(query="no_such_memory")["items"] == []


def test_literal_matching_and_hidden_records(memories):
    assert [item["content"] for item in search(query="shared event", literal=True)["items"]] == ["shared event"]
    assert search(query="shared%", literal=True)["items"] == []
    with store.connection(database.DB_PATH) as con:
        con.execute("UPDATE MemoryEvents SET is_deleted=1")
        con.execute("UPDATE MemorySummaryCache SET status='pending'")
        con.execute("UPDATE ActiveMemoryEntries SET status='retracted'")
    assert search(query="shared")["items"] == []
    assert read(id="E000001")["found"] is False
    assert read(id="summary:storyline:test")["found"] is False
    assert read(id="M000001")["found"] is False


@pytest.mark.parametrize("table,column,item_id,event", [
    ("ActiveMemoryEntries", "content", "M000001", {"memory_kind": "active", "memory_id": "M000001"}),
    ("MemoryEvents", "summary", "E000001", {"event_id": 1}),
    ("MemorySummaryCache", "summary", "summary:storyline:test",
     {"memory_kind": "summary", "summary_id": "summary:storyline:test"}),
])
@pytest.mark.parametrize("length_delta", [-1, 0, 1])
def test_preview_boundaries_and_lossless_read(memories, table, column, item_id, event, length_delta):
    # A match at the end must remain searchable even when omitted from the preview.
    length = MEMORY_PREVIEW_CHARS + length_delta
    content = ("中<&🙂\n" * length)[:length - 8] + "tail-key"
    with store.connection(database.DB_PATH) as con:
        con.execute(f"UPDATE {table} SET {column}=?", (content,))
    item = next(item for item in search(query="tail-key", literal=True)["items"] if item["id"] == item_id)
    assert item["content"] == content[:MEMORY_PREVIEW_CHARS]
    assert item["truncated"] is (length_delta > 0)
    assert read(id=item_id)["item"]["content"] == content
    block = build_memory_block(build_memory_xml(recalled_events=[{**event, "summary": content}]))
    entry = ET.fromstring(prepare_memory_context(block, [])).find("mem")
    assert entry.text == item["content"]
    assert (entry.get("truncated") == "true") is item["truncated"]


def test_long_memory_dedup_requires_full_current_content(memories):
    content = "正文<&🙂" * MEMORY_PREVIEW_CHARS + "旧结论"
    event = {"memory_kind": "active", "memory_id": "M000001", "summary": content}
    with store.connection(database.DB_PATH) as con:
        con.execute("UPDATE ActiveMemoryEntries SET content=? WHERE id=1", (content,))
    block = build_memory_block(build_memory_xml(recalled_events=[event]))
    response = ToolResponse(name="memory_search", namespace="memory_manage", response=search(query="M000001"))
    assert ET.fromstring(prepare_memory_context(block, [response])).find("mem") is not None
    full = ToolResponse(name="memory_read", namespace="memory_manage", response=read(id="M000001"))
    assert ET.fromstring(prepare_memory_context(block, [full])).find("mem") is None
    # Equal previews must not hide a changed conclusion beyond the visible prefix.
    changed = content[:-3] + "新结论"
    with store.connection(database.DB_PATH) as con:
        con.execute("UPDATE ActiveMemoryEntries SET content=? WHERE id=1", (changed,))
    updated = build_memory_block(build_memory_xml(recalled_events=[{**event, "summary": changed}]))
    assert ET.fromstring(prepare_memory_context(updated, [response, full])).find("mem") is not None
    assert read(id="M000001")["item"]["content"] == changed
    # Once the full response leaves visible history, automatic previews return.
    assert ET.fromstring(prepare_memory_context(block, [])).find("mem").get("truncated") == "true"


@pytest.mark.parametrize("name", ["memory_search", "memory_read"])
@pytest.mark.parametrize("event,item_id", [
    ({"memory_kind": "active", "memory_id": "M000001"}, "M000001"),
    ({"event_id": 1}, "E000001"),
    ({"memory_kind": "summary", "summary_id": "summary:storyline:test"}, "summary:storyline:test"),
])
def test_dedup_keeps_tool_result_and_recovers_after_compression(name, event, item_id):
    content = "记忆 <with> & escaped text\nsecond line"
    item = {"id": item_id, "content": content}
    payload = {"items": [item]} if name == "memory_search" else {"found": True, "item": item}
    response = ToolResponse(name=name, namespace="memory_manage", response=payload)
    flow = ConsciousnessFlow()
    flow.append_round([], [response], cognition="remember")
    # Old cycles also keep tool result bodies, so exclusions must survive that transition.
    for _ in range(3):
        flow.append_round([], [], cognition="later")
    block = build_memory_block(build_memory_xml(recalled_events=[{**event, "summary": content}]))
    assert not ET.fromstring(prepare_memory_context(block, flow.visible_tool_responses())).findall("mem")
    assert response.response == payload
    updated = build_memory_block(build_memory_xml(recalled_events=[{**event, "summary": "new revision"}]))
    assert ET.fromstring(prepare_memory_context(updated, flow.visible_tool_responses())).findtext("mem") == "new revision"
    assert flow.queue_compression_summary("lossy summary", coverage_end_seq=1)
    assert flow.promote_ready_compression_summary(max_rounds=3)
    assert prepare_memory_context(block, flow.visible_tool_responses()) == block


def test_dedup_does_not_merge_different_ids():
    block = build_memory_block(build_memory_xml(recalled_events=[
        {"event_id": 1, "summary": "same words"},
        {"memory_kind": "active", "memory_id": "M000001", "summary": "same words"},
    ]))
    responses = [ToolResponse(name="memory_search", namespace="memory_manage", response={
        "items": [{"id": "M000001", "content": "same words"}],
    })]
    root = ET.fromstring(prepare_memory_context(block, responses))
    assert [entry.get("id") for entry in root.findall("mem")] == ["E000001"]


def test_memory_namespace_and_bound_skill_stay_available(monkeypatch):
    from tools import build_tools
    from tools.namespaces import load_namespace_registry, NamespaceRuntimeState
    from skills import build_skill_block_for_namespaces
    from skills import registry as skills
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    monkeypatch.setattr(app_state, "namespace_runtime_state", state)
    monkeypatch.setattr(skills, "load_skill_body", lambda _: "skill sentinel")
    assert "memory_manage" in state.active_namespaces(registry)
    state.close("memory_manage", registry)
    assert "memory_manage" in state.active_namespaces(registry)
    collection = build_tools(config={})
    for name in ("memory_write", "memory_search", "memory_read"):
        spec = collection.active_specs[f"memory_manage.{name}"]
        assert spec.visible_namespace == "memory_manage"
        assert spec.always_available
    block = build_skill_block_for_namespaces(state.active_namespaces(registry), registry)
    assert "skill sentinel" in block
    assert 'from="namespace.memory_manage"' in block


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("long_memory", [False, True])
def test_actual_model_request_deduplicates_after_summary_promotion(monkeypatch, compressed, long_memory):
    from types import SimpleNamespace
    from tools import build_tools
    from llm.core.round_runner import LLMRoundRunner
    from llm.prompt.sections import UserPromptSections
    from llm.compression.config import MIN_LLM_CONTENTS_MAX_ROUNDS

    content = "unique memory body" + ("长" * MEMORY_PREVIEW_CHARS if long_memory else "")
    flow = ConsciousnessFlow()
    flow.append_round([], [ToolResponse(
        name="memory_read", namespace="memory_manage", result_cdata=True,
        response={"found": True, "item": {"id": "M000001", "content": content}},
    )], cognition="first")
    for _ in range(MIN_LLM_CONTENTS_MAX_ROUNDS):
        flow.append_round([], [], cognition="later")
    if compressed:
        assert flow.queue_compression_summary("compressed history", coverage_end_seq=1)
    runner = object.__new__(LLMRoundRunner)
    runner.provider = "test"
    runner.model = "test-model"
    runner._vision_enabled = True
    runner._assistant_prefill_supported = True
    runner._prompt_snapshot_cfg = {"enabled": False}
    runner._discarded_response_log_cfg = {"enabled": False}
    runner._last_main_stable_prompt_prefix = None
    monkeypatch.setattr(runner, "_normalize_generation_for_transport", lambda gen: dict(gen or {}))
    monkeypatch.setattr("llm.core.round_runner._record_usage_event", lambda **_: None)
    captured = []
    def completion(**kwargs):
        captured.extend(kwargs["all_messages"])
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(
                content="<cognition>done</cognition><action></action>", reasoning_content="",
            ),
        )])
    monkeypatch.setattr(runner, "_create_chat_completion", completion)
    block = build_memory_block(build_memory_xml(recalled_events=[{
        "memory_kind": "active", "memory_id": "M000001", "summary": content,
    }]))
    result = runner.call_one_round(
        lambda *_, **__: "system", "<world/>", {"llm_contents_max_rounds": 1},
        build_tools(config={}), flow,
        prompt_sections=UserPromptSections(memory=block, world="<world/>"),
    )
    assert not result.failed
    text = "\n".join(message["content"] for message in captured if isinstance(message.get("content"), str))
    assert text.count("unique memory body") == 1
    tail = next(message["content"] for message in reversed(captured)
                if isinstance(message.get("content"), str) and "<memory>" in message["content"])
    assert ("unique memory body" in tail) is compressed
    if compressed:
        memory_block = tail[tail.index("<memory>"):tail.index("</memory>") + len("</memory>")]
        entry = ET.fromstring(memory_block).find("mem")
        assert entry.text == content[:MEMORY_PREVIEW_CHARS]
        assert (entry.get("truncated") == "true") is long_memory
    else:
        assert content in text
