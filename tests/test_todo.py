from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
from xml.etree import ElementTree as ET

import pytest

import app_state
import database
from llm.prompt import container, goals, todo
from tools import build_tools
from tools.core import update_plan


@pytest.fixture(autouse=True)
def isolated_todo():
    previous = todo.get_snapshot()
    todo.restore(None)
    yield
    todo.restore(previous)


@pytest.fixture
def handler(monkeypatch):
    asyncio.run(database.init_db())
    loop = asyncio.new_event_loop()
    ready = Event()
    loop.call_soon(ready.set)
    thread = Thread(target=loop.run_forever)
    thread.start()
    ready.wait()
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        yield build_tools({}).active_specs["core.update_plan"].handler
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.close()


def test_updates_replace_persist_restore_and_clear(handler):
    assert asyncio.run(database.load_todo_snapshot()) is None
    first = [{"step": "one", "status": "in_progress"}, {"step": "two", "status": "pending"}]
    assert handler(plan=first, explanation="initial")["ok"]
    assert todo.get_snapshot()["plan"] == first
    assert todo.get_snapshot()["explanation"] == "initial"

    revised = [{"step": "two", "status": "completed"}]
    assert handler(plan=revised)["ok"]
    saved = asyncio.run(database.load_todo_snapshot())
    assert saved == todo.get_snapshot()
    assert saved["plan"] == revised
    assert saved["explanation"] is None
    assert saved["updated_at"] > 0
    todo.restore(None)
    todo.restore(asyncio.run(database.load_todo_snapshot()))
    assert todo.get_snapshot() == saved
    assert ET.fromstring(container.build_container_xml()).find("preset/todo/item").get("status") == "completed"

    # Public snapshots must not allow callers to mutate the authoritative state.
    detached = todo.get_snapshot()
    detached["plan"][0]["step"] = "uncommitted"
    assert todo.get_snapshot() == saved

    assert handler(plan=[])["ok"]
    todo.restore(asyncio.run(database.load_todo_snapshot()))
    assert todo.get_snapshot()["plan"] == []
    assert ET.fromstring(container.build_container_xml()).find("preset/todo") is None


@pytest.mark.parametrize("arguments", [
    {}, {"plan": None}, {"plan": [], "unknown": True},
    {"plan": [{"step": "x"}]},
    {"plan": [{"step": "x", "status": "blocked"}]},
    {"plan": [{"step": 1, "status": "pending"}]},
    {"plan": [{"step": "x", "status": "pending", "id": "1"}]},
    {"plan": [], "explanation": 123},
])
def test_invalid_arguments_do_not_write(handler, arguments):
    assert handler(plan=[{"step": "keep", "status": "pending"}])["ok"]
    before = todo.get_snapshot()
    assert handler(**arguments).get("error")
    assert todo.get_snapshot() == before
    assert asyncio.run(database.load_todo_snapshot()) == before


def test_codex_advisory_constraints_are_not_hard_limits(handler):
    plan = [{"step": "", "status": "in_progress"}] * 20
    assert handler(plan=plan)["ok"]
    assert todo.get_snapshot()["plan"] == plan


def test_write_failure_keeps_previous_snapshot(handler, monkeypatch):
    assert handler(plan=[{"step": "keep", "status": "pending"}])["ok"]
    before = todo.get_snapshot()

    async def fail(snapshot):
        raise OSError("fixture write failure")

    monkeypatch.setattr(database, "save_todo_snapshot", fail)
    assert handler(plan=[])["ok"] is False
    assert todo.get_snapshot() == before
    assert asyncio.run(database.load_todo_snapshot()) == before


def test_unavailable_loop_does_not_update(monkeypatch):
    monkeypatch.setattr(app_state, "main_loop", None)
    before = todo.get_snapshot()
    assert update_plan.execute(plan=[])["ok"] is False
    assert todo.get_snapshot() == before


def test_concurrent_calls_serialize_and_publish_after_commit(handler, monkeypatch):
    assert handler(plan=[{"step": "old", "status": "pending"}])["ok"]
    before = todo.get_snapshot()
    save = database.save_todo_snapshot
    entered = Event()
    second_started = Event()
    release = Event()
    active = 0
    peak = 0
    update = todo.update_plan

    async def observed_update(plan, explanation=None):
        if plan[0]["step"] == "second":
            # The callback runs after this task yields at the write lock (or
            # inside slow_save if serialization has accidentally been removed).
            asyncio.get_running_loop().call_soon(second_started.set)
        await update(plan, explanation)

    async def slow_save(snapshot):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.005)
            await save(snapshot)
        finally:
            active -= 1

    monkeypatch.setattr(database, "save_todo_snapshot", slow_save)
    monkeypatch.setattr(todo, "update_plan", observed_update)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(handler, plan=[{"step": "first", "status": "completed"}])
        try:
            assert entered.wait(timeout=5)
            second = pool.submit(handler, plan=[{"step": "second", "status": "pending"}])
            assert second_started.wait(timeout=5)
            assert todo.get_snapshot() == before
            assert asyncio.run(database.load_todo_snapshot()) == before
        finally:
            release.set()
        assert first.result(timeout=5)["ok"]
        assert second.result(timeout=5)["ok"]
    assert peak == 1
    assert todo.get_snapshot()["plan"] == [{"step": "second", "status": "pending"}]
    assert asyncio.run(database.load_todo_snapshot()) == todo.get_snapshot()


def test_container_order_and_xml_escaping(handler, monkeypatch):
    monkeypatch.setattr(goals, "_goals", [{
        "goal_id": "fixture", "created_at": 0, "goal": "goal", "background": "context",
    }])
    text = '<tag attr="x"> & </item>'
    plan = [{"step": text, "status": "pending"}, {"step": "next", "status": "completed"}]
    assert handler(plan=plan, explanation=text)["ok"]
    root = ET.fromstring(container.build_container_xml())
    preset = root.find("preset")
    tags = [child.tag for child in preset]
    assert tags.index("goal") < tags.index("todo")
    block = preset.find("todo")
    assert block.find("des") is not None
    assert block.findtext("explanation") == text
    assert [{"step": item.text, "status": item.get("status")} for item in block.findall("item")] == plan
