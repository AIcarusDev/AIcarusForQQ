from __future__ import annotations

import asyncio
from xml.etree import ElementTree as ET

import pytest

import app_state
import database
from llm.prompt import container
from tools.core import container_manage


@pytest.fixture(autouse=True)
def isolated_container():
    previous = container.get_all()
    container.restore([])
    yield
    container.restore(previous)


def test_tool_roundtrip_preserves_text_and_protects_preset(monkeypatch):
    async def scenario():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await database.init_db()
        preset = await container.add_item("preset", "protected")
        content = '  # note\n<item id="fake">& 内容</item>\n  '
        first = await asyncio.to_thread(container_manage.execute, action="add", key="same", content=content)
        second = await asyncio.to_thread(container_manage.execute, action="add", key="same", content="second")
        assert first["ok"] and second["ok"]
        assert first["item_id"] != second["item_id"]
        container.restore(await database.load_container_items())
        root = ET.fromstring(container.build_container_xml(contracts=[]))
        item = root.find(f'custom/item[@id="{first["item_id"]}"]')
        assert item is not None and item.text == content
        assert len(item) == 0
        protected = await asyncio.to_thread(container_manage.execute, action="delete", item_id=preset["item_id"])
        assert protected["ok"] is False
        deleted = await asyncio.to_thread(container_manage.execute, action="delete", item_id=first["item_id"])
        assert deleted["ok"] is True
        missing = await asyncio.to_thread(container_manage.execute, action="delete", item_id=first["item_id"])
        assert missing["ok"] is False
        container.restore(await database.load_container_items())
        assert {it["item_id"] for it in container.get_all()} == {preset["item_id"], second["item_id"]}

    asyncio.run(scenario())


@pytest.mark.parametrize("args", [
    {"action": "add", "content": " \n\t"},
    {"action": "add", "content": "valid", "section": "preset"},
    {"action": "delete", "item_id": " "},
    {"action": "delete", "item_id": "cont_test", "content": "unexpected"},
])
def test_invalid_input_does_not_mutate(args):
    result = container_manage.execute(**args)
    assert "error" in result
    assert container.get_all() == []


def test_failed_persistence_leaves_visible_state_unchanged(monkeypatch):
    async def fail(*args, **kwargs):
        raise OSError("fixture write failure")

    async def scenario():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await database.init_db()
        existing = await container.add_item("custom", "keep")
        monkeypatch.setattr(database, "write_container_item", fail)
        monkeypatch.setattr(database, "soft_delete_container_item", fail)
        added = await asyncio.to_thread(container_manage.execute, action="add", content="new")
        removed = await asyncio.to_thread(container_manage.execute, action="delete", item_id=existing["item_id"])
        assert added["ok"] is False and removed["ok"] is False
        assert container.get_all() == [existing]
        assert [it["item_id"] for it in await database.load_container_items()] == [existing["item_id"]]

    asyncio.run(scenario())


def test_writes_are_serialized_and_uncommitted_content_is_hidden(monkeypatch):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def write(**kwargs):
            calls.append(kwargs["content"])
            if len(calls) == 1:
                entered.set()
                await release.wait()

        monkeypatch.setattr(database, "write_container_item", write)
        first = asyncio.create_task(container.add_item("custom", "first"))
        await entered.wait()
        second = asyncio.create_task(container.add_item("custom", "second"))
        await asyncio.sleep(0)
        assert calls == ["first"]
        assert container.get_all() == []
        release.set()
        await asyncio.gather(first, second)
        assert [it["content"] for it in container.get_all()] == ["first", "second"]

    asyncio.run(scenario())
