from __future__ import annotations

import asyncio
import sqlite3
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


def test_legacy_migration_keeps_newest_five_with_fresh_lifetime(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    monkeypatch.setattr(database, "DB_PATH", str(path))
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE bot_container_items (
            item_id TEXT PRIMARY KEY, section TEXT NOT NULL, item_key TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT 0, updated_at INTEGER NOT NULL DEFAULT 0,
            is_deleted INTEGER NOT NULL DEFAULT 0)""")
        db.executemany(
            """INSERT INTO bot_container_items
               (item_id, section, content, created_at, updated_at) VALUES (?, 'custom', ?, ?, ?)""",
            [(f"legacy_{i}", str(i), i, i) for i in range(16)],
        )
    async def scenario():
        await database.init_db()
        rows = await database.load_container_items()
        assert {row["item_id"] for row in rows} == {f"legacy_{i}" for i in range(11, 16)}
        assert {row["remaining_rounds"] for row in rows} == {8}
        container.restore(rows)
        root = ET.fromstring(container.build_container_xml(contracts=[]))
        assert len(root.findall("custom/item")) == 5
        await database.init_db()
        assert len(await database.load_container_items()) == 5
    asyncio.run(scenario())
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM bot_container_items WHERE is_deleted=1").fetchone()[0] == 11


def test_keep_refreshes_lifetime_and_eviction_priority(monkeypatch):
    async def scenario():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await database.init_db()
        ids = []
        for i in range(5):
            result = await asyncio.to_thread(container_manage.execute, action="add", content=str(i))
            assert result["ok"]
            ids.append(result["item_id"])
        for _ in range(3):
            await container.advance_completed_round()
        kept = await asyncio.to_thread(container_manage.execute, action="keep", item_id=ids[0])
        assert kept["ok"] and kept["remaining_rounds"] == 8
        added = await asyncio.to_thread(container_manage.execute, action="add", content="sixth")
        assert added["ok"]
        active = {row["item_id"]: row for row in await database.load_container_items()
                  if row["section"] == "custom"}
        assert len(active) == 5
        assert ids[0] in active and ids[1] not in active
        assert active[ids[0]]["remaining_rounds"] == 8
        assert active[added["item_id"]]["remaining_rounds"] == 8
        container.restore(await database.load_container_items())
        assert {it["item_id"] for it in container.get_items("custom")} == set(active)
        assert not (await asyncio.to_thread(container_manage.execute, action="keep", item_id=ids[1]))["ok"]
    asyncio.run(scenario())


def test_completed_rounds_expire_custom_but_not_preset(monkeypatch):
    async def scenario():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await database.init_db()
        preset = await container.add_item("preset", "protected")
        added = await asyncio.to_thread(container_manage.execute, action="add", content="temporary")
        item_id = added["item_id"]
        for remaining in range(7, 0, -1):
            await container.advance_completed_round()
            root = ET.fromstring(container.build_container_xml(contracts=[]))
            item = root.find(f'custom/item[@id="{item_id}"]')
            assert item is not None and item.attrib["remaining_rounds"] == str(remaining)
        await container.advance_completed_round()
        assert [it["item_id"] for it in container.get_all()] == [preset["item_id"]]
        assert not (await asyncio.to_thread(container_manage.execute, action="keep", item_id=item_id))["ok"]
        container.restore(await database.load_container_items())
        assert [it["item_id"] for it in container.get_all()] == [preset["item_id"]]
    asyncio.run(scenario())


def test_failed_keep_or_round_update_does_not_change_memory(monkeypatch):
    async def fail(*args, **kwargs):
        raise OSError("fixture write failure")

    async def scenario():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await database.init_db()
        item = await container.add_item("custom", "valuable")
        await container.advance_completed_round()
        before = dict(container.get_items("custom")[0])
        monkeypatch.setattr(database, "keep_custom_container_item", fail)
        monkeypatch.setattr(database, "advance_custom_container_round", fail)
        result = await asyncio.to_thread(container_manage.execute, action="keep", item_id=item["item_id"])
        assert not result["ok"]
        with pytest.raises(OSError):
            await container.advance_completed_round()
        assert container.get_items("custom")[0] == before
        assert (await database.load_container_items())[0]["remaining_rounds"] == 7
    asyncio.run(scenario())


def test_failed_eviction_rolls_back_new_item(tmp_path, monkeypatch):
    path = tmp_path / "eviction.db"
    monkeypatch.setattr(database, "DB_PATH", str(path))

    async def scenario():
        monkeypatch.setattr(app_state, "main_loop", asyncio.get_running_loop())
        await database.init_db()
        for i in range(5):
            assert (await asyncio.to_thread(container_manage.execute,
                                            action="add", content=str(i)))["ok"]
        before = {it["item_id"] for it in container.get_items("custom")}
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TRIGGER reject_container_eviction
                BEFORE UPDATE OF is_deleted ON bot_container_items
                WHEN NEW.is_deleted=1
                BEGIN SELECT RAISE(ABORT, 'fixture eviction failure'); END""")
        failed = await asyncio.to_thread(container_manage.execute, action="add", content="sixth")
        assert not failed["ok"]
        assert {it["item_id"] for it in container.get_items("custom")} == before
        assert {it["item_id"] for it in await database.load_container_items()} == before

    asyncio.run(scenario())
