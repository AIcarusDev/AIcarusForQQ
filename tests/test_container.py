from __future__ import annotations

import asyncio
import pytest

import database
from llm.prompt import container
from llm.prompt.user_prompt_builder import build_main_user_prompt
from llm.session import create_session
from platforms import PlatformRegistry
from platforms.qq import QQRuntime


@pytest.fixture(autouse=True)
def clean_container_state():
    """每个测试前后确保 container 内存状态被清空。"""
    container.restore([])
    yield
    container.restore([])


def test_container_empty_renders_self_closing_tag():
    assert container.build_container_xml() == "<container/>"


def test_container_renders_strict_skeleton_with_items():
    # 只有 custom 条目
    container.restore([
        {
            "item_id": "cont_001",
            "section": "custom",
            "item_key": "user_note",
            "content": "some custom content",
        }
    ])
    xml = container.build_container_xml()
    assert xml.startswith("<container>\n  <preset/>\n  <custom>\n")
    assert '<item id="cont_001" key="user_note">some custom content</item>' in xml
    assert xml.endswith("</custom>\n</container>")

    # 只有 preset 条目
    container.restore([
        {
            "item_id": "cont_002",
            "section": "preset",
            "item_key": "sys_rule",
            "content": "rule content",
        }
    ])
    xml_preset = container.build_container_xml()
    assert "<preset>\n" in xml_preset
    assert '<item id="cont_002" key="sys_rule">rule content</item>' in xml_preset
    assert "  <custom/>" in xml_preset

    # 两侧均有条目
    container.restore([
        {
            "item_id": "cont_p1",
            "section": "preset",
            "item_key": "p_key",
            "content": "p_val",
        },
        {
            "item_id": "cont_c1",
            "section": "custom",
            "item_key": "c_key",
            "content": "c_val",
        },
    ])
    xml_both = container.build_container_xml()
    assert "<preset>\n" in xml_both
    assert "<custom>\n" in xml_both
    assert "<preset/>" not in xml_both
    assert "<custom/>" not in xml_both


def test_container_in_memory_crud():
    async def scenario():
        item = await container.add_item("custom", "test_content", key="k1", metadata={"a": 1}, persist=False)
        assert item["section"] == "custom"
        assert item["content"] == "test_content"
        assert item["item_key"] == "k1"

        all_items = container.get_all()
        assert len(all_items) == 1
        assert container.get_items("custom") == all_items
        assert container.get_items("preset") == []

        removed = await container.remove_item(item["item_id"], persist=False)
        assert removed is True
        assert len(container.get_all()) == 0
        assert container.build_container_xml() == "<container/>"

    asyncio.run(scenario())


def test_container_database_persistence(tmp_path, monkeypatch):
    async def scenario():
        test_db = str(tmp_path / "test_container.db")
        monkeypatch.setattr(database, "DB_PATH", test_db)
        await database.init_db()

        await database.write_container_item("cont_db1", "preset", "rule1", "hello world", {"author": "admin"})
        rows = await database.load_container_items()
        assert len(rows) == 1
        assert rows[0]["item_id"] == "cont_db1"
        assert rows[0]["section"] == "preset"
        assert rows[0]["item_key"] == "rule1"
        assert rows[0]["content"] == "hello world"
        assert rows[0]["metadata"] == {"author": "admin"}

        # 测试软删除
        deleted = await database.soft_delete_container_item("cont_db1")
        assert deleted is True
        rows_after = await database.load_container_items()
        assert len(rows_after) == 0

    asyncio.run(scenario())


def test_container_injected_at_the_end_of_user_prompt():
    import app_state

    previous = getattr(app_state, "platform_registry", None)
    app_state.platform_registry = PlatformRegistry()
    app_state.platform_registry.register(QQRuntime({}))
    try:
        session = create_session("group_123456")
        prompt = build_main_user_prompt(session)

        if isinstance(prompt, str):
            assert prompt.strip().endswith("<container/>")
            assert "</world>\n<container/>" in prompt
        elif isinstance(prompt, list):
            last_text = prompt[-1]["text"]
            assert last_text.strip().endswith("<container/>")
    finally:
        app_state.platform_registry = previous
