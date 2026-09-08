from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import database
from llm.prompt import container, goals
from llm.prompt.user_prompt_builder import build_main_user_prompt
from llm.session import create_session
from platforms import PlatformRegistry
from platforms.qq import QQRuntime


@pytest.fixture(autouse=True)
def clean_container_state():
    """每个测试前后确保 container 内存状态被清空。"""
    previous_goals = goals.get_all()
    goals.restore([])
    container.restore([])
    yield
    container.restore([])
    goals.restore(previous_goals)


def test_provider_controls_format_description_and_visibility():
    now = datetime(2026, 9, 9, tzinfo=timezone.utc)
    seen = []

    def render(timestamp):
        seen.append(timestamp)
        return '<entry code="fixture">safe &amp; text</entry>'

    contract = container.ContainerContract(
        tag="fixture", section="custom", description="concept <&>", render=render,
    )
    root = ET.fromstring(container.build_container_xml(now, contracts=[contract]))
    assert seen == [now]
    assert root.findtext("custom/fixture/des") == "concept <&>"
    assert root.findtext("custom/fixture/entry") == "safe & text"
    hidden = container.ContainerContract(
        tag="hidden", section="preset", description="unused", render=lambda _: "",
    )
    assert container.build_container_xml(contracts=[hidden]) == "<container/>"


@pytest.mark.parametrize("multimodal", [False, True])
def test_goal_provider_reaches_main_prompt_with_escaped_content(monkeypatch, multimodal):
    import llm.prompt.user_prompt_builder as builder

    goals.restore([{
        "goal_id": 'goal_"<&', "created_at": 0,
        "goal": "目标 <&>", "background": "背景 <&>",
    }])
    world = "<world><platform/></world>"
    image_part = {"type": "image_url", "image_url": {"url": "fixture"}}
    content = [{"type": "text", "text": world}, image_part] if multimodal else world
    monkeypatch.setattr(builder, "_wrap_platform_block_with_world", lambda *args: content)
    monkeypatch.setattr(builder.browser, "build_browser_world_content", lambda: "")
    prompt = builder.build_main_user_prompt(create_session("group_123456"))
    if multimodal:
        assert image_part in prompt
        text = "".join(part["text"] for part in prompt if part["type"] == "text")
    else:
        text = prompt
    root = ET.fromstring(text[text.index("<container>"):])
    block = root.find("preset/goal")
    assert block is not None
    assert block.findtext("des") == goals.CONTAINER_CONTRACT.description
    item = block.find("active/item")
    assert item is not None
    assert item.attrib["id"] == 'goal_"<&'
    assert item.findtext("goal") == "目标 <&>"
    assert item.findtext("background") == "背景 <&>"


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
