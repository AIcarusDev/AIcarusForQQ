from __future__ import annotations

import pytest
from datetime import datetime, timezone

from llm.prompt import goals
from tools.core import goal_manage


@pytest.fixture(autouse=True)
def isolate_goal_state():
    previous = goals.get_all()
    goals.restore([])
    yield
    goals.restore(previous)


@pytest.mark.parametrize("resolution", goals.VALID_RESOLUTIONS)
def test_goal_manage_persists_creation_and_resolution(monkeypatch, resolution):
    import asyncio
    import sqlite3
    from types import SimpleNamespace

    import app_state
    import database
    from tools import build_tools

    asyncio.run(database.init_db())
    monkeypatch.setattr(app_state, "main_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(
        goal_manage, "run_coroutine_sync",
        lambda coroutine, loop, timeout: asyncio.run(coroutine),
    )
    collection = build_tools({})
    spec = collection.active_specs["core.goal_manage"]
    created = spec.handler(action="create", goal=" 测试目标 ", background=" 测试背景 ")
    assert created["ok"] is True
    from llm.prompt.container import build_container_xml
    from xml.etree import ElementTree as ET

    goal_block = ET.fromstring(build_container_xml()).find("preset/goal")
    assert goal_block is not None
    assert goal_block.find("des") is not None
    assert goal_block.findtext("active/item/goal") == "测试目标"
    goal_id = created["created"]["goal_id"]
    loaded = asyncio.run(database.load_goals())
    assert [(row["goal"], row["background"]) for row in loaded] == [("测试目标", "测试背景")]
    duplicate = spec.handler(action="create", goal="测试目标", background="不同背景")
    assert duplicate["ok"] is False
    assert len(asyncio.run(database.load_goals())) == 1

    deleted = spec.handler(action="delete", goal_id=goal_id, resolution=resolution)
    assert deleted["ok"] is True
    assert deleted["action"] == "delete"
    assert goals.get_all() == []
    assert build_container_xml() == "<container/>"
    assert asyncio.run(database.load_goals()) == []
    with sqlite3.connect(database.DB_PATH) as connection:
        assert connection.execute(
            "SELECT goal, background, status, resolution, is_deleted FROM bot_goals WHERE goal_id=?",
            (goal_id,),
        ).fetchone() == ("测试目标", "测试背景", "resolved", resolution, 0)
    assert spec.handler(action="delete", goal_id=goal_id, resolution=resolution)["ok"] is False


@pytest.mark.parametrize("arguments", [
    {"goal": "目标", "background": "背景"},
    {"action": "create", "goal": "目标"},
    {"action": "create", "goal": "目标", "background": "背景", "goal_id": "goal_x"},
    {"action": "create", "goal": "目标", "background": "背景", "reason": "旧字段"},
    {"action": "delete", "goal_id": "goal_x"},
    {"action": "delete", "goal_id": "goal_x", "resolution": "invalid"},
    {"action": "delete", "goal_id": "goal_x", "resolution": "completed", "goal": "目标"},
    {"action": "create", "goal": "  ", "background": "背景"},
    {"action": "create", "goal": "目标", "background": "  "},
    {"action": "delete", "goal_id": "  ", "resolution": "completed"},
])
def test_goal_manage_rejects_invalid_arguments_without_writes(monkeypatch, arguments):
    from types import SimpleNamespace
    import app_state

    monkeypatch.setattr(app_state, "main_loop", SimpleNamespace(is_running=lambda: True))

    def unexpected_write(*args, **kwargs):
        pytest.fail("Invalid arguments must not reach persistence")

    monkeypatch.setattr(goal_manage, "run_coroutine_sync", unexpected_write)
    result = goal_manage.execute(**arguments)
    assert result.get("error") or result.get("ok") is False
    assert goals.get_all() == []


def test_empty_active_goals_xml():
    goals.restore([])
    xml = goals.build_active_goals_xml()
    assert xml == ""


@pytest.mark.anyio
async def test_add_goal_and_xml_rendering():
    goals.restore([])
    now = datetime(2026, 9, 8, 4, 30, 0, tzinfo=timezone.utc)

    # In memory test
    entry = {
        "goal_id": "goal_test1234",
        "created_at": int(now.timestamp() * 1000),
        "updated_at": int(now.timestamp() * 1000),
        "goal": "完成目标管理工具重构",
        "background": "用户要求合并目标工具",
        "status": "active",
        "resolution": "",
    }
    goals.restore([entry])

    xml = goals.build_active_goals_xml(now=now)
    assert '<active items="1/10">' in xml
    assert '<item id="goal_test1234">' in xml
    assert "<goal>完成目标管理工具重构</goal>" in xml
    assert "<background>用户要求合并目标工具</background>" in xml
    assert "<age>刚刚</age>" in xml
    # Assert origin, title, content, reason are absent
    assert "<origin>" not in xml
    assert "<title>" not in xml
    assert "<content>" not in xml
    assert "<reason>" not in xml


def test_goal_create_args_validation():
    args = goal_manage.GoalCreateArgs(
        action="create",
        goal="测试目标",
        background="测试背景信息",
    )
    assert args.goal == "测试目标"
    assert args.background == "测试背景信息"

    with pytest.raises(Exception):
        goal_manage.GoalCreateArgs(action="create", goal="", background="只有背景")

    with pytest.raises(Exception):
        goal_manage.GoalCreateArgs(action="create", goal="只有目标", background="")


def test_goal_delete_args_validation():
    args = goal_manage.GoalDeleteArgs(
        action="delete",
        goal_id="goal_test1234",
        resolution="completed",
    )
    assert args.goal_id == "goal_test1234"
    assert args.resolution == "completed"

    with pytest.raises(Exception):
        goal_manage.GoalDeleteArgs(action="delete", goal_id="", resolution="completed")

    with pytest.raises(Exception):
        goal_manage.GoalDeleteArgs(action="delete", goal_id="goal_test1234", resolution="invalid_type")  # type: ignore
