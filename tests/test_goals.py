from __future__ import annotations

import pytest
from datetime import datetime, timezone

from llm.prompt import goals
from tools.goals import goal_create
from tools.goals import goal_resolve


def test_empty_active_goals_xml():
    goals.restore([])
    xml = goals.build_active_goals_xml()
    assert '<active items="0/10">' in xml
    assert "<origin>" not in xml
    assert "<title>" not in xml
    assert "<content>" not in xml


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
        "background": "用户要求将 goal_manage 升级为独立常驻命名空间 goals",
        "status": "active",
        "resolution": "",
    }
    goals.restore([entry])

    xml = goals.build_active_goals_xml(now=now)
    assert '<active items="1/10">' in xml
    assert '<item id="goal_test1234">' in xml
    assert "<goal>完成目标管理工具重构</goal>" in xml
    assert "<background>用户要求将 goal_manage 升级为独立常驻命名空间 goals</background>" in xml
    assert "<age>刚刚</age>" in xml
    # Assert origin, title, content, reason are absent
    assert "<origin>" not in xml
    assert "<title>" not in xml
    assert "<content>" not in xml
    assert "<reason>" not in xml


def test_goal_create_args_validation():
    args = goal_create.GoalCreateArgs(
        goal="测试目标",
        background="测试背景信息",
    )
    assert args.goal == "测试目标"
    assert args.background == "测试背景信息"

    with pytest.raises(Exception):
        goal_create.GoalCreateArgs(goal="", background="只有背景")

    with pytest.raises(Exception):
        goal_create.GoalCreateArgs(goal="只有目标", background="")


def test_goal_resolve_args_validation():
    args = goal_resolve.GoalResolveArgs(
        goal_id="goal_test1234",
        resolution="completed",
    )
    assert args.goal_id == "goal_test1234"
    assert args.resolution == "completed"

    with pytest.raises(Exception):
        goal_resolve.GoalResolveArgs(goal_id="", resolution="completed")

    with pytest.raises(Exception):
        goal_resolve.GoalResolveArgs(goal_id="goal_test1234", resolution="invalid_type")  # type: ignore
