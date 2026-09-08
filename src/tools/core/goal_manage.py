"""Create or remove an active goal through core.goal_manage."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import Field, RootModel

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, tool


class GoalCreateArgs(ToolArgsModel):
    action: Literal["create"]
    goal: str = Field(min_length=1, description="目标本身，简洁明确。")
    background: str = Field(min_length=1, description="目标的背景、原因、来源等上下文。请基于具体情况，写入需要的背景信息。")


class GoalDeleteArgs(ToolArgsModel):
    action: Literal["delete"]
    goal_id: str = Field(min_length=1, description="要移除的目标 ID，来自活跃目标列表。")
    resolution: Literal["completed", "abandoned", "duplicate", "superseded", "mistaken"] = Field(
        description="目标结束的方式：completed(完成), abandoned(放弃), duplicate(重复), superseded(被替代), mistaken(误建)。",
    )


class GoalManageArgs(RootModel[Annotated[GoalCreateArgs | GoalDeleteArgs, Field(discriminator="action")]]):
    pass


@tool(
    name="goal_manage",
    description="创建目标，或从活跃目标中移除指定目标。删除时保留历史记录及结束方式。",
    args_model=GoalManageArgs,
)
def execute(args: GoalManageArgs) -> dict:
    if isinstance(args.root, GoalCreateArgs):
        return _create(args.root)
    return _delete(args.root)


def _create(args: GoalCreateArgs) -> dict:
    import app_state
    from llm.prompt import goals as _goals

    loop: asyncio.AbstractEventLoop | None = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"error": "主事件循环不可用，无法创建目标"}

    goal = args.goal.strip()
    background = args.background.strip()
    if not goal or not background:
        return {"ok": False, "error": "目标与背景均不能为空"}

    existing = _goals.get_all()
    for item in existing:
        if (item.get("goal") or item.get("title")) == goal:
            return {
                "ok": False,
                "message": "已存在相同目标，无需重复创建",
                "goal": goal,
                "total": len(existing),
            }

    try:
        entry = run_coroutine_sync(
            _goals.add_goal(goal=goal, background=background),
            loop,
            timeout=10,
        )
    except Exception as exc:
        return {"error": f"创建目标失败: {exc}"}

    return {
        "ok": True,
        "action": "create",
        "created": {
            "goal_id": entry["goal_id"],
            "goal": entry["goal"],
            "background": entry["background"],
        },
        "total": len(_goals.get_all()),
    }


def _delete(args: GoalDeleteArgs) -> dict:
    import app_state
    from llm.prompt import goals as _goals

    loop: asyncio.AbstractEventLoop | None = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"error": "主事件循环不可用，无法结束目标"}

    goal_id = args.goal_id.strip()
    if not goal_id:
        return {"ok": False, "error": "goal_id 不能为空"}

    try:
        resolved = run_coroutine_sync(
            _goals.resolve_goal(goal_id, resolution=args.resolution),
            loop,
            timeout=10,
        )
    except Exception as exc:
        return {"error": f"结束目标失败: {exc}"}

    if resolved is None:
        return {
            "ok": False,
            "message": f"未找到活跃目标 {goal_id}",
            "total": len(_goals.get_all()),
        }

    return {
        "ok": True,
        "action": "delete",
        "resolved": {
            "goal_id": resolved["goal_id"],
            "goal": resolved.get("goal") or resolved.get("title") or "",
            "status": resolved["status"],
            "resolution": resolved["resolution"],
        },
        "total": len(_goals.get_all()),
    }


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
