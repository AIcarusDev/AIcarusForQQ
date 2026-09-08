"""goal_create.py - create an active goal."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import Field

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, tool


class GoalCreateArgs(ToolArgsModel):
    goal: str = Field(min_length=1, description="目标本身，简洁明确。")
    background: str = Field(min_length=1, description="目标的背景、原因、来源等上下文信息。")


@tool(
    name="goal_create",
    description="创建一个新的活跃目标。写入目标本身以及相关的背景、原因或来源。",
    args_model=GoalCreateArgs,
)
def execute(args: GoalCreateArgs) -> dict:
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


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
