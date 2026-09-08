"""goal_resolve.py - resolve an active goal."""

from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import Field

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, tool


class GoalResolveArgs(ToolArgsModel):
    goal_id: str = Field(
        min_length=1,
        description="要结束的目标 ID，来自 active goals（例如 goal_xxxxxxxx）。",
    )
    resolution: Literal["completed", "abandoned", "duplicate", "superseded", "mistaken"] = Field(
        description="目标结束的方式：completed(完成), abandoned(放弃), duplicate(重复), superseded(被替代), mistaken(误建)。",
    )


@tool(
    name="goal_resolve",
    description="结束一个指定的活跃目标并说明结束方式（完成、放弃、重复、被替代或误建）。",
    args_model=GoalResolveArgs,
)
def execute(args: GoalResolveArgs) -> dict:
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
        "action": "resolve",
        "resolved": {
            "goal_id": resolved["goal_id"],
            "goal": resolved.get("goal") or resolved.get("title") or "",
            "status": resolved["status"],
            "resolution": resolved["resolution"],
        },
        "total": len(_goals.get_all()),
    }


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
