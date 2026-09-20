"""Codex-compatible checklist updates, persisted globally by Core."""

from typing import Literal

from pydantic import Field

from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, tool


class PlanItem(ToolArgsModel):
    step: str = Field(description="Task step text.")
    status: Literal["pending", "in_progress", "completed"] = Field(description="Step status.")


class UpdatePlanArgs(ToolArgsModel):
    explanation: str | None = Field(default=None, description="Optional explanation for this plan update.")
    plan: list[PlanItem] = Field(description="The complete ordered list of steps; an empty list clears it.")


@tool(
    name="update_plan",
    description=(
        "Updates the task plan. Provide an optional explanation and the complete list of plan items, "
        "each with a step and status. At most one step can be in_progress at a time. "
        "The list is shared globally, persists across restarts, and appears in the preset container."
    ),
    args_model=UpdatePlanArgs,
)
def execute(args: UpdatePlanArgs) -> dict:
    import app_state
    from llm.prompt import todo

    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"ok": False, "error": "主事件循环不可用，无法更新待办"}
    try:
        run_coroutine_sync(
            todo.update_plan([item.model_dump() for item in args.plan], args.explanation),
            loop,
        )
    except Exception as exc:
        return {"ok": False, "error": f"更新待办失败: {exc}"}
    return {"ok": True, "message": "Plan updated"}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
