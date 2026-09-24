"""Submit one batch, capturing the current round's cognition as plain text."""
from pydantic import Field, field_validator

from tools.contract import ToolArgsModel, tool
from tools._async_bridge import run_coroutine_sync


class MemoryWriteArgs(ToolArgsModel):
    memories: list[str] = Field(min_length=1, description="要保存、新增或修正的记忆内容。")

    @field_validator("memories")
    @classmethod
    def nonblank(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("记忆不能为空白")
        return values


@tool(name="memory_write", args_model=MemoryWriteArgs, description="提交长期记忆的新增或修正，返回批次 ID。")
def execute(args: MemoryWriteArgs) -> dict:
    import app_state
    from llm.core.round_context import get_current_inner_state
    from memory.active.workflow import submit

    state = get_current_inner_state()
    cognition = str(state.get("cognition") or state.get("think") or "")
    if not cognition.strip():
        return {"saved": False, "error": "current_cognition_unavailable"}
    loop = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"saved": False, "error": "main_loop_unavailable"}
    return run_coroutine_sync(submit(cognition, args.memories), loop)


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
