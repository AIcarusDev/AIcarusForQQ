"""Submit one batch, capturing the current round's cognition as plain text."""
from pydantic import Field, field_validator

from tools.contract import ToolArgsModel, tool
from tools._async_bridge import run_coroutine_sync


class MemoryWriteArgs(ToolArgsModel):
    memories: list[str] = Field(min_length=1, description="本次要记住或修正的多条内容，一起整理。用完整自然语言说明主体、条件和不确定性；修正/撤回可在文字中注明已有记忆 ID。")

    @field_validator("memories")
    @classmethod
    def nonblank(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("记忆不能为空白")
        return values


@tool(name="memory_write", args_model=MemoryWriteArgs, description=(
    "主动保存值得长期记住的内容，也可提出修正或撤回。一次多条按同一批次持久化，后台依次整理。"
    "自动附带本轮认知；无需来源引用或 Computer。返回 batch_id，可用 memory_read 查看处理结果。"
    "saved 表示已保存，尚不表示已整理；不要因 pending 或 processor_ready=false 重复提交。"
))
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
