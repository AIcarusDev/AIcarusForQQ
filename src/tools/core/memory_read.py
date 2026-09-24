"""Read a formal memory or a submitted batch."""
from pydantic import Field
from tools.contract import ToolArgsModel, tool


class MemoryReadArgs(ToolArgsModel):
    id: str = Field(pattern=r"^[MB][0-9]{6,}$", description="M 开头的正式记忆 ID，或 B 开头的批次 ID。")
    history: bool = Field(default=False, description="读取正式记忆时是否附带修订历史。")


PARALLEL_SAFE = True
PARALLEL_KEY = "memory_read"
RESULT_CDATA = True


@tool(name="memory_read", args_model=MemoryReadArgs, description="读取主动记忆全文/历史，或批次的原始内容、认知及整理状态。无需 Computer。")
def execute(args: MemoryReadArgs) -> dict:
    from memory.active import store
    result = store.read(store.default_path(), args.id, history=args.history)
    return {"found": result is not None, "item": result}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
