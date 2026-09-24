"""Read a formal memory or a submitted batch."""
from pydantic import Field
from tools.contract import ToolArgsModel, tool


class MemoryReadArgs(ToolArgsModel):
    id: str = Field(pattern=r"^(?:[MBE][0-9]{6,}|summary:[A-Za-z0-9_:-]+)$", description="搜索返回的记忆 ID，或写入返回的批次 ID。")
    history: bool = Field(default=False, description="读取正式记忆时是否附带修订历史。")


PARALLEL_SAFE = True
PARALLEL_KEY = "memory_read"
RESULT_CDATA = True


@tool(name="memory_read", args_model=MemoryReadArgs, description="按 ID 读取记忆，或查看写入批次的处理结果。")
def execute(args: MemoryReadArgs) -> dict:
    from memory.active import store
    from memory.access import read
    result = read(store.default_path(), args.id, history=args.history)
    return {"found": result is not None, "item": result}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
