"""Expand a memory preview to its complete current text."""
from pydantic import Field
from tools.contract import ToolArgsModel, tool


class MemoryReadArgs(ToolArgsModel):
    id: str = Field(pattern=r"^(?:[ME][0-9]{6,}|summary:[A-Za-z0-9_:-]+)$", description="被截断记忆的 ID。")


PARALLEL_SAFE = True
PARALLEL_KEY = "memory_read"
RESULT_CDATA = True


@tool(name="memory_read", args_model=MemoryReadArgs, description="按 ID 读取被截断记忆的完整正文。")
def execute(args: MemoryReadArgs) -> dict:
    from memory.active import store
    from memory.access import read
    result = read(store.default_path(), args.id)
    return {"found": result is not None, "item": result}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
