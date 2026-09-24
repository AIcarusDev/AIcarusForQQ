"""Explicit long-term memory search, independent of automatic recall."""
from pydantic import Field
from tools.contract import ToolArgsModel, tool


class MemorySearchArgs(ToolArgsModel):
    query: str = Field(min_length=1, description="关键词、名字、原句或记忆 ID。")
    literal: bool = Field(default=False, description="true 时按完整原句字面匹配，否则中文分词匹配。")
    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


PARALLEL_SAFE = True
PARALLEL_KEY = "memory_read"
RESULT_CDATA = True


@tool(name="memory_search", args_model=MemorySearchArgs, description="搜索长期记忆，返回正文和 ID。")
def execute(args: MemorySearchArgs) -> dict:
    from memory.active import store
    from memory.access import search
    items = search(store.default_path(), args.query, limit=args.limit + 1, offset=args.offset, literal=args.literal)
    return {"items": items[:args.limit], "has_more": len(items) > args.limit, "offset": args.offset}


TOOL_CONTRACT = getattr(execute, "__tool_contract__", None)
