from __future__ import annotations

from pydantic import Field

from project_source import get_default_service
from tools.contract import ToolArgsModel, ToolContract


class ListArgs(ToolArgsModel):
    path: str = Field(default=".", min_length=1, max_length=2048, description="项目相对目录路径。")
    offset: int = Field(default=0, ge=0, description="分页偏移。")
    limit: int = Field(default=100, ge=1, le=200, description="返回条目数量。")


TOOL_CONTRACT = ToolContract(
    name="list",
    description="列出当前项目中一个目录的直属条目。",
    args_model=ListArgs,
)

PARALLEL_SAFE = True
RESULT_CDATA = True


def execute(path: str = ".", offset: int = 0, limit: int = 100):
    return get_default_service().list_directory(path, offset=offset, limit=limit)
