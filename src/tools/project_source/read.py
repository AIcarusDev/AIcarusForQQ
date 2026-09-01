from __future__ import annotations

from pydantic import Field

from project_source import get_default_service
from tools.contract import ToolArgsModel, ToolContract


class ReadArgs(ToolArgsModel):
    path: str = Field(min_length=1, max_length=2048, description="项目相对文件路径。")
    start_line: int = Field(default=1, ge=1, description="首次读取的起始行号。")
    line_count: int = Field(default=200, ge=1, le=1000, description="本次最多读取的行数。")
    cursor: str | None = Field(default=None, min_length=1, max_length=4096, description="上次读取返回的继续位置。")


TOOL_CONTRACT = ToolContract(
    name="read",
    description="按行读取当前项目中的一个静态文本文件，并返回继续读取位置。",
    args_model=ReadArgs,
)

PARALLEL_SAFE = True
RESULT_CDATA = True


def execute(
    path: str,
    start_line: int = 1,
    line_count: int = 200,
    cursor: str | None = None,
):
    return get_default_service().read_file(
        path,
        start_line=start_line,
        line_count=line_count,
        cursor=cursor,
    )
