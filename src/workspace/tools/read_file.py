from __future__ import annotations

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

from ._common import read_result, run_on_main_loop


class ReadFileArgs(ToolArgsModel):
    path: str = Field(min_length=1, description="要读取的 UTF-8 文本文件路径。")
    start_line: int = Field(default=1, ge=1, description="起始行号，从 1 开始。")
    line_count: int = Field(default=2000, ge=1, le=2000, description="最多读取的行数，默认 2000。")


TOOL_CONTRACT = ToolContract(
    name="read_file",
    description="按行读取 UTF-8 文本文件，返回带行号内容和继续读取位置；单次原文超过 5000 字符时不返回正文，需缩小 start_line 和 line_count。",
    args_model=ReadFileArgs,
)
REQUIRES_CONTEXT = ["workspace_service", "main_loop"]
PARALLEL_SAFE = True
RESULT_CDATA = True


def make_handler(workspace_service, main_loop):
    def handler(**kwargs):
        async def operation():
            return read_result(await workspace_service.read_file(**kwargs))
        return run_on_main_loop(operation(), main_loop)
    return handler
