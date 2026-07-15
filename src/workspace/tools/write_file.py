from __future__ import annotations

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

from ._common import run_on_main_loop


class WriteFileArgs(ToolArgsModel):
    path: str = Field(min_length=1, description="要写入的文件路径。")
    content: str = Field(description="完整 UTF-8 文件内容。")
    create_parents: bool = Field(default=False, description="是否创建缺失的父目录。")


TOOL_CONTRACT = ToolContract(
    name="write_file",
    description="原子创建或完整写入文本文件；覆盖已有文件前需要完整读取。",
    args_model=WriteFileArgs,
)
REQUIRES_CONTEXT = ["workspace_service", "main_loop"]


def make_handler(workspace_service, main_loop):
    def handler(**kwargs):
        return run_on_main_loop(workspace_service.write_file(**kwargs), main_loop)
    return handler
