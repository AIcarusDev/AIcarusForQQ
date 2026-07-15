from __future__ import annotations

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

from ._common import run_on_main_loop


class ExactEdit(ToolArgsModel):
    old_text: str = Field(min_length=1, description="要替换的原文本，默认必须恰好匹配一次。")
    new_text: str = Field(description="替换后的文本。")
    replace_all: bool = Field(default=False, description="为 true 时替换全部匹配。")


class EditFileArgs(ToolArgsModel):
    path: str = Field(min_length=1, description="此前已通过 read_file 读取的文件路径。")
    edits: list[ExactEdit] = Field(min_length=1, description="按顺序原子执行的精确替换。")


TOOL_CONTRACT = ToolContract(
    name="edit_file",
    description="对已读取且未变化的文本文件执行一组原子精确替换。",
    args_model=EditFileArgs,
)
REQUIRES_CONTEXT = ["workspace_service", "main_loop"]


def make_handler(workspace_service, main_loop):
    def handler(**kwargs):
        edits = [item if isinstance(item, dict) else item.model_dump() for item in kwargs["edits"]]
        return run_on_main_loop(workspace_service.edit_file(kwargs["path"], edits), main_loop)
    return handler
