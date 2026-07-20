"""Read a downloaded attachment without executing it."""

from __future__ import annotations

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

from ._common import run_on_main_loop


class ReadArgs(ToolArgsModel):
    attachment_id: str = Field(min_length=1, description="download 返回的 attachment_id。")
    offset: int = Field(default=0, ge=0, description="文本读取的起始字符位置。")
    limit: int = Field(default=5000, ge=1, le=5000, description="本次最多返回的文本字符数。")
    page_start: int = Field(default=1, ge=1, description="PDF 起始页码，从 1 开始。")
    page_count: int = Field(default=5, ge=1, le=10, description="PDF 本次读取页数。")


TOOL_CONTRACT = ToolContract(
    name="read",
    description=("只读查看已下载附件。文本支持 offset 分页，PDF 支持按页提取；"
                 "图片返回可供视觉工具使用的 image_ref 或基础元数据；二进制文件只返回元数据，绝不执行。"),
    args_model=ReadArgs,
)
REQUIRES_CONTEXT = ["attachment_service", "main_loop"]
RESULT_CDATA = True


def make_handler(attachment_service, main_loop):
    async def execute_async(**kwargs):
        return await attachment_service.read(**kwargs)

    def handler(**kwargs):
        return run_on_main_loop(execute_async(**kwargs), main_loop)

    return handler
