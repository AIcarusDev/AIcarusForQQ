"""Read supported QQ files, optionally downloading a current message first."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Literal, Union

from pydantic import Field, RootModel, model_validator

from platforms.qq.files.service import QQFileError, get_qq_file_service
from platforms.qq.session_context import ensure_session_provider
from tools._async_bridge import run_coroutine_sync
from tools.contract import ToolArgsModel, ToolContract


class PathReadSource(ToolArgsModel):
    path: str = Field(min_length=1, description="要读取的绝对 Linux 文件路径，必须位于当前 QQ 账号的 file 根目录内。")


class MessageReadSource(ToolArgsModel):
    message_id: str = Field(
        min_length=1,
        description="当前 QQ 会话中的文件消息 ID。",
        json_schema_extra={"x-coerce-integer": True},
    )


ReadSource = Union[PathReadSource, MessageReadSource]


class TextLinesSelection(ToolArgsModel):
    type: Literal["text_lines"] = Field(description="按 UTF-8 文本行定位。")
    start_line: int = Field(default=1, ge=1, description="起始行号，从 1 开始，默认 1。")
    end_line: int | None = Field(default=None, ge=1, description="可选结束行号（包含），必须不小于 start_line。")

    @model_validator(mode="after")
    def validate_end(self):
        if self.end_line is not None and self.end_line < self.start_line:
            raise ValueError("end_line 必须不小于 start_line")
        return self


class PdfPagesSelection(ToolArgsModel):
    type: Literal["pdf_pages"] = Field(description="按 PDF 页码定位。")
    start_page: int = Field(default=1, ge=1, description="起始页码，从 1 开始，默认 1。")
    end_page: int | None = Field(default=None, ge=1, description="可选结束页码（包含），必须不小于 start_page。")

    @model_validator(mode="after")
    def validate_end(self):
        if self.end_page is not None and self.end_page < self.start_page:
            raise ValueError("end_page 必须不小于 start_page")
        return self


class DocxBlocksSelection(ToolArgsModel):
    type: Literal["docx_blocks"] = Field(description="按 DOCX 正文块定位；标题、段落和表格各占一个块。")
    start_block: int = Field(default=1, ge=1, description="起始块编号，从 1 开始，默认 1。")
    end_block: int | None = Field(default=None, ge=1, description="可选结束块编号（包含），必须不小于 start_block。")

    @model_validator(mode="after")
    def validate_end(self):
        if self.end_block is not None and self.end_block < self.start_block:
            raise ValueError("end_block 必须不小于 start_block")
        return self


class XlsxRangeSelection(ToolArgsModel):
    type: Literal["xlsx_range"] = Field(description="按 XLSX 工作表和 A1 单元格区域定位。")
    sheet: str = Field(min_length=1, description="工作表名称，精确匹配。")
    cell_range: str | None = Field(default=None, description="可选 A1 区域，例如 A1 或 A1:F50；省略时从该表已用区域开头读取。")


class PptxSlidesSelection(ToolArgsModel):
    type: Literal["pptx_slides"] = Field(description="按 PPTX 幻灯片编号定位。")
    start_slide: int = Field(default=1, ge=1, description="起始幻灯片编号，从 1 开始，默认 1。")
    end_slide: int | None = Field(default=None, ge=1, description="可选结束幻灯片编号（包含），必须不小于 start_slide。")

    @model_validator(mode="after")
    def validate_end(self):
        if self.end_slide is not None and self.end_slide < self.start_slide:
            raise ValueError("end_slide 必须不小于 start_slide")
        return self


ReadSelection = Annotated[
    Union[TextLinesSelection, PdfPagesSelection, DocxBlocksSelection, XlsxRangeSelection, PptxSlidesSelection],
    Field(discriminator="type"),
]


class ReadStartArgs(ToolArgsModel):
    source: ReadSource = Field(description="从现有本地路径或当前会话文件消息开始读取。")
    selection: ReadSelection | None = Field(default=None, description="可选起点或范围；省略时从文档开头顺序读取。")

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        source = schema.get("properties", {}).get("source", {})
        if "anyOf" in source:
            source["oneOf"] = source.pop("anyOf")
        return schema


class ReadContinueArgs(ToolArgsModel):
    cursor: str = Field(min_length=1, max_length=2048, description="上次 read 返回的 next_cursor；使用时不能传 source 或 selection。")


class ReadArgs(RootModel[Union[ReadStartArgs, ReadContinueArgs]]):
    pass


TOOL_CONTRACT = ToolContract(
    name="read",
    description="读取当前 QQ 账号文件根目录中的 UTF-8 文本、PDF、DOCX、XLSX 或 PPTX。",
    args_model=ReadArgs,
)

REQUIRES_CONTEXT = ["qq_client", "qq_session_provider", "workspace_service"]
RESULT_CDATA = True


def _plain(value: Any) -> Any:
    return value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value


def make_handler(qq_client: Any, qq_session_provider: Any, workspace_service: Any):
    session_provider = ensure_session_provider(qq_session_provider)
    service = get_qq_file_service(qq_client, workspace_service)

    def execute(**kwargs: Any) -> dict[str, Any]:
        loop: asyncio.AbstractEventLoop | None = getattr(qq_client, "_loop", None)
        if loop is None or not loop.is_running():
            return {"ok": False, "error": {"code": "runtime_unavailable", "message": "主事件循环当前不可用", "retryable": True, "path": None, "details": None}}
        source = _plain(kwargs.get("source"))
        selection = _plain(kwargs.get("selection"))
        try:
            return run_coroutine_sync(
                service.read(
                    source=source,
                    selection=selection,
                    cursor=str(kwargs["cursor"]) if kwargs.get("cursor") else None,
                    session=session_provider(),
                ),
                loop,
                timeout=150.0,
            )
        except QQFileError as exc:
            details = dict(exc.details)
            path = details.pop("path", None)
            return {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": str(exc),
                    "retryable": exc.retryable,
                    "path": path,
                    "details": details or None,
                },
            }
        except Exception:
            return {"ok": False, "error": {"code": "internal_error", "message": "QQ 文件读取失败", "retryable": True, "path": None, "details": None}}

    return execute
