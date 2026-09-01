from __future__ import annotations

from typing import Literal

from pydantic import Field

from project_source import get_default_service
from tools.contract import ToolArgsModel, ToolContract


class SearchArgs(ToolArgsModel):
    query: str = Field(min_length=1, max_length=1000, description="要查找的字面文本。")
    path: str = Field(default=".", min_length=1, max_length=2048, description="项目相对搜索目录。")
    glob: str | None = Field(default=None, min_length=1, max_length=512, description="可选文件 glob。")
    mode: Literal["content", "path"] = Field(default="content", description="搜索文件正文或路径。")
    case_sensitive: bool = Field(default=False, description="是否区分大小写。")
    offset: int = Field(default=0, ge=0, description="匹配结果分页偏移。")
    limit: int = Field(default=100, ge=1, le=200, description="返回匹配数量。")


TOOL_CONTRACT = ToolContract(
    name="search",
    description="在当前项目的静态文本文件路径或正文中搜索匹配项。",
    args_model=SearchArgs,
)

PARALLEL_SAFE = True
RESULT_CDATA = True


def execute(
    query: str,
    path: str = ".",
    glob: str | None = None,
    mode: Literal["content", "path"] = "content",
    case_sensitive: bool = False,
    offset: int = 0,
    limit: int = 100,
):
    return get_default_service().search(
        query,
        path=path,
        glob=glob,
        mode=mode,
        case_sensitive=case_sensitive,
        offset=offset,
        limit=limit,
    )
