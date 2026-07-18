from __future__ import annotations

from typing import Literal

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract
from workspace.config import DEFAULT_AGENT_HOME

from ._common import list_result, run_on_main_loop


class SearchArgs(ToolArgsModel):
    pattern: str = Field(min_length=1, description="正则表达式；literal=true 时按字面文本搜索。")
    path: str = Field(default=DEFAULT_AGENT_HOME, description="搜索根目录，默认为 /home/agent。")
    glob: str | None = Field(default=None, description="可选文件 glob 过滤。")
    mode: Literal["content", "files_with_matches", "count"] = Field(default="content", description="返回匹配内容、文件名或计数。")
    literal: bool = Field(default=False, description="是否按字面文本搜索。")
    case_sensitive: bool = Field(default=False, description="是否区分大小写。")
    context_before: int = Field(default=0, ge=0, le=20, description="匹配前上下文行数。")
    context_after: int = Field(default=0, ge=0, le=20, description="匹配后上下文行数。")
    multiline: bool = Field(default=False, description="是否允许跨行匹配。")
    offset: int = Field(default=0, ge=0, description="分页偏移，默认 0。")
    limit: int = Field(default=250, ge=1, le=1000, description="返回数量，默认 250，最大 1000。")


TOOL_CONTRACT = ToolContract(
    name="search",
    description="搜索文件内容，支持 glob、上下文、多行和分页。",
    args_model=SearchArgs,
)
REQUIRES_CONTEXT = ["workspace_service", "main_loop"]
PARALLEL_SAFE = True
RESULT_CDATA = True


def make_handler(workspace_service, main_loop):
    def handler(**kwargs):
        async def operation():
            return list_result(await workspace_service.search(**kwargs))
        return run_on_main_loop(operation(), main_loop)
    return handler
