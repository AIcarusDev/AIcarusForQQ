from __future__ import annotations

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract
from workspace.config import DEFAULT_AGENT_HOME

from ._common import list_result, run_on_main_loop


class FindFilesArgs(ToolArgsModel):
    pattern: str = Field(min_length=1, description="文件 glob，例如 **/*.py。")
    path: str = Field(default=DEFAULT_AGENT_HOME, description="搜索根目录，默为是 /home/agent。")
    offset: int = Field(default=0, ge=0, description="分页偏移，默认 0。")
    limit: int = Field(default=100, ge=1, le=500, description="返回数量，默认 100，最大 500。")


TOOL_CONTRACT = ToolContract(
    name="find_files",
    description="按 glob 查找文件，稳定排序并分页返回路径。",
    args_model=FindFilesArgs,
)
REQUIRES_CONTEXT = ["workspace_service", "main_loop"]
PARALLEL_SAFE = True
RESULT_CDATA = True


def make_handler(workspace_service, main_loop):
    def handler(**kwargs):
        async def operation():
            return list_result(await workspace_service.find_files(**kwargs))
        return run_on_main_loop(operation(), main_loop)
    return handler
