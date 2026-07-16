from __future__ import annotations

from tools.contract import ToolArgsModel, ToolContract

from ._common import preview_result, run_on_main_loop


class PreviewArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="preview",
    description="获取 Agent 电脑 6080 服务在宿主浏览器中的本机回环 URL；随后可用 browser_control 打开该 URL。",
    args_model=PreviewArgs,
)
REQUIRES_CONTEXT = ["workspace_service", "main_loop"]
PARALLEL_SAFE = True


def make_handler(workspace_service, main_loop):
    def handler(**kwargs):
        async def operation():
            return preview_result(await workspace_service.preview())

        return run_on_main_loop(operation(), main_loop)

    return handler
