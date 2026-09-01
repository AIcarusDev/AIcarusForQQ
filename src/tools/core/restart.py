"""restart - schedule a graceful self restart."""

from __future__ import annotations

import logging

from platforms.focus import current_focus_key
from runtime import core_restart
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools.restart")

REQUIRES_CONTEXT: list[str] = ["session"]


class RestartArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="restart",
    description=(
        "重启自己的进程。"
        "在运行时配置、依赖或代码已变化且确实需要重新加载时可使用。"
        "注意：成功的重启操作可能会导致一些平台（例如 qq）的框架暂时断开连接、相关功能不可用；若出现此类情况，等待对应框架自动重连即可。"
    ),
    args_model=RestartArgs,
)


def make_handler(session):
    def execute(**kwargs) -> dict:
        try:
            import app_state

            focus_key = current_focus_key(getattr(app_state, "current_focus", None))
            if not focus_key:
                return {
                    "ok": False,
                    "error": "当前没有可恢复的 QQ 焦点，无法安排重启后自动激活。",
                }

            result = core_restart.request_restart(
                focus_key=focus_key,
                requested_by="tool:restart",
            )
            result["deferred"] = True
            return result
        except Exception as exc:
            logger.warning("[restart] scheduling failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"安排自身重启失败: {exc}"}

    return execute
