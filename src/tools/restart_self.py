"""restart_self.py - schedule a graceful self restart."""

from __future__ import annotations

import logging

from runtime import core_restart

logger = logging.getLogger("AICQ.tools.restart_self")

ALWAYS_AVAILABLE: bool = False
REQUIRES_CONTEXT: list[str] = ["session"]

DECLARATION: dict = {
    "name": "restart_self",
    "description": (
        "重启自己的进程。"
        "只在运行时配置、依赖或代码已变化且确实需要重新加载时使用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def make_handler(session):
    def execute(**kwargs) -> dict:
        try:
            import app_state

            focus_key = getattr(app_state, "current_focus", None)
            if not focus_key:
                return {
                    "ok": False,
                    "error": "当前没有可恢复的 QQ 焦点，无法安排重启后自动激活。",
                }

            result = core_restart.request_restart(
                focus_key=str(focus_key),
                requested_by="tool:restart_self",
            )
            result["deferred"] = True
            return result
        except Exception as exc:
            logger.warning("[restart_self] scheduling failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"安排自身重启失败: {exc}"}

    return execute
