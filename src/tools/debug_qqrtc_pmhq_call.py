"""Debug helper for sending raw PMHQ calls through the QQRTC bridge plugin."""

from __future__ import annotations

import asyncio
from typing import Any

from tools._async_bridge import run_coroutine_sync

DECLARATION: dict = {
    "name": "debug_qqrtc_pmhq_call",
    "description": (
        "调试用：通过 QQRTC 插件向 LLBot/PMHQ 发送原始 call 请求，用于逆向定位 QQ 语音电话的拨打/挂断接口。"
        "只有在明确知道候选 PMHQ func 和 args 时使用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "func": {
                "type": "string",
                "description": "PMHQ func，例如 wrapperSession.getMsgService().sendMsg。",
            },
            "args": {
                "type": "array",
                "description": "PMHQ func 参数数组。",
                "items": {},
            },
            "plugin_id": {
                "type": "string",
                "description": "可选，指定使用哪个在线 QQRTC 插件。",
            },
            "timeout_ms": {
                "type": "integer",
                "description": "PMHQ 调用超时毫秒数，默认 10000。",
            },
        },
        "required": ["func", "args"],
    },
}

ALWAYS_AVAILABLE: bool = False


def condition(config: dict) -> bool:
    return bool((config.get("qqrtc") or {}).get("enabled", False))


def execute(
    func: str,
    args: list[Any] | None = None,
    plugin_id: str = "",
    timeout_ms: int = 10000,
    **_kwargs: Any,
) -> dict[str, Any]:
    import app_state

    server = app_state.qqrtc_server
    if server is None:
        return {"error": "QQRTC 服务端未启用"}

    loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return {"error": "主事件循环不可用"}

    func = str(func or "").strip()
    if not func:
        return {"error": "func 不能为空"}
    if args is None:
        args = []
    if not isinstance(args, list):
        return {"error": "args 必须是数组"}

    timeout_ms = max(1, int(timeout_ms or 10000))
    timeout_seconds = max(1.0, timeout_ms / 1000.0)
    try:
        return run_coroutine_sync(
            server.dispatch_command(
                "raw_call",
                {"func": func, "args": args, "timeout_ms": timeout_ms},
                plugin_id=str(plugin_id or "").strip() or None,
                timeout=timeout_seconds + 2,
            ),
            loop,
            timeout=timeout_seconds + 4,
        )
    except Exception as exc:
        return {"error": f"PMHQ raw call 发送失败: {exc}"}
