"""Send a QQ realtime-call control command through the bridge plugin."""

from __future__ import annotations

import asyncio
from typing import Any

from tools._async_bridge import run_coroutine_sync

DECLARATION: dict = {
    "name": "control_qqrtc_call",
    "description": (
        "通过 QQRTC 插件控制 QQ 实时语音/通话。"
        "用于让机器人主动拨打、接听、拒绝或挂断 QQ 语音通话。"
        "项目侧只负责下发控制命令；插件侧需要实现对应 LLBot/PMHQ 调用后命令才会真正生效。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call", "accept", "join", "hangup", "reject"],
                "description": "要执行的通话动作。",
            },
            "session_id": {
                "type": "string",
                "description": "QQRTC session_id。挂断、接听、拒绝已有通话时使用，通常可从 get_qqrtc_calls 获取。",
            },
            "peer_id": {
                "type": "string",
                "description": "目标 QQ 号。主动拨打电话时使用。",
                "x-coerce-integer": True,
            },
            "plugin_id": {
                "type": "string",
                "description": "可选，指定使用哪个在线 QQRTC 插件。",
            },
        },
        "required": ["action"],
    },
}

ALWAYS_AVAILABLE: bool = False


def condition(config: dict) -> bool:
    return bool((config.get("qqrtc") or {}).get("enabled", False))


def execute(
    action: str,
    session_id: str = "",
    peer_id: str = "",
    plugin_id: str = "",
    **_kwargs: Any,
) -> dict[str, Any]:
    import app_state

    server = app_state.qqrtc_server
    if server is None:
        return {"error": "QQRTC 服务端未启用"}

    loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
    if loop is None or not loop.is_running():
        return {"error": "主事件循环不可用"}

    action = str(action or "").strip()
    session_id = str(session_id or "").strip()
    peer_id = str(peer_id or "").strip()
    if action == "call":
        if not peer_id:
            return {"error": "主动拨打电话需要 peer_id"}
    elif not session_id:
        return {"error": f"{action} 需要 session_id"}

    try:
        result = run_coroutine_sync(
            server.dispatch_command(
                action,
                {"session_id": session_id, "peer_id": peer_id},
                plugin_id=str(plugin_id or "").strip() or None,
                timeout=float((app_state.qqrtc_cfg or {}).get("command_timeout", 10)),
            ),
            loop,
            timeout=float((app_state.qqrtc_cfg or {}).get("command_timeout", 10)) + 2,
        )
    except Exception as exc:
        return {"error": f"QQRTC 命令发送失败: {exc}"}
    return result
