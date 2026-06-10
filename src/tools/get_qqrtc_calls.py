"""Inspect recent QQ realtime-call bridge events."""

from __future__ import annotations

from typing import Any

DECLARATION: dict = {
    "name": "get_qqrtc_calls",
    "description": (
        "查看 QQ 实时语音/通话桥最近捕获到的通话事件、在线插件和活跃会话。"
        "当需要判断当前是否有人来电、是谁、最近通话状态如何时使用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回最近多少条事件，默认 20，最大 100。",
            },
            "session_id": {
                "type": "string",
                "description": "可选，只查看某个 QQRTC session_id 的事件。",
            },
        },
    },
}

ALWAYS_AVAILABLE: bool = False


def condition(config: dict) -> bool:
    return bool((config.get("qqrtc") or {}).get("enabled", False))


def execute(limit: int = 20, session_id: str = "", **_kwargs: Any) -> dict[str, Any]:
    import app_state

    server = app_state.qqrtc_server
    if server is None:
        return {"error": "QQRTC 服务端未启用"}
    limit = max(1, min(int(limit or 20), 100))
    session_id = str(session_id or "").strip()
    return {
        "plugins": server.list_plugins(),
        "active_calls": server.list_active_calls(),
        "events": server.list_events(limit=limit, session_id=session_id or None),
    }
