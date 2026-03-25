"""write_memory.py — 写入一条长期记忆

将重要信息写入模型的长期记忆，重启后持久保留。
需要运行时上下文：session（自动注入会话溯源信息）。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "max_calls_per_response": 3,
    "name": "write_memory",
    "description": (
        "主动的记住某事。"
        "记忆总条数有上限（15条），超出时最旧的将被遗忘。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "要记住的内容，简洁明确，不超过 100 字。",
            },
            "source": {
                "type": "string",
                "description": (
                    "记忆来源的自然语言描述，例如「和小明讨论周报时」、「观察群聊活跃规律时」。"
                    "会话相关信息（群名/群号）由系统自动附加，无需在此重复。"
                ),
            },
            "reason": {
                "type": "string",
                "description": "记住这条信息的动机，例如「对方明确表达了偏好，下次互动需注意」。",
            },
        },
        "required": ["content", "source", "reason"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(content: str, source: str, reason: str, **kwargs) -> dict:
        import app_state
        from llm import memory as _memory
        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用，无法写入记忆"}
        coro = _memory.add_memory(
            content=content,
            source=source,
            reason=reason,
            conv_type=session.conv_type,
            conv_id=session.conv_id,
            conv_name=session.conv_name,
        )
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            memory_id = future.result(timeout=10)
        except Exception as e:
            return {"error": f"写入记忆失败: {e}"}
        return {"ok": True, "memory_id": memory_id, "total": len(_memory.get_all())}

    return execute
