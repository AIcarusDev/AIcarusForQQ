"""delete_memory.py — 删除一条长期记忆

从长期记忆中删除指定 ID 的条目。记忆 ID 可在 system prompt 的 <memory> 块中找到。
"""

import asyncio

# build_tools() 用此字段获取工具名；实际 schema 由 get_declaration() 动态生成
DECLARATION: dict = {
    "name": "delete_memory",
}


def condition(config: dict) -> bool:
    """仅在存在可删除的记忆时才暴露此工具。"""
    from llm import memory as _memory
    return len(_memory.get_all()) > 0


def get_declaration() -> dict:
    """动态生成工具 schema：memory_id 枚举为当前实际存在的 ID 列表。"""
    from llm import memory as _memory
    ids = [m["memory_id"] for m in _memory.get_all()]
    return {
        "max_calls_per_response": 3,
        "name": "delete_memory",
        "description": (
            "主动忘掉你的一条记忆。"
            "在记忆内容过时、不再准确或不再有价值时使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "enum": ids,
                    "description": "要删除的记忆 ID，只能从当前列出的值中选取。",
                },
                "reason": {
                    "type": "string",
                    "description": "删除此记忆的原因。",
                },
            },
            "required": ["memory_id"],
        },
    }


def execute(memory_id: str, reason: str = "", **kwargs) -> dict:
    import app_state
    from llm import memory as _memory
    loop: asyncio.AbstractEventLoop | None = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"error": "主事件循环不可用，无法删除记忆"}
    coro = _memory.remove_memory(memory_id)
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        found = future.result(timeout=10)
    except Exception as e:
        return {"error": f"删除记忆失败: {e}"}
    if found:
        return {"ok": True, "message": f"记忆 {memory_id} 已删除", "total": len(_memory.get_all())}
    return {"ok": False, "message": f"未找到记忆 {memory_id}，可能已被删除"}
