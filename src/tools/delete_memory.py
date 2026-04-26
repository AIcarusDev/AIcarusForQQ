"""delete_memory.py — 软删除一条事件记忆。"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "name": "delete_memory",
    "description": "软删除一条已经被召回展示过的记忆事件（按事件 id）。",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "要删除的事件 id（来自被注入的记忆区块或 recall_memory 返回值）。",
            },
            "reason": {
                "type": "string",
                "description": "为什么要删除这条记忆。",
            },
        },
        "required": ["memory_id", "reason"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(memory_id: str, reason: str = "", **kwargs) -> dict:
        import app_state
        from memory.repo.events import soft_delete_event

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        try:
            eid = int(str(memory_id).strip())
        except (TypeError, ValueError):
            return {"error": f"非法 memory_id: {memory_id!r}"}

        async def _delete() -> bool:
            return await soft_delete_event(eid)

        try:
            future = asyncio.run_coroutine_threadsafe(_delete(), loop)
            ok = future.result(timeout=10)
        except Exception as e:
            return {"error": f"删除失败: {e}"}

        if not ok:
            return {"deleted": False, "note": f"未找到活跃记忆 id={eid}"}
        return {"deleted": True, "id": eid, "reason": reason}

    return execute
