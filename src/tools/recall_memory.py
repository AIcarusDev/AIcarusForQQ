"""recall_memory.py — 主动回忆工具（事件层 + FTS5）

走 MemoryEvents 三路融合召回：meta + 实体边 + FTS5(query)。
返回 Top N 条事件作为 tool_result 注入本轮 context。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "name": "recall_memory",
    "description": (
        "主动回忆某件事。当你直觉上觉得记得某事但细节模糊时使用。"
        "输入联想词或相关概念，系统会用 FTS5 全文检索 + 实体边在事件图中召回。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词（联想词、近义词、相关概念皆可，会做中文分词）。",
            },
            "motivation": {
                "type": "string",
                "description": "为什么要主动回忆这件事。",
            },
        },
        "required": ["query", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(query: str, motivation: str = "", **kwargs) -> dict:
        import app_state
        from memory import load_events_for_recall

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        # 推导 sender_id 与 context_scope
        sender_id = ""
        for m in reversed(session.context_messages):
            if m.get("role") == "user" and m.get("sender_id"):
                sender_id = str(m["sender_id"])
                break
        sender_entity = f"User:qq_{sender_id}" if sender_id else ""

        conv_type = getattr(session, "conv_type", "") or ""
        conv_id = getattr(session, "conv_id", "") or ""
        context_scope = f"{conv_type}:qq_{conv_id}" if conv_type and conv_id else ""

        kw = (query or "").strip()

        async def _recall() -> list[dict]:
            return await load_events_for_recall(
                sender_entity=sender_entity,
                context_scope=context_scope,
                limit=10,
                query=kw,
            )

        try:
            future = asyncio.run_coroutine_threadsafe(_recall(), loop)
            events = future.result(timeout=10)
        except Exception as e:
            return {"error": f"召回失败: {e}"}

        if not events:
            return {"found": 0, "memories": [], "note": "未找到相关记忆"}

        memories = []
        for e in events:
            memories.append({
                "id": e.get("event_id"),
                "summary": e.get("summary", ""),
                "event_type": e.get("event_type", ""),
                "polarity": e.get("polarity", ""),
                "modality": e.get("modality", ""),
                "ctx": e.get("context_type", ""),
                "confidence": round(float(e.get("confidence") or 0.0), 2),
            })
        return {"found": len(memories), "memories": memories}

    return execute
