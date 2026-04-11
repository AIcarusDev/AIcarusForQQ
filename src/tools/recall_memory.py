"""recall_memory.py — 主动回忆工具

当模型直觉上觉得记得某件事但细节模糊时，可主动调用此工具。
走完整 FTS5 双通道召回流水线，返回 Top N 条相关记忆作为 tool_result 注入本轮 context。
与 system prompt 被动注入的区别：此工具按需触发，结果对模型可见但不修改 system prompt。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "name": "recall_memory",
    "description": (
        "主动回忆某件事。当你直觉上觉得记得某事但细节模糊时使用。"
        "输入联想词或相关概念即可，不必精确，系统会进行 FTS5 全文检索与语义召回。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "检索关键词，可以是联想词、近义词、相关概念，"
                    "例如「小明 食物偏好」、「星际争霸 游戏喜好」。"
                ),
            },
            "motivation": {
                "type": "string",
                "description": "为什么要主动回忆这件事，例如「用户问起了我们之前聊过的话题」。",
            },
        },
        "required": ["query", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(query: str, motivation: str = "", **kwargs) -> dict:
        import app_state
        from llm.prompt.memory import recall_memories
        from llm.memory_tokenizer import build_fts_query

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        # 推导 sender_id（与被动注入一致）
        sender_id = ""
        for m in reversed(session.context_messages):
            if m.get("role") == "user" and m.get("sender_id"):
                sender_id = str(m["sender_id"])
                break

        # 用 query 原文走分词 + 双通道召回（传入 config，使用配置超参）
        memory_cfg = getattr(app_state, "config", {}).get("memory", {})

        async def _recall():
            return await recall_memories(
                message_text=query,
                sender_id=sender_id,
                config=memory_cfg,
            )

        try:
            future = asyncio.run_coroutine_threadsafe(_recall(), loop)
            results = future.result(timeout=10)
        except Exception as e:
            return {"error": f"召回失败: {e}"}

        if not results:
            return {"found": 0, "memories": [], "note": "未找到相关记忆"}

        memories = []
        for r in results:
            entry: dict = {
                "id": r.get("id"),
                "subject": r.get("subject", ""),
                "predicate": r.get("predicate", ""),
                "content": r.get("object_text", ""),
                "confidence": round(r.get("confidence", 0.0), 2),
                "source": r.get("source", ""),
                "reason": r.get("reason", ""),
            }
            memories.append(entry)

        return {"found": len(memories), "memories": memories}

    return execute
