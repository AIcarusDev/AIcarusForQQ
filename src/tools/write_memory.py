"""write_memory.py — 写入一条长期记忆

将重要信息写入模型的长期记忆，重启后持久保留。
需要运行时上下文：session（自动注入会话溯源信息）。
"""

import asyncio
from typing import Any, Callable

# build_tools() 用此字段获取工具名；实际 schema 由 get_declaration() 动态生成
DECLARATION: dict = {
    "name": "write_memory",
}


def get_declaration() -> dict:
    """动态生成工具 schema：描述中包含当前记忆用量。"""
    from llm.prompt import memory as _memory
    current = len(_memory.get_all())
    max_entries = _memory.get_max_entries()
    return {
        "name": "write_memory",
        "description": (
            "主动的记住某事。"
            f"记忆总条数有上限（当前 {current}/{max_entries}），超出时最旧的将被遗忘。\n"
            "可用简洁自由文本（content），也可用结构化三元组（predicate + object_text）精确表达关系。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "自由文本记忆内容，简洁明确，不超过 100 字。"
                        "与 predicate/object_text 二选一：优先使用结构化三元组，"
                        "无法结构化时才用此字段。"
                    ),
                },
                "predicate": {
                    "type": "string",
                    "description": (
                        "关系/属性描述，即『谁』的「什么」，例如「喜欢」、「不喜欢」、「职业是」、「曾经说过」。"
                        "与 object_text 配合使用，二者同时填写时忽略 content 字段。"
                    ),
                },
                "object_text": {
                    "type": "string",
                    "description": (
                        "关系的宾语/具体内容，例如「打乒乓球」、「程序员」、「不喜欢吃香菜」。"
                        "与 predicate 配合使用。"
                    ),
                },
                "source": {
                    "type": "string",
                    "description": (
                        "记忆来源的自然语言描述，例如「和小明讨论周报时」、「观察群聊活跃规律时」。"
                        "会话相关信息（群名/群号）由系统自动附加，无需在此重复。"
                    ),
                },
                "motivation": {
                    "type": "string",
                    "description": "记住这条信息的动机，例如「对方明确表达了偏好，下次互动需注意」。",
                },
            },
            "required": ["source", "motivation"],
        },
    }

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        source: str,
        motivation: str = "",
        content: str = "",
        predicate: str = "",
        object_text: str = "",
        **kwargs,
    ) -> dict:
        import app_state
        from llm.prompt import memory as _memory
        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用，无法写入记忆"}

        # 结构化三元组优先；回退到自由文本 content
        if predicate and object_text:
            final_predicate = predicate
            final_content = object_text
        elif content:
            final_predicate = "[note]"
            final_content = content
        else:
            return {"error": "content 或 (predicate + object_text) 至少填写一项"}

        # 自动从最近一条用户消息推导 subject，避免要求模型手动填写（幻觉风险）
        sender_id = ""
        for m in reversed(session.context_messages):
            if m.get("role") == "user" and m.get("sender_id"):
                sender_id = str(m["sender_id"])
                break
        subject = f"User:qq_{sender_id}" if sender_id else "Self"

        coro = _memory.add_memory(
            content=final_content,
            predicate=final_predicate,
            source=source,
            reason=motivation,
            conv_type=session.conv_type,
            conv_id=session.conv_id,
            conv_name=session.conv_name,
            subject=subject,
        )
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            triple_id = future.result(timeout=10)
        except Exception as e:
            return {"error": f"写入记忆失败: {e}"}
        return {"ok": True, "memory_id": str(triple_id), "total": len(_memory.get_all())}

    return execute
