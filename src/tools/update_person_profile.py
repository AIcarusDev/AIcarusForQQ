"""update_person_profile.py — 更新实体侧写工具

填充 entity_profiles 表中的主观侧写字段（sex/age/area/notes）。
这是模型的主观推断，不是事实断言，用于辅助分类和记忆关联。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "name": "update_person_profile",
    "description": (
        "更新对某个人的认识（推断性别、年龄、地区、备注等）。"
        "这是你的主观推断，不是事实断言。只填写你有一定把握的字段，不确定的留空。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform_id": {
                "type": "string",
                "description": "对方的 QQ 号（数字字符串）。",
            },
            "updates": {
                "type": "object",
                "description": "要更新的字段，只填写你有把握的，其余留空。",
                "properties": {
                    "sex": {
                        "type": "string",
                        "enum": ["male", "female", "unknown"],
                        "description": "推断性别。",
                    },
                    "age": {
                        "type": "integer",
                        "description": "推断年龄（整数，例如 25）。",
                    },
                    "area": {
                        "type": "string",
                        "description": "推断地区，自然语言描述，例如「上海」、「东南亚某国」。",
                    },
                    "notes": {
                        "type": "string",
                        "description": (
                            "关于这个人的自由文本备注（覆盖写入，非追加）。"
                            "可记录性格、习惯、关系背景等，100 字内。"
                        ),
                    },
                },
                "additionalProperties": False,
            },
            "motivation": {
                "type": "string",
                "description": "为什么做出这些推断，例如「对方自述在上海工作，口音也符合」。",
            },
        },
        "required": ["platform_id", "updates", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        platform_id: str,
        updates: dict,
        motivation: str = "",
        **kwargs,
    ) -> dict:
        import app_state
        from database import update_person_profile as _db_update

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        if not updates:
            return {"error": "updates 字段为空，没有要更新的内容"}

        # 只传入 updates 中实际存在的 key
        allowed = {"sex", "age", "area", "notes"}
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return {"error": f"updates 中没有合法字段，允许的字段: {allowed}"}

        async def _run():
            return await _db_update(
                platform_id=str(platform_id),
                platform="qq",
                **filtered,
            )

        try:
            future = asyncio.run_coroutine_threadsafe(_run(), loop)
            found = future.result(timeout=10)
        except Exception as e:
            return {"error": f"更新失败: {e}"}

        if not found:
            return {
                "ok": False,
                "note": f"未找到 QQ {platform_id} 对应的账号记录（对方可能尚未与 bot 互动过）",
            }
        return {"ok": True, "updated_fields": list(filtered.keys())}

    return execute
