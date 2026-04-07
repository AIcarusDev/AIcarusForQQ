"""suggest_person_merge.py — 实体泼溅合并建议工具 (Phase 3B)

当模型判断两个 person_id 极有可能是同一个人（多账号、改名等），
调用此工具记录合并建议，供运营者人工审核后决定是否合并。

建议状态流转：pending → confirmed / rejected
已有 pending 建议的同一 pair 重复提交时，会更新 similarity/reason（幂等）。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "name": "suggest_person_merge",
    "description": (
        "提议将两个 person_id 合并为同一个人。"
        "当你发现两个账号极有可能是同一人（换号、多账号、改名后再加群等），"
        "使用此工具记录合并建议，由运营者事后审核。不会立即合并数据。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "person_id_a": {
                "type": "string",
                "description": "第一个账号的 person_id（通常是 QQ 号）。",
            },
            "person_id_b": {
                "type": "string",
                "description": "第二个账号的 person_id，与 person_id_a 不同。",
            },
            "similarity": {
                "type": "number",
                "description": (
                    "你对「两者是同一人」的把握程度，0.0～1.0。"
                    "0.9 以上才建议填写，否则不要使用这个工具。"
                ),
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "reason": {
                "type": "string",
                "description": (
                    "判断依据，例如「昵称相同且说话风格高度一致」、"
                    "「自述换号且主动告知」。50 字内。"
                ),
            },
        },
        "required": ["person_id_a", "person_id_b", "similarity", "reason"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        person_id_a: str,
        person_id_b: str,
        similarity: float,
        reason: str,
        **kwargs,
    ) -> dict:
        import app_state
        from database import upsert_merge_suggestion

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        if person_id_a == person_id_b:
            return {"error": "person_id_a 与 person_id_b 不能相同"}

        if not (0.0 <= float(similarity) <= 1.0):
            return {"error": "similarity 必须在 0.0～1.0 之间"}

        async def _upsert():
            return await upsert_merge_suggestion(
                person_id_a=person_id_a,
                person_id_b=person_id_b,
                similarity=float(similarity),
                reason=str(reason),
            )

        try:
            future = asyncio.run_coroutine_threadsafe(_upsert(), loop)
            suggestion_id = future.result(timeout=10)
        except Exception as e:
            return {"error": f"写入合并建议失败: {e}"}

        return {
            "suggestion_id": suggestion_id,
            "status": "pending",
            "note": "合并建议已记录，等待人工审核",
        }

    return execute
