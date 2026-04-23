"""suggest_person_merge.py — 实体泼溅合并建议工具 (Phase 3B)

当模型判断两个实体侧写（EntityProfile）极有可能对应同一个意识个体（多账号、改名等），
调用此工具记录合并建议，供运营者人工审核后决定是否合并。

建议状态流转：pending → confirmed / rejected
已有 pending 建议的同一 pair 重复提交时，会更新 similarity/reason（幂等）。
"""

import asyncio
from typing import Any, Callable

DECLARATION: dict = {
    "name": "suggest_person_merge",
    "description": (
        "提议将两个 EntityProfile 合并为同一个意识个体。"
        "当你发现两个账号极有可能是同一人（换号、多账号、改名后再加群等），"
        "使用此工具记录合并建议，由运营者事后审核。不会立即合并数据。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "profile_id_a": {
                "type": "string",
                "description": "第一个账号的实体侧写 ID（profile_id，通常是 QQ 号）。",
            },
            "profile_id_b": {
                "type": "string",
                "description": "第二个账号的实体侧写 ID，与 profile_id_a 不同。",
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
        "required": ["profile_id_a", "profile_id_b", "similarity", "reason"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        profile_id_a: str,
        profile_id_b: str,
        similarity: float,
        reason: str,
        **kwargs,
    ) -> dict:
        import app_state
        from database import upsert_merge_suggestion

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}

        if profile_id_a == profile_id_b:
            return {"error": "profile_id_a 与 profile_id_b 不能相同"}

        if not (0.0 <= float(similarity) <= 1.0):
            return {"error": "similarity 必须在 0.0～1.0 之间"}

        async def _upsert():
            return await upsert_merge_suggestion(
                profile_id_a=profile_id_a,
                profile_id_b=profile_id_b,
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
