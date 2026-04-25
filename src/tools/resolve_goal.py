"""resolve_goal.py — 结束活跃目标"""

import asyncio
from typing import Any

from llm.prompt.goals import VALID_RESOLUTIONS

DECLARATION: dict = {
    "name": "resolve_goal",
}


def condition(config: dict) -> bool:
    from llm.prompt import goals as _goals

    return len(_goals.get_all()) > 0


def _normalize_goal_ids(
    goal_ids: list[Any] | str | None,
    goal_id: str | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    raw_ids: list[Any] = []
    if isinstance(goal_ids, list) and goal_ids:
        raw_ids.extend(goal_ids)
    elif isinstance(goal_ids, str) and goal_ids.strip():
        raw_ids.append(goal_ids)
    elif goal_id is not None:
        raw_ids.append(goal_id)

    cleaned: list[str] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_ids:
        normalized = str(item or "").strip()
        if not normalized:
            skipped.append({"goal_id": "", "reason": "empty"})
            continue
        if normalized in seen:
            skipped.append({"goal_id": normalized, "reason": "duplicate"})
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    return cleaned, skipped


async def _resolve_many_goals(goal_ids: list[str], resolution: str) -> tuple[list[dict], list[str]]:
    from llm.prompt import goals as _goals

    resolved_goals: list[dict] = []
    not_found: list[str] = []
    for current_goal_id in goal_ids:
        resolved = await _goals.resolve_goal(current_goal_id, resolution=resolution)
        if resolved is None:
            not_found.append(current_goal_id)
            continue
        resolved_goals.append(resolved)
    return resolved_goals, not_found


def get_declaration() -> dict:
    from llm.prompt import goals as _goals

    ids = [goal["goal_id"] for goal in _goals.get_all()]
    return {
        "name": "resolve_goal",
        "description": "结束一个或多个活跃目标，并说明它们是完成、放弃、重复、被替代还是误建。这很重要，不要让已完成或无效的目标堆叠。",
        "parameters": {
            "type": "object",
            "properties": {
                "goal_ids": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "enum": ids,
                    },
                    "description": "要结束的目标 ID 列表。支持一次结束多个目标。",
                },
                "resolution": {
                    "type": "string",
                    "enum": list(VALID_RESOLUTIONS),
                    "description": "该目标结束的方式。",
                },
                "motivation": {
                    "type": "string"
                },
            },
            "required": ["goal_ids", "resolution", "motivation"],
        },
    }


def execute(
    goal_ids: list[str] | str | None = None,
    resolution: str = "",
    motivation: str = "",
    goal_id: str | None = None,
    **kwargs,
) -> dict:
    import app_state
    from llm.prompt import goals as _goals

    loop: asyncio.AbstractEventLoop | None = app_state.main_loop
    if loop is None or not loop.is_running():
        return {"error": "主事件循环不可用，无法结束目标"}

    normalized_goal_ids, skipped = _normalize_goal_ids(goal_ids, goal_id=goal_id)
    if not normalized_goal_ids:
        return {
            "ok": False,
            "message": "没有可结束的目标，goal_ids 不能为空",
            "skipped": skipped,
            "total": len(_goals.get_all()),
        }

    coro = _resolve_many_goals(normalized_goal_ids, resolution=resolution)
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        resolved_goals, not_found = future.result(timeout=10)
    except Exception as e:
        return {"error": f"结束目标失败: {e}"}

    resolved_payload = [
        {
            "goal_id": resolved["goal_id"],
            "title": resolved["title"],
            "status": resolved["status"],
            "resolution": resolved["resolution"],
        }
        for resolved in resolved_goals
    ]

    if resolved_payload:
        message = f"已结束 {len(resolved_payload)} 个目标，resolution={resolution}"
        if not_found:
            message += f"；未找到 {len(not_found)} 个"
        if skipped:
            message += f"；跳过 {len(skipped)} 个重复或空值"

        result = {
            "ok": True,
            "message": message,
            "resolved": resolved_payload,
            "not_found": not_found,
            "skipped": skipped,
            "total": len(_goals.get_all()),
        }
        if len(resolved_payload) == 1:
            result["goal"] = resolved_payload[0]
        return result

    return {
        "ok": False,
        "message": "未找到任何目标，可能已被结束、删除，或输入重复/为空",
        "resolved": [],
        "not_found": not_found,
        "skipped": skipped,
        "total": len(_goals.get_all()),
    }