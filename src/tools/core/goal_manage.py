"""goal_manage.py - create or resolve active goals."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Literal

from llm.prompt.goals import VALID_RESOLUTIONS
from tools._async_bridge import run_coroutine_sync
from pydantic import Field, RootModel

from tools.contract import ToolArgsModel, ToolContract

class GoalItem(ToolArgsModel):
    title: str = Field(min_length=1, description="目标标题，简洁明确。")
    content: str = Field(min_length=1, description="目标的具体描述。")
    reason: str = Field(min_length=1, description="创建这个目标的原因，会随目标显示在 `<goals>` 中。")


class GoalCreateArgs(ToolArgsModel):
    action: Literal["create"] = Field(description="create=创建目标。")
    goals: list[GoalItem] = Field(min_length=1, description="action=create 时要创建的目标列表。")


class GoalResolveArgs(ToolArgsModel):
    action: Literal["resolve"] = Field(description="resolve=结束目标。")
    goal_ids: list[str] = Field(
        min_length=1,
        json_schema_extra={"uniqueItems": True},
        description="action=resolve 时要结束的目标 ID 列表，来自 active goals。",
    )
    resolution: Literal["completed", "abandoned", "duplicate", "superseded", "mistaken"] = Field(
        description="action=resolve 时该目标结束的方式。",
    )


class GoalManageArgs(RootModel[GoalCreateArgs | GoalResolveArgs]):
    pass


TOOL_CONTRACT = ToolContract(
    name="goal_manage",
    description=(
        "创建或结束一个或多个活跃目标。"
        "action=create 时创建目标；action=resolve 时结束目标并说明完成、放弃、重复、被替代或误建。"
    ),
    args_model=GoalManageArgs,
)

REQUIRES_CONTEXT: list[str] = ["session"]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_goal_items(
    goal_items: list[dict[str, Any]],
    existing_goals: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    existing_keys = {
        (_normalize_text(goal.get("title")), _normalize_text(goal.get("content")))
        for goal in existing_goals
    }
    seen_keys = set(existing_keys)
    cleaned: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for item in goal_items:
        title = _normalize_text(item.get("title"))
        content = _normalize_text(item.get("content"))
        reason = _normalize_text(item.get("reason"))
        if not title or not content or not reason:
            skipped.append({"title": title, "content": content, "goal_reason": reason, "reason": "empty"})
            continue
        key = (title, content)
        if key in seen_keys:
            skipped.append({"title": title, "content": content, "reason": "duplicate"})
            continue
        seen_keys.add(key)
        cleaned.append({"title": title, "content": content, "reason": reason})
    return cleaned, skipped


def _normalize_goal_ids(goal_ids: list[Any] | str | None) -> tuple[list[str], list[dict[str, str]]]:
    raw_ids = goal_ids if isinstance(goal_ids, list) else [goal_ids]
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
    for goal_id in goal_ids:
        resolved = await _goals.resolve_goal(goal_id, resolution=resolution)
        if resolved is None:
            not_found.append(goal_id)
        else:
            resolved_goals.append(resolved)
    return resolved_goals, not_found


def make_handler(session: Any) -> Callable:
    def execute(
        action: str,
        goals: list[dict] | None = None,
        goal_ids: list[str] | str | None = None,
        resolution: str = "",
        **_: Any,
    ) -> dict:
        import app_state
        from llm.prompt import goals as _goals

        loop: asyncio.AbstractEventLoop | None = app_state.main_loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用，无法管理目标"}

        if action == "create":
            normalized_goals, skipped = _normalize_goal_items(goals or [], _goals.get_all())
            if not normalized_goals:
                return {
                    "ok": False,
                    "message": "没有可创建的新目标，可能都为空或与现有目标重复",
                    "skipped": skipped,
                    "total": len(_goals.get_all()),
                }
            try:
                created_rows = run_coroutine_sync(
                    _goals.add_goals(
                        goal_items=normalized_goals,
                        conv_type=session.conv_type,
                        conv_id=session.conv_id,
                        conv_name=session.conv_name,
                    ),
                    loop,
                    timeout=10,
                )
            except Exception as exc:
                return {"error": f"创建目标失败: {exc}"}
            return {
                "ok": True,
                "action": "create",
                "created": [{"goal_id": row["goal_id"], "title": row["title"]} for row in created_rows],
                "skipped": skipped,
                "total": len(_goals.get_all()),
            }

        if action != "resolve":
            return {"ok": False, "error": f"未知 goal action: {action!r}"}

        normalized_goal_ids, skipped = _normalize_goal_ids(goal_ids)
        if not normalized_goal_ids:
            return {
                "ok": False,
                "message": "没有可结束的目标，goal_ids 不能为空",
                "skipped": skipped,
                "total": len(_goals.get_all()),
            }
        try:
            resolved_goals, not_found = run_coroutine_sync(
                _resolve_many_goals(normalized_goal_ids, resolution=resolution),
                loop,
                timeout=10,
            )
        except Exception as exc:
            return {"error": f"结束目标失败: {exc}"}
        resolved_payload = [
            {
                "goal_id": goal["goal_id"],
                "title": goal["title"],
                "status": goal["status"],
                "resolution": goal["resolution"],
            }
            for goal in resolved_goals
        ]
        return {
            "ok": bool(resolved_payload),
            "action": "resolve",
            "resolved": resolved_payload,
            "not_found": not_found,
            "skipped": skipped,
            "total": len(_goals.get_all()),
        }

    return execute
