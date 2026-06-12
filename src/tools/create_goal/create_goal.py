"""create_goal.py — 创建活跃目标工具"""

import asyncio
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

from .prompt import DESCRIPTION

DECLARATION: dict = {
  "name": "create_goal",
  "description": DESCRIPTION,
  "parameters": {
    "type": "object",
    "properties": {
      "goals": {
        "type": "array",
        "minItems": 1,
        "description": "要创建的目标列表",
        "items": {
          "type": "object",
          "properties": {
            "title": {
              "type": "string",
              "description": "目标的标题，简洁明确。"
            },
            "content": {
              "type": "string",
              "description": "目标的具体描述。"
            },
            "reason": {
              "type": "string",
              "description": "创建这个目标的原因，会随目标显示在 `<goals>` 中。让自己知道缘由。"
            },
          },
          "required": ["title", "content", "reason"],
        },
      },
    },
    "required": ["goals"],
  },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def _normalize_text(value: Any) -> str:
  return str(value or "").strip()


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
  goals = args.get("goals")
  if not isinstance(goals, list):
    return args, []

  fallback_reason: Any | None = None
  fallback_source = ""
  if "reason" in args:
    fallback_reason = args.get("reason")
    fallback_source = "root reason"
  elif "motivation" in args:
    fallback_reason = args.get("motivation")
    fallback_source = "legacy root motivation"

  repaired_goals = goals
  changes: list[str] = []
  for index, item in enumerate(goals):
    if not isinstance(item, dict):
      continue

    updated_item = item
    if "reason" not in updated_item and "motivation" in updated_item:
      updated_item = dict(updated_item)
      updated_item["reason"] = updated_item.get("motivation")
      changes.append(f"mapped legacy goals[{index}].motivation to reason")

    if "reason" not in updated_item and fallback_source:
      if updated_item is item:
        updated_item = dict(updated_item)
      updated_item["reason"] = fallback_reason
      changes.append(f"mapped {fallback_source} to goals[{index}].reason")

    if updated_item is not item:
      if repaired_goals is goals:
        repaired_goals = list(goals)
      repaired_goals[index] = updated_item

  if repaired_goals is goals and "reason" not in args:
    return args, changes

  repaired = dict(args)
  if repaired_goals is not goals:
    repaired["goals"] = repaired_goals
  if "reason" in repaired:
    repaired.pop("reason")
    changes.append("removed legacy root reason")
  return repaired, changes


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
      skipped.append({
        "title": title,
        "content": content,
        "goal_reason": reason,
        "reason": "empty",
      })
      continue

    key = (title, content)
    if key in seen_keys:
      skipped.append({
        "title": title,
        "content": content,
        "reason": "duplicate",
      })
      continue

    seen_keys.add(key)
    cleaned.append({"title": title, "content": content, "reason": reason})

  return cleaned, skipped


def make_handler(session: Any) -> Callable:
  def execute(goals: list[dict], reason: str = "", **kwargs) -> dict:
    import app_state
    from llm.prompt import goals as _goals

    loop: asyncio.AbstractEventLoop | None = app_state.main_loop
    if loop is None or not loop.is_running():
      return {"error": "主事件循环不可用，无法创建目标"}

    fallback_reason = _normalize_text(reason or kwargs.get("motivation"))
    if fallback_reason:
      goals = [
        {**goal, "reason": goal.get("reason") or fallback_reason}
        if isinstance(goal, dict) else goal
        for goal in goals
      ]

    normalized_goals, skipped = _normalize_goal_items(goals, _goals.get_all())
    if not normalized_goals:
      return {
        "ok": False,
        "message": "没有可创建的新目标，可能都为空或与现有目标重复",
        "skipped": skipped,
        "total": len(_goals.get_all()),
      }

    coro = _goals.add_goals(
      goal_items=normalized_goals,
      conv_type=session.conv_type,
      conv_id=session.conv_id,
      conv_name=session.conv_name,
    )
    try:
      created_rows = run_coroutine_sync(coro, loop, timeout=10)
    except Exception as e:
      return {"error": f"创建目标失败: {e}"}

    return {
      "ok": True,
      "created": [
        {"goal_id": row["goal_id"], "title": row["title"]}
        for row in created_rows
      ],
      "skipped": skipped,
      "total": len(_goals.get_all()),
    }

  return execute