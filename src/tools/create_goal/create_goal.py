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
        "description": "要创建的目标列表，每项包含 title 和 content。",
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
          },
          "required": ["title", "content"],
        },
      },
      "motivation": {
        "type": "string"
      },
    },
    "required": ["goals", "motivation"],
  },
}

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
    if not title or not content:
      skipped.append({
        "title": title,
        "content": content,
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
    cleaned.append({"title": title, "content": content})

  return cleaned, skipped


def make_handler(session: Any) -> Callable:
  def execute(goals: list[dict], motivation: str = "", **kwargs) -> dict:
    import app_state
    from llm.prompt import goals as _goals

    loop: asyncio.AbstractEventLoop | None = app_state.main_loop
    if loop is None or not loop.is_running():
      return {"error": "主事件循环不可用，无法创建目标"}

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
      reason=motivation,
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