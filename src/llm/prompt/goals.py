"""goals.py — 模型活跃目标管理

全局维护一个内存中的目标列表，支持写入、结束、渲染为 XML。
启动时从数据库恢复，运行时通过 create_goal / resolve_goal 工具调用更新。
"""

import html
import secrets
import time
from datetime import datetime, timezone


_goals: list[dict] = []
_max_entries: int = 10
VALID_RESOLUTIONS: tuple[str, ...] = (
    "completed",
    "abandoned",
    "duplicate",
    "superseded",
    "mistaken",
)


def restore(rows: list[dict]) -> None:
    global _goals
    _goals = list(rows)


def get_all() -> list[dict]:
    return list(_goals)


def get_max_entries() -> int:
    return _max_entries


def _detach_goal(goal_id: str) -> dict | None:
    global _goals

    for index, goal in enumerate(_goals):
        if goal["goal_id"] == goal_id:
            found = goal
            _goals = _goals[:index] + _goals[index + 1 :]
            return found
    return None


def _age_text(created_at_ms: int, now: datetime) -> str:
    delta_sec = int(now.timestamp() - created_at_ms / 1000)
    if delta_sec < 60:
        return "刚刚"
    if delta_sec < 3600:
        return f"{delta_sec // 60}分钟前"
    if delta_sec < 86400:
        return f"{delta_sec // 3600}小时前"
    if delta_sec < 86400 * 7:
        return f"{delta_sec // 86400}天前"
    if delta_sec < 86400 * 30:
        return f"{delta_sec // (86400 * 7)}周前"
    return f"{delta_sec // (86400 * 30)}个月前"


def _origin_display(conv_type: str, conv_name: str, conv_id: str) -> str:
    if conv_type == "group":
        source = "来自群聊"
    elif conv_type == "private":
        source = "来自私聊"
    else:
        source = "来自当前上下文"

    if conv_id:
        suffix = f"{conv_name}({conv_id})" if conv_name else conv_id
        return f"{source} · {suffix}"
    return source


def build_active_goals_xml(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)

    total = len(_goals)
    cap = _max_entries
    if not _goals:
        return "\n".join(
            [
                f'<active items="0/{cap}">',
                "  你现在漫无目的，如果需要的话，使用 `create_goal` 创建目标。",
                "</active>",
            ]
        )

    lines = [f'<active items="{total}/{cap}">']
    for goal in _goals:
        goal_id = goal["goal_id"]
        age = _age_text(goal["created_at"], now)
        origin = _origin_display(
            goal.get("conv_type", ""),
            goal.get("conv_name", ""),
            goal.get("conv_id", ""),
        )
        lines.append(f'  <item id="{goal_id}">')
        lines.append(f'    <title>{html.escape(goal["title"])}</title>')
        lines.append(f'    <content>{html.escape(goal["content"])}</content>')
        lines.append(f'    <age>{age}</age>')
        lines.append(f'    <reason>{html.escape(goal["reason"])}</reason>')
        lines.append(f'    <origin>{html.escape(origin)}</origin>')
        lines.append("  </item>")
    lines.append("</active>")
    return "\n".join(lines)


def _next_id() -> str:
    used = {goal["goal_id"] for goal in _goals}
    while True:
        candidate = f"goal_{secrets.token_hex(4)}"
        if candidate not in used:
            return candidate


async def add_goals(
    goal_items: list[dict[str, str]],
    reason: str,
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
) -> list[dict]:
    from database import soft_delete_goal as _db_delete, write_goal as _db_write

    created_rows: list[dict] = []
    for goal_item in goal_items:
        goal_id = _next_id()
        created_at = int(time.time() * 1000)
        entry = {
            "goal_id": goal_id,
            "created_at": created_at,
            "updated_at": created_at,
            "title": goal_item["title"],
            "content": goal_item["content"],
            "reason": reason,
            "conv_type": conv_type,
            "conv_id": conv_id,
            "conv_name": conv_name,
            "status": "active",
            "resolution": "",
        }

        while len(_goals) >= _max_entries:
            oldest = _goals.pop(0)
            await _db_delete(oldest["goal_id"])

        _goals.append(entry)
        await _db_write(
            goal_id=goal_id,
            title=goal_item["title"],
            content=goal_item["content"],
            reason=reason,
            conv_type=conv_type,
            conv_id=conv_id,
            conv_name=conv_name,
            status="active",
            resolution="",
        )
        created_rows.append(entry)

    return created_rows


async def resolve_goal(goal_id: str, resolution: str) -> dict | None:
    from database import resolve_goal as _db_resolve_goal

    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(f"无效的 goal resolution: {resolution}")

    found_goal = _detach_goal(goal_id)
    if found_goal is None:
        return None

    found_goal["status"] = "resolved"
    found_goal["resolution"] = resolution
    found_goal["updated_at"] = int(time.time() * 1000)
    await _db_resolve_goal(goal_id, resolution=resolution)
    return found_goal