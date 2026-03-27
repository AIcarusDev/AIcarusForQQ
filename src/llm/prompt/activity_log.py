"""activity_log.py — 意识流活动日志

全局记录 bot 的意识活动轨迹（chat / watcher 切换历史），
用于向 LLM prompt 注入 <activity_log> XML 块。

内存中维护有序列表，同时异步持久化到 DB。
"""

import time
import uuid
from dataclasses import dataclass
from typing import Optional


# ── 数据结构 ────────────────────────────────────────────

@dataclass
class ActivityEntry:
    entry_id: str
    entry_type: str          # 'chat' | 'watcher' | 'hibernate'
    created_at: float        # time.time()

    # chat 专属
    conv_type: str = ""      # 'group' | 'private'
    conv_id: str = ""
    conv_name: str = ""

    # enter
    enter_attitude: str = ""     # 'active' | 'passive'
    enter_motivation: str = ""   # active 进入时的动机
    enter_remark: str = ""       # passive 进入时的描述（如 "收到@，被动激活"）
    enter_from: str = ""         # shift 来源，格式 "type:id:name"（可选）

    # hibernate 专属
    hibernate_minutes: int = 0   # 计划休眠时长（分钟）

    # end（ended_at 为 None 则为 current）
    ended_at: Optional[float] = None
    end_attitude: str = ""
    end_action: str = ""     # 'idle' | 'shift' | 'engage' | 'interrupted' | 'woke_up'
    end_motivation: str = ""
    end_remark: str = ""     # 被动中断时的描述


# ── 全局状态 ────────────────────────────────────────────

_log: list[ActivityEntry] = []
_max_entries: int = 10


def configure(max_entries: int) -> None:
    global _max_entries
    _max_entries = max_entries


def get_current() -> ActivityEntry | None:
    """返回尚未关闭的当前条目（如果存在）。"""
    if _log and _log[-1].ended_at is None:
        return _log[-1]
    return None


# ── 内存辅助 ────────────────────────────────────────────

def _append(entry: ActivityEntry) -> None:
    _log.append(entry)
    _trim()


def _trim() -> None:
    """超出限制时从头部丢弃最旧的已关闭记录，保留 current。"""
    while len(_log) > _max_entries:
        for i, e in enumerate(_log):
            if e.ended_at is not None:
                _log.pop(i)
                break
        else:
            break  # 全是未关闭条目，不丢


# ── 公共异步 API ─────────────────────────────────────────

async def open_entry(
    entry_type: str,
    enter_attitude: str = "",
    enter_motivation: str = "",
    enter_remark: str = "",
    enter_from: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    hibernate_minutes: int = 0,
) -> ActivityEntry:
    """开始一条新记录（enter 端），更新内存并持久化到 DB。"""
    from database import save_activity_entry
    entry = ActivityEntry(
        entry_id=uuid.uuid4().hex,
        entry_type=entry_type,
        created_at=time.time(),
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
        enter_attitude=enter_attitude,
        enter_motivation=enter_motivation,
        enter_remark=enter_remark,
        enter_from=enter_from,
        hibernate_minutes=hibernate_minutes,
    )
    _append(entry)
    await save_activity_entry(entry)
    return entry


async def close_current(
    end_attitude: str,
    end_action: str,
    end_motivation: str = "",
    end_remark: str = "",
) -> ActivityEntry | None:
    """关闭当前进行中的记录（end 端），更新内存并持久化到 DB。"""
    from database import update_activity_entry
    current = get_current()
    if current is None:
        return None
    current.ended_at = time.time()
    current.end_attitude = end_attitude
    current.end_action = end_action
    current.end_motivation = end_motivation
    current.end_remark = end_remark
    await update_activity_entry(current)
    return current


def close_current_sync(
    end_attitude: str,
    end_action: str,
    end_motivation: str = "",
    end_remark: str = "",
) -> ActivityEntry | None:
    """仅更新内存（不写 DB），用于 shutdown 等需要同步的场合。"""
    current = get_current()
    if current is None:
        return None
    current.ended_at = time.time()
    current.end_attitude = end_attitude
    current.end_action = end_action
    current.end_motivation = end_motivation
    current.end_remark = end_remark
    return current


def restore_from_db(rows: list[dict]) -> None:
    """startup 时从 DB 恢复内存状态。"""
    _log.clear()
    for row in rows:
        entry = ActivityEntry(
            entry_id=row["entry_id"],
            entry_type=row["entry_type"],
            created_at=row["created_at"],
            conv_type=row.get("conv_type", ""),
            conv_id=row.get("conv_id", ""),
            conv_name=row.get("conv_name", ""),
            enter_attitude=row.get("enter_attitude", ""),
            enter_motivation=row.get("enter_motivation", ""),
            enter_remark=row.get("enter_remark", ""),
            enter_from=row.get("enter_from", ""),
            hibernate_minutes=row.get("hibernate_minutes", 0),
            ended_at=row.get("ended_at"),
            end_attitude=row.get("end_attitude", ""),
            end_action=row.get("end_action", ""),
            end_motivation=row.get("end_motivation", ""),
            end_remark=row.get("end_remark", ""),
        )
        _log.append(entry)


# ── XML 构建 ─────────────────────────────────────────────

def _escape_attr(value: str) -> str:
    """转义 XML 属性值中的特殊字符（用于拼接到 \"...\" 引号内）。"""
    return (
        value
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _format_timeago(ts: float) -> str:
    diff = max(0.0, time.time() - ts)
    minutes = int(diff / 60)
    if minutes < 1:
        return "刚刚"
    if minutes < 60:
        return f"约 {minutes} 分钟前"
    hours = minutes // 60
    if hours < 24:
        return f"约 {hours} 小时前"
    return f"约 {hours // 24} 天前"


def _format_duration(start: float, end: float | None) -> str:
    if end is None:
        return "ForNow"
    minutes = max(0, int((end - start) / 60))
    if minutes < 1:
        return "不到 1 分钟"
    if minutes < 60:
        return f"约 {minutes} 分钟"
    hours = minutes // 60
    rem = minutes % 60
    if rem == 0:
        return f"约 {hours} 小时"
    return f"约 {hours} 小时 {rem} 分钟"


def _build_entry_xml(entry: ActivityEntry, is_current: bool) -> str:
    lines = []
    timeago = _format_timeago(entry.created_at)
    tag = "current" if is_current else "past"
    lines.append(f'<{tag} timeago="{timeago}" type="{entry.entry_type}">')

    if entry.entry_type == "chat":
        # 会话信息
        conv_attrs = f'type="{entry.conv_type}" id="{entry.conv_id}"'
        if entry.conv_name:
            conv_attrs += f' name="{_escape_attr(entry.conv_name)}"'
        lines.append(f'<conversation {conv_attrs}/>')

        # enter 节点
        enter_parts = [f'attitude="{entry.enter_attitude}"']
        if entry.enter_attitude == "active":
            if entry.enter_motivation:
                enter_parts.append(f'motivation="{_escape_attr(entry.enter_motivation)}"')
            if entry.enter_from:
                enter_parts.append(f'from="{_escape_attr(entry.enter_from)}"')
        else:
            enter_parts.append('motivation="null"')
            if entry.enter_remark:
                enter_parts.append(f'remark="{_escape_attr(entry.enter_remark)}"')
        lines.append(f'<enter {" ".join(enter_parts)}/>')

    elif entry.entry_type == "hibernate":
        lines.append(f'<hibernate planned_minutes="{entry.hibernate_minutes}"/>')

    # duration
    duration = _format_duration(entry.created_at, entry.ended_at)
    lines.append(f'<duration time="{duration}"/>')

    # end（仅 past）
    if not is_current and entry.ended_at is not None:
        end_parts = [f'attitude="{entry.end_attitude}"', f'action="{entry.end_action}"']
        if entry.end_attitude == "active" and entry.end_motivation:
            end_parts.append(f'motivation="{_escape_attr(entry.end_motivation)}"')
        if entry.end_remark:
            end_parts.append(f'remark="{_escape_attr(entry.end_remark)}"')
        lines.append(f'<end {" ".join(end_parts)}/>')

    lines.append(f'</{tag}>')
    return "\n".join(lines)


def build_activity_log_xml() -> str:
    """构建 <activity_log> XML 字符串，供 prompt 使用。"""
    if not _log:
        return "<activity_log>\n<!-- 暂无活动记录 -->\n</activity_log>"
    parts = ["<activity_log>"]
    for i, entry in enumerate(_log):
        is_current = (i == len(_log) - 1) and (entry.ended_at is None)
        parts.append(_build_entry_xml(entry, is_current))
    parts.append("</activity_log>")
    return "\n".join(parts)
