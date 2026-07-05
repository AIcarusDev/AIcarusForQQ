
"""QQ unread session preview XML rendering.

仅负责生成 <unread_info> 块，不承担主模型 user prompt 的总装。
"""

import html
import re
from datetime import datetime, timezone
from typing import Any

from llm.prompt.xml_builder import (
    _format_relative_time,
    _hydrate_dynamic_group_display_names,
)


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SESSION_PREVIEW_LIMIT = 5


def _media_label(seg: dict) -> str:
    seg_type = str(seg.get("type", "") or "")
    if seg_type == "image":
        return "[图片]"
    if seg_type == "sticker":
        return "[动画表情]"
    if seg_type == "file":
        filename = str(seg.get("filename", "") or "").strip()
        return f"[文件:{filename}]" if filename else "[文件]"
    if seg_type == "voice":
        return "[语音]"
    if seg_type == "video":
        return "[视频]"
    if seg_type == "forward":
        return "[合并转发]"
    if seg_type == "card":
        label = str(seg.get("label", "") or "").strip()
        return f"[{label}]" if label else "[卡片消息]"
    label = str(seg.get("label", "") or seg_type or "消息").strip()
    return f"[{label}]"


def _render_preview_segments(segments: list[dict]) -> str:
    """Render content segments as plain preview text, never XML or media sentinels."""
    parts: list[str] = []
    for seg in segments:
        seg_type = str(seg.get("type", "") or "")
        if seg_type == "text":
            parts.append(str(seg.get("text", "") or ""))
        elif seg_type == "mention":
            display = str(seg.get("display", "") or "").strip()
            uid = str(seg.get("uid", "") or "").strip()
            parts.append(display or (f"@{uid}" if uid else "@"))
        elif seg_type == "emoji":
            name = str(seg.get("name", "") or "").strip()
            eid = str(seg.get("id", "") or "").strip()
            if name:
                parts.append(f"[{name}]")
            elif eid:
                parts.append(f"[表情:{eid}]")
            else:
                parts.append("[表情]")
        else:
            parts.append(_media_label(seg))
    return "".join(parts)


def _sanitize_preview_text(text: str, max_len: int) -> str:
    text = _CONTROL_CHAR_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return html.escape(text, quote=False)


def _render_preview_text(msg: dict, max_len: int = 30) -> str:
    """从消息条目中提取用于预览的纯文本（截断），供 build_unread_info_xml 使用。"""
    segments = msg.get("content_segments")
    if segments:
        text = _render_preview_segments(segments)
    else:
        content_type = str(msg.get("content_type", "text") or "text")
        if content_type == "image":
            text = "[图片]"
        elif content_type == "sticker":
            text = "[动画表情]"
        elif content_type == "file":
            text = "[文件]"
        elif content_type == "voice":
            text = "[语音]"
        elif content_type == "video":
            text = "[视频]"
        elif content_type == "recall":
            text = "[撤回了一条消息]"
        else:
            text = str(msg.get("content", "") or "")
    return _sanitize_preview_text(text, max_len)


def _attrs(**values: str) -> str:
    return " ".join(
        f'{key}="{html.escape(str(value), quote=True)}"'
        for key, value in values.items()
        if value
    )


def _session_identity(session: Any) -> tuple[str, str]:
    conv_type = str(getattr(session, "conv_type", "") or "")
    conv_id = str(getattr(session, "conv_id", "") or "")
    if conv_type not in {"group", "private", "temp"} or not conv_id:
        return "", ""
    return conv_type, conv_id


def _unread_session_identities(sessions_dict: dict, current_key: str) -> set[tuple[str, str]]:
    unread: set[tuple[str, str]] = set()
    for key, s in sessions_dict.items():
        if key == current_key:
            continue
        if getattr(s, "unread_count", 0) <= 0:
            continue
        identity = _session_identity(s)
        if identity != ("", ""):
            unread.add(identity)
    return unread


def _latest_activity_timestamp(session: Any) -> str:
    for msg in reversed(getattr(session, "context_messages", []) or []):
        if msg.get("role") == "note":
            continue
        timestamp = str(msg.get("timestamp", "") or "").strip()
        if timestamp:
            return timestamp
    return ""


def _timestamp_sort_key(timestamp: str) -> float:
    try:
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError):
        return 0.0


def build_recent_active_sessions_xml(
    sessions_dict: dict,
    current_key: str,
    *,
    limit: int = _SESSION_PREVIEW_LIMIT,
) -> str:
    """Render recently visible QQ sessions that do not currently have unread messages."""
    unread_identities = _unread_session_identities(sessions_dict, current_key)
    seen: set[tuple[str, str]] = set()
    rows: list[tuple[float, int, str]] = []

    for index, (key, s) in enumerate(sessions_dict.items()):
        if key == current_key:
            continue
        identity = _session_identity(s)
        if identity == ("", "") or identity in seen or identity in unread_identities:
            continue
        if getattr(s, "unread_count", 0) > 0:
            continue
        timestamp = _latest_activity_timestamp(s)
        if not timestamp:
            continue

        conv_type, conv_id = identity
        rel_time = _format_relative_time(timestamp)
        if conv_type == "group":
            session_attrs = _attrs(
                type="group",
                id=conv_id,
                name=str(getattr(s, "conv_name", "") or ""),
                last_active=rel_time,
            )
        elif conv_type == "private":
            session_attrs = _attrs(
                type="private",
                id=conv_id,
                nickname=str(getattr(s, "conv_name", "") or ""),
                last_active=rel_time,
            )
        else:
            session_attrs = _attrs(
                type="temp",
                id=conv_id,
                user_id=conv_id,
                nickname=str(getattr(s, "conv_name", "") or ""),
                source_group_id=str(getattr(s, "temp_source_group_id", "") or ""),
                source_group_name=str(getattr(s, "temp_source_group_name", "") or ""),
                last_active=rel_time,
            )
        if not session_attrs:
            continue
        seen.add(identity)
        rows.append((_timestamp_sort_key(timestamp), index, f"  <session {session_attrs}/>"))

    if not rows:
        return ""

    rows.sort(key=lambda item: (-item[0], item[1]))
    selected = [line for _sort_key, _index, line in rows[: max(0, int(limit))]]
    if not selected:
        return ""
    return "<recent_active_sessions>\n" + "\n".join(selected) + "\n</recent_active_sessions>"


def build_unread_info_xml(
    sessions_dict: dict,
    current_key: str,
    *,
    limit: int = _SESSION_PREVIEW_LIMIT,
) -> str:
    """生成 <unread_info> 块，按最近未读时间列出除当前会话外最多 limit 个未读会话预览。

    sessions_dict  — 全局 sessions 字典（key → ConversationSession）
    current_key    — 当前 bot 正在处理的会话 key（格式 "type_id"），排除在外

    无未读时返回空字符串。
    """
    rows: list[tuple[float, int, list[str]]] = []
    for index, (key, s) in enumerate(sessions_dict.items()):
        if key == current_key:
            continue
        if s.unread_count <= 0:
            continue

        # 取最后一条真实用户消息（跳过 bot 和 note）作为 preview
        last_msg = None
        for m in reversed(s.context_messages):
            if m.get("role") not in ("bot", "note"):
                last_msg = m
                break
        if last_msg is None:
            continue
        if s.conv_type == "group":
            last_msg = _hydrate_dynamic_group_display_names([last_msg], s._get_conv_meta())[0]

        unread_display = "99+" if s.unread_count > 99 else str(s.unread_count)
        timestamp = str(last_msg.get("timestamp", "") or "").strip()
        rel_time = _format_relative_time(timestamp)
        preview_text = _render_preview_text(last_msg)
        lines: list[str] = []

        if s.conv_type == "group":
            session_attrs = _attrs(
                type="group",
                id=str(s.conv_id),
                name=str(s.conv_name or ""),
                unread=unread_display,
            )
            preview_attrs = _attrs(timestamp=rel_time, sender=str(last_msg.get("sender_name", "") or ""))
            lines.append(f"  <session {session_attrs}>")
            lines.append(f"    <preview {preview_attrs}>{preview_text}</preview>")
            lines.append("  </session>")
            rows.append((_timestamp_sort_key(timestamp), index, lines))
        elif s.conv_type == "private":
            session_attrs = _attrs(
                type="private",
                id=str(s.conv_id),
                nickname=str(s.conv_name or ""),
                unread=unread_display,
            )
            preview_attrs = _attrs(timestamp=rel_time)
            lines.append(f"  <session {session_attrs}>")
            lines.append(f"    <preview {preview_attrs}>{preview_text}</preview>")
            lines.append("  </session>")
            rows.append((_timestamp_sort_key(timestamp), index, lines))
        elif s.conv_type == "temp":
            session_attrs = _attrs(
                type="temp",
                id=str(s.conv_id),
                user_id=str(s.conv_id),
                nickname=str(s.conv_name or ""),
                unread=unread_display,
                source_group_id=str(getattr(s, "temp_source_group_id", "") or ""),
                source_group_name=str(getattr(s, "temp_source_group_name", "") or ""),
            )
            preview_attrs = _attrs(timestamp=rel_time)
            lines.append(f"  <session {session_attrs}>")
            lines.append(f"    <preview {preview_attrs}>{preview_text}</preview>")
            lines.append("  </session>")
            rows.append((_timestamp_sort_key(timestamp), index, lines))

    if not rows:
        return ""
    rows.sort(key=lambda item: (-item[0], item[1]))
    selected_rows = rows[: max(0, int(limit))]
    if not selected_rows:
        return ""
    selected_lines = [line for _sort_key, _index, row_lines in selected_rows for line in row_lines]
    return "<unread_info>\n" + "\n".join(selected_lines) + "\n</unread_info>"
