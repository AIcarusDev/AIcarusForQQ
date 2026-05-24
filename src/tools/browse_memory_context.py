"""browse_memory_context.py — 查看某条记忆事件的原始对话上下文

给定一个记忆事件 ID，在 chat_messages 中查找该事件发生时的前后原始消息，
让模型能够看到记忆形成时的完整对话片段，验证或加深对该记忆的理解。
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "all"
ALWAYS_AVAILABLE: bool = False

DECLARATION: dict = {
    "name": "browse_memory_context",
    "description": (
        "查看某条记忆事件形成时的原始对话上下文。"
        "当你对某条记忆的准确性存疑、需要了解记忆背景、或想回顾当时的完整讨论时使用。"
        "输入 <recent_events> 中某条 <event> 的 id 属性值，"
        "返回该事件发生前后的原始聊天记录片段。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "integer",
                "description": "要查询的记忆事件 id（来自 <event id=...>）。",
            },
            "before": {
                "type": "integer",
                "description": "向前查看的消息条数，默认 5，最大 15。",
            },
            "after": {
                "type": "integer",
                "description": "向后查看的消息条数，默认 3，最大 10。",
            },
            "motivation": {
                "type": "string",
                "description": "为什么要查看这条记忆的原始上下文。",
            },
        },
        "required": ["event_id", "motivation"],
    },
}


def execute(
    event_id: int,
    before: int = 5,
    after: int = 3,
    motivation: str = "",
    **kwargs: Any,
) -> dict:
    from database import DB_PATH

    event_id = int(event_id)
    before = max(1, min(int(before), 15))
    after = max(0, min(int(after), 10))

    logger.info(
        "[tools] browse_memory_context: event_id=%d before=%d after=%d motivation=%r",
        event_id, before, after, motivation,
    )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        # 1. 查找事件元数据
        ev = conn.execute(
            "SELECT event_id, event_type, summary, conv_type, conv_id, conv_name, occurred_at "
            "FROM MemoryEvents WHERE event_id=? AND is_deleted=0",
            (event_id,),
        ).fetchone()

        if ev is None:
            return {"error": f"未找到 event_id={event_id} 的记忆事件（或已被删除）"}

        conv_type = ev["conv_type"] or ""
        conv_id = ev["conv_id"] or ""
        conv_name = ev["conv_name"] or ""
        occurred_at_ms = ev["occurred_at"] or 0
        event_summary = ev["summary"] or ""
        event_type = ev["event_type"] or ""

        if not conv_type or not conv_id:
            return {
                "error": "该记忆事件没有关联的会话信息，无法查询原始上下文",
                "event": {"id": event_id, "summary": event_summary},
            }

        session_key = f"{conv_type}_{conv_id}"

        # 2. occurred_at 之前的 before 条（取最近的，反转后得正序）
        before_rows = conn.execute(
            "SELECT id, role, sender_name, sender_id, timestamp, content "
            "FROM chat_messages "
            "WHERE session_key=? AND role<>'note' AND created_at <= ? "
            "ORDER BY created_at DESC, id DESC LIMIT ?",
            (session_key, occurred_at_ms, before),
        ).fetchall()
        before_rows = list(reversed(before_rows))

        # 3. occurred_at 之后的 after 条
        after_rows = conn.execute(
            "SELECT id, role, sender_name, sender_id, timestamp, content "
            "FROM chat_messages "
            "WHERE session_key=? AND role<>'note' AND created_at > ? "
            "ORDER BY created_at ASC, id ASC LIMIT ?",
            (session_key, occurred_at_ms, after),
        ).fetchall()

    if not before_rows and not after_rows:
        return {
            "error": "未找到该事件关联会话的聊天记录（可能已超出保留窗口）",
            "event": {
                "id": event_id,
                "type": event_type,
                "summary": event_summary,
                "scene": conv_name or session_key,
            },
        }

    def _fmt_row(r: sqlite3.Row) -> dict:
        role = r["role"] or ""
        sender_name = (r["sender_name"] or "").strip()
        sender_id = (r["sender_id"] or "").strip()
        if role == "assistant":
            label = "我"
        elif sender_id:
            label = f"{sender_name}#{sender_id}" if sender_name else sender_id
        else:
            label = sender_name or role
        return {
            "speaker": label,
            "time": r["timestamp"] or "",
            "content": r["content"] or "",
        }

    messages = [_fmt_row(r) for r in list(before_rows) + list(after_rows)]

    occurred_str = (
        datetime.fromtimestamp(occurred_at_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if occurred_at_ms
        else ""
    )

    return {
        "event": {
            "id": event_id,
            "type": event_type,
            "summary": event_summary,
            "occurred_at": occurred_str,
            "scene": conv_name or session_key,
        },
        "messages": messages,
        "note": (
            f"以上是记忆「{event_summary[:40]}」形成时"
            f"（{occurred_str}）在「{conv_name or session_key}」中的原始对话片段，"
            f"共 {len(messages)} 条。"
        ),
    }
