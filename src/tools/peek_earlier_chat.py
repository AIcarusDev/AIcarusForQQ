"""peek_earlier_chat.py — 获取当前上下文窗口之前的历史消息

按数据库 id 顺序，查找早于当前上下文窗口的 N 条历史消息，
以与主 prompt 完全相同的 XML 格式返回，供模型了解更早的对话背景。

上下文边界锚定方式：
  在数据库中找到 context_messages 里最老那条消息的自增 id，
  然后取 id < 该值的记录，避免与当前上下文重叠。

"""

import json
import logging
import sqlite3
from typing import Any, Callable

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "all"

DECLARATION: dict = {
    "name": "peek_earlier_chat",
    "description": (
        "获取当前聊天上下文窗口之前的历史消息，以 XML 格式返回。"
        "适合用于：当前窗口上下文不足、需要追溯当前讨论的历史背景。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "要获取的历史消息条数，范围 5~15，过大的值会被截断到配置上限。",
            },
            "motivation": {
                "type": "string",
                "description": "说明为什么需要查阅更早的历史背景。",
            },
        },
        "required": ["motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session", "config"]


def make_handler(session: Any, config: dict) -> Callable:
    def execute(count: int = 10, motivation: str = "", **kwargs) -> dict:
        from database import DB_PATH
        from llm.prompt.xml_builder import build_chat_log_xml

        # 从配置读取条数上限，强制 5~15
        cfg_max = config.get("tools", {}).get("earlier_context_count", 10)
        try:
            cfg_max = int(cfg_max)
        except (ValueError, TypeError):
            cfg_max = 10
        cfg_max = max(5, min(cfg_max, 15))

        # 实际请求条数，不超过配置上限
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = cfg_max
        count = max(5, min(count, cfg_max))

        # 会话 key
        session_key = (
            f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
        )
        if not session_key:
            return {"error": "当前会话信息不可用"}

        context_messages = session.context_messages
        if not context_messages:
            return {"xml": "", "message": "当前上下文为空，无法确定窗口边界。"}

        # 收集当前上下文中所有非空 message_id，用来在 DB 里定位边界
        context_msg_ids = [
            m.get("message_id", "")
            for m in context_messages
            if m.get("message_id", "").strip()
        ]

        logger.info(
            "[tools] peek_earlier_chat: session=%s count=%d ctx_ids=%d",
            session_key,
            count,
            len(context_msg_ids),
        )

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                # 找出当前上下文窗口在 DB 中最小的自增 id（即最早那条）
                oldest_db_id: int | None = None
                if context_msg_ids:
                    placeholders = ",".join("?" * len(context_msg_ids))
                    row = conn.execute(
                        f"SELECT MIN(id) AS min_id FROM chat_messages "
                        f"WHERE session_key=? AND message_id IN ({placeholders})",
                        [session_key] + context_msg_ids,
                    ).fetchone()
                    if row and row["min_id"] is not None:
                        oldest_db_id = int(row["min_id"])

                if oldest_db_id is None:
                    return {
                        "xml": "",
                        "message": (
                            "无法在数据库中定位当前上下文边界，"
                            "可能当前上下文消息尚未持久化或 message_id 缺失。"
                        ),
                    }

                # 取上下文窗口之前的 count 条（DESC 取最近的，结果再反转为正序）
                rows = conn.execute(
                    """SELECT role, message_id, sender_id, sender_name, sender_role,
                              timestamp, content, content_type, content_segments, images
                       FROM chat_messages
                       WHERE session_key=? AND id < ?
                       ORDER BY id DESC
                       LIMIT ?""",
                    (session_key, oldest_db_id, count),
                ).fetchall()

                if not rows:
                    return {
                        "xml": "",
                        "message": "当前上下文窗口之前没有更早的历史消息。",
                    }

                # 反转为时间正序
                msgs: list[dict] = []
                for r in reversed(rows):
                    entry: dict = {
                        "role": r["role"],
                        "message_id": r["message_id"],
                        "sender_id": r["sender_id"],
                        "sender_name": r["sender_name"],
                        "sender_role": r["sender_role"],
                        "timestamp": r["timestamp"],
                        "content": r["content"],
                        "content_type": r["content_type"],
                        "content_segments": json.loads(r["content_segments"] or "[]"),
                    }
                    images_raw = json.loads(r["images"] or "[]")
                    if images_raw:
                        entry["images"] = images_raw
                    msgs.append(entry)

                conv_meta: dict = {
                    "type": session.conv_type,
                    "id": session.conv_id,
                    "name": session.conv_name,
                    "member_count": getattr(session, "conv_member_count", 0),
                    "bot_id": getattr(session, "_qq_id", ""),
                    "bot_name": getattr(session, "_qq_name", ""),
                    "bot_card": getattr(session, "_qq_card", ""),
                }

                xml_str = build_chat_log_xml(msgs, conv_meta)
                return {"xml": xml_str, "count": len(msgs)}

        except Exception:
            logger.exception("[tools] peek_earlier_chat 查询失败")
            return {"error": "查询历史消息时发生内部错误，请稍后再试。"}

    return execute
