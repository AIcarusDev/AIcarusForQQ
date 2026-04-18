"""search_history.py — 搜索当前会话的历史聊天记录

按关键词（多词 AND）在数据库中搜索历史消息，返回命中消息及其前后上下文。
需要运行时上下文：session（获取当前会话 session_key）。
"""

import logging
import sqlite3
from typing import Any, Callable

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "all"

DECLARATION: dict = {
    "name": "search_history",
    "description": (
        "搜索当前会话（群聊或私聊）的历史聊天记录。"
        "按关键词匹配历史消息，并返回每条命中消息前后的上下文。"
        "适合用于：查找某话题是什么时候聊过的、追溯某次讨论的细节、了解某人说过什么。"
        "仅搜索文字内容，不检索图片。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "搜索关键词列表，多个关键词同时满足才算命中（AND 逻辑）。"
                    "例如 [\"天气\", \"明天\"] 会匹配同时含有「天气」和「明天」的消息。"
                ),
                "minItems": 1,
            },
            "sender_id": {
                "type": "string",
                "description": "（可选）只搜索指定 QQ 号发送的消息。留空则搜索所有人。",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回几条命中结果，默认 3，最大 5。",
            },
            "context_window": {
                "type": "integer",
                "description": "每条命中结果前后各保留几条上下文消息，默认 7，最大 10。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["keywords", "motivation"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session: Any) -> Callable:
    def execute(
        keywords: list,
        sender_id: str = "",
        limit: int = 3,
        context_window: int = 7,
        motivation: str = "",
        **kwargs,
    ) -> dict:
        from database import DB_PATH

        # 参数校验
        kws = [str(k).strip() for k in (keywords if isinstance(keywords, list) else [keywords]) if str(k).strip()]
        if not kws:
            return {"error": "关键词不能为空"}
        limit = max(1, min(int(limit), 5))
        context_window = max(0, min(int(context_window), 10))

        # 构造当前会话的 session_key
        session_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
        if not session_key:
            return {"error": "当前会话信息不可用"}

        logger.info(
            "[tools] search_history: session_key=%s keywords=%r limit=%d context_window=%d",
            session_key, kws, limit, context_window,
        )

        try:
            # 同步 sqlite3 读取（WAL 模式支持并发读，不阻塞 aiosqlite 的写入）
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row

                # 构造多关键词 AND 查询
                like_clauses = " AND ".join(["content LIKE ?"] * len(kws))
                like_params = [f"%{k}%" for k in kws]

                if sender_id:
                    where = f"session_key=? AND ({like_clauses}) AND sender_id=?"
                    params: list = [session_key] + like_params + [str(sender_id)]
                else:
                    where = f"session_key=? AND ({like_clauses})"
                    params = [session_key] + like_params

                hits_sql = (
                    f"SELECT id, role, sender_name, sender_id, timestamp, content "
                    f"FROM chat_messages WHERE {where} ORDER BY id DESC LIMIT ?"
                )
                hits_rows = conn.execute(hits_sql, params + [limit]).fetchall()

                if not hits_rows:
                    return {
                        "total_hits": 0,
                        "message": "未找到匹配的历史消息。",
                        "results": [],
                    }

                results = []
                for hit_row in reversed(hits_rows):  # 还原时间正序
                    hit_id = hit_row["id"]

                    # 取命中消息之前 context_window 条
                    before_rows = conn.execute(
                        "SELECT id, role, sender_name, sender_id, timestamp, content "
                        "FROM chat_messages "
                        "WHERE session_key=? AND id < ? "
                        "ORDER BY id DESC LIMIT ?",
                        (session_key, hit_id, context_window),
                    ).fetchall()

                    # 取命中消息 + 之后 context_window 条
                    after_rows = conn.execute(
                        "SELECT id, role, sender_name, sender_id, timestamp, content "
                        "FROM chat_messages "
                        "WHERE session_key=? AND id >= ? "
                        "ORDER BY id ASC LIMIT ?",
                        (session_key, hit_id, context_window + 1),
                    ).fetchall()

                    # 合并：前面部分逆序 + 后面部分
                    context_rows = list(reversed(before_rows)) + list(after_rows)

                    context_msgs = []
                    for row in context_rows:
                        msg = {
                            "sender": row["sender_name"] or row["sender_id"] or row["role"],
                            "time": row["timestamp"],
                            "content": row["content"],
                        }
                        if row["id"] == hit_id:
                            msg["is_hit"] = True
                        context_msgs.append(msg)

                    results.append({
                        "hit": {
                            "sender": hit_row["sender_name"] or hit_row["sender_id"] or hit_row["role"],
                            "time": hit_row["timestamp"],
                            "content": hit_row["content"],
                        },
                        "context": context_msgs,
                    })

                logger.info(
                    "[tools] search_history: 找到 %d 条命中 session_key=%s",
                    len(results), session_key,
                )
                return {
                    "total_hits": len(results),
                    "results": results,
                }

        except Exception as e:
            logger.warning("[tools] search_history: 查询异常 — %s", e)
            return {"error": f"搜索失败: {e}"}

    return execute
