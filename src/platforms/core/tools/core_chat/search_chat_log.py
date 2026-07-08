"""Search the fixed Core one-to-one chat history."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Callable, Literal

from pydantic import Field

from database import (
    CHAT_MESSAGE_ORDER_ASC_SQL,
    CHAT_MESSAGE_ORDER_DESC_SQL,
    CHAT_MESSAGE_SORT_KEY_SQL,
)
from platforms.core.session_context import CORE_MAIN_FOCUS, NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools.core_chat")


class SearchChatLogArgs(ToolArgsModel):
    query: str = Field(
        min_length=1,
        max_length=120,
        description="要搜索的文字。多个关键词用空格分隔。若为多个关键词，所有关键词都匹配才算命中。",
    )
    sender: Literal["any", "guardian", "self"] = Field(
        default="any",
        description="限制发送方：any 双方，guardian 只搜监护人消息，self 搜自己消息。",
    )
    limit: int = Field(default=5, ge=1, le=10, description="最多返回几条命中结果，默认 5，最大 10。")
    context_window: int = Field(
        default=2,
        ge=0,
        le=5,
        description="每条命中前后各返回几条上下文，默认 2，最大 5。",
    )


TOOL_CONTRACT = ToolContract(
    name="search_chat_log",
    description=(
        "搜索当前页面的历史聊天记录。"
        "返回命中消息的 message_id、发送方、时间、文本和少量上下文；"
    ),
    args_model=SearchChatLogArgs,
)

REQUIRES_CONTEXT: list[str] = ["core_session_provider"]
PARALLEL_SAFE = True
PARALLEL_KEY = "core_session_read"


def _terms(query: str) -> list[str]:
    return [part.strip() for part in str(query or "").split() if part.strip()]


def _sender_role(sender: str) -> str:
    if sender == "guardian":
        return "user"
    if sender == "self":
        return "bot"
    return ""


def _display_role(role: str) -> str:
    if role == "bot":
        return "self"
    if role == "user":
        return "guardian"
    return role or "unknown"


def _row_message(row: sqlite3.Row, *, hit_id: int | None = None) -> dict[str, Any]:
    message = {
        "message_id": str(row["message_id"] or ""),
        "sender": _display_role(str(row["role"] or "")),
        "time": str(row["timestamp"] or ""),
        "content": str(row["content"] or ""),
    }
    if hit_id is not None and int(row["id"]) == hit_id:
        message["is_hit"] = True
    return message


def make_handler(core_session_provider: Callable[[], Any | None]) -> Callable:
    core_session_provider = ensure_session_provider(core_session_provider)

    def execute(
        query: str = "",
        sender: str = "any",
        limit: int = 5,
        context_window: int = 2,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        from database import DB_PATH

        session = core_session_provider()
        if session is None:
            return {"ok": False, "error": NO_CURRENT_SESSION_ERROR, "results": []}

        terms = _terms(query)
        if not terms:
            query_text = str(query or "").strip()
            if query_text:
                terms = [query_text]
        if not terms:
            return {"ok": False, "error": "搜索关键词不能为空。", "results": []}

        try:
            bounded_limit = max(1, min(int(limit), 10))
        except (TypeError, ValueError):
            bounded_limit = 5
        try:
            bounded_context = max(0, min(int(context_window), 5))
        except (TypeError, ValueError):
            bounded_context = 2

        role_filter = _sender_role(str(sender or "any").strip())
        session_key = CORE_MAIN_FOCUS.key()

        logger.info(
            "[core_chat] search_chat_log: session_key=%s terms=%r sender=%s limit=%d context=%d",
            session_key,
            terms,
            sender,
            bounded_limit,
            bounded_context,
        )

        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                like_clauses = " AND ".join(["content LIKE ?"] * len(terms))
                params: list[Any] = [session_key, *[f"%{term}%" for term in terms]]
                where = [
                    "session_key=?",
                    "role IN ('user', 'bot')",
                    "content<>''",
                    f"({like_clauses})",
                ]
                if role_filter:
                    where.append("role=?")
                    params.append(role_filter)

                hits = conn.execute(
                    f"""SELECT id, role, message_id, timestamp, content,
                               {CHAT_MESSAGE_SORT_KEY_SQL} AS sort_key
                        FROM chat_messages
                        WHERE {' AND '.join(where)}
                        ORDER BY {CHAT_MESSAGE_ORDER_DESC_SQL}
                        LIMIT ?""",
                    [*params, bounded_limit],
                ).fetchall()

                if not hits:
                    return {
                        "ok": True,
                        "total_hits": 0,
                        "message": "未找到匹配的历史消息。",
                        "results": [],
                    }

                results: list[dict[str, Any]] = []
                for hit in reversed(hits):
                    hit_id = int(hit["id"])
                    hit_sort = float(hit["sort_key"])

                    before = conn.execute(
                        f"""SELECT id, role, message_id, timestamp, content
                            FROM chat_messages
                            WHERE session_key=?
                              AND role IN ('user', 'bot')
                              AND (
                                  {CHAT_MESSAGE_SORT_KEY_SQL} < ?
                                  OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id < ?)
                              )
                            ORDER BY {CHAT_MESSAGE_ORDER_DESC_SQL}
                            LIMIT ?""",
                        (session_key, hit_sort, hit_sort, hit_id, bounded_context),
                    ).fetchall()

                    after = conn.execute(
                        f"""SELECT id, role, message_id, timestamp, content
                            FROM chat_messages
                            WHERE session_key=?
                              AND role IN ('user', 'bot')
                              AND (
                                  {CHAT_MESSAGE_SORT_KEY_SQL} > ?
                                  OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id >= ?)
                              )
                            ORDER BY {CHAT_MESSAGE_ORDER_ASC_SQL}
                            LIMIT ?""",
                        (session_key, hit_sort, hit_sort, hit_id, bounded_context + 1),
                    ).fetchall()

                    context = [
                        _row_message(row, hit_id=hit_id)
                        for row in [*reversed(before), *after]
                    ]
                    results.append(
                        {
                            "hit": _row_message(hit),
                            "context": context,
                        }
                    )

                return {
                    "ok": True,
                    "total_hits": len(results),
                    "results": results,
                }
        except Exception as exc:
            logger.warning("[core_chat] search_chat_log failed: %s", exc)
            return {"ok": False, "error": f"搜索失败: {exc}", "results": []}

    return execute
