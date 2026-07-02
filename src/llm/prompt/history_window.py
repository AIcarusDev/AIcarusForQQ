"""history_window.py — 浏览态聊天历史窗口加载

scroll_chat_log 工具配套：当 session.chat_window_view 处于 history 模式时，
按视口锚点 top_db_id 从数据库取一段历史消息，转换成与
session.context_messages 完全相同的 dict 结构，供现有 build_chat_log_xml /
build_multimodal_content 路径直接复用。
"""

from __future__ import annotations

import json
import logging
import sqlite3

from database import (
    CHAT_MESSAGE_ORDER_ASC_SQL,
    CHAT_MESSAGE_ORDER_DESC_SQL,
    CHAT_MESSAGE_SORT_KEY_SQL,
)

logger = logging.getLogger("AICQ.llm.history")


def _row_to_entry(row: sqlite3.Row) -> dict:
    """sqlite Row → 与 session.context_messages 一致的 entry dict。"""
    entry: dict = {
        "role": row["role"],
        "message_id": row["message_id"],
        "sender_id": row["sender_id"],
        "sender_name": row["sender_name"],
        "sender_card": row["sender_card"],
        "sender_nickname": row["sender_nickname"],
        "sender_role": row["sender_role"],
        "sender_title": row["sender_title"],
        "sender_level": row["sender_level"],
        "timestamp": row["timestamp"],
        "content": row["content"],
        "content_type": row["content_type"],
        "content_segments": json.loads(row["content_segments"] or "[]"),
    }
    if delivery_state := str(row["delivery_state"] or ""):
        entry["delivery_state"] = delivery_state
    if delivery_error := str(row["delivery_error"] or ""):
        entry["delivery_error"] = delivery_error
    if reply_to := str(row["reply_to"] or ""):
        entry["reply_to"] = reply_to
    images_raw = json.loads(row["images"] or "[]")
    if images_raw:
        entry["images"] = images_raw
    return entry


def _session_key(session) -> str:
    return f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""


def _hydrate_history_quote_extras(
    session,
    conn: sqlite3.Connection,
    session_key: str,
    visible_entries: list[dict],
) -> None:
    """恢复历史窗口页外引用的预览数据。"""
    visible_ids = {str(entry.get("message_id", "")) for entry in visible_entries}
    needed = sorted({
        str(entry.get("reply_to", ""))
        for entry in visible_entries
        if entry.get("reply_to")
        and str(entry.get("reply_to", "")) not in visible_ids
        and str(entry.get("reply_to", "")) not in session.quoted_extra
    })
    if not needed:
        return

    placeholders = ",".join("?" * len(needed))
    rows = conn.execute(
        f"""SELECT role, message_id, sender_id, sender_name,
                   sender_card, sender_nickname, sender_role,
                   sender_title, sender_level, timestamp, reply_to,
                   content, content_type, content_segments, images,
                   delivery_state, delivery_error
            FROM chat_messages
            WHERE session_key=? AND message_id IN ({placeholders})""",
        [session_key, *needed],
    ).fetchall()
    for row in rows:
        entry = _row_to_entry(row)
        session.quoted_extra[str(entry.get("message_id", ""))] = entry


def _oldest_context_db_id(session, conn: sqlite3.Connection, session_key: str) -> int | None:
    """定位 session.context_messages 中按消息时间排序最早一条消息的 DB id。"""
    msg_ids = [
        m.get("message_id", "")
        for m in session.context_messages
        if m.get("message_id", "").strip()
    ]
    if not msg_ids:
        return None
    placeholders = ",".join("?" * len(msg_ids))
    row = conn.execute(
        f"SELECT id FROM chat_messages "
        f"WHERE session_key=? AND message_id IN ({placeholders}) "
        f"ORDER BY {CHAT_MESSAGE_ORDER_ASC_SQL} LIMIT 1",
        [session_key] + msg_ids,
    ).fetchone()
    if row and row["id"] is not None:
        return int(row["id"])
    return None


def _anchor_sort_key(conn: sqlite3.Connection, session_key: str, anchor_db_id: int) -> float | None:
    """返回锚点消息的时间排序键。"""
    if anchor_db_id <= 0:
        return None
    row = conn.execute(
        f"SELECT {CHAT_MESSAGE_SORT_KEY_SQL} AS sort_key "
        "FROM chat_messages WHERE session_key=? AND id=?",
        (session_key, anchor_db_id),
    ).fetchone()
    if not row or row["sort_key"] is None:
        return None
    return float(row["sort_key"])


def _has_rows_before(conn: sqlite3.Connection, session_key: str, anchor_db_id: int) -> bool:
    """判断锚点之前是否还有更早记录。"""
    anchor_sort = _anchor_sort_key(conn, session_key, anchor_db_id)
    if anchor_sort is None:
        return False
    row = conn.execute(
        f"""SELECT 1 FROM chat_messages
            WHERE session_key=?
              AND (
                  {CHAT_MESSAGE_SORT_KEY_SQL} < ?
                  OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id < ?)
              )
            LIMIT 1""",
        (session_key, anchor_sort, anchor_sort, anchor_db_id),
    ).fetchone()
    return row is not None


def load_history_window(session, top_db_id: int, page_size: int) -> list[dict]:
    """以 top_db_id 为窗口最上方一条消息的锚点，向后取 page_size 条历史消息。

    返回时间正序的 entry 列表（与 session.context_messages 同结构）。
    """
    from database import DB_PATH

    session_key = _session_key(session)
    if not session_key:
        return []

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            anchor_sort = _anchor_sort_key(conn, session_key, top_db_id)
            if anchor_sort is None:
                return []
            rows = conn.execute(
                f"""SELECT role, message_id, sender_id, sender_name,
                          sender_card, sender_nickname, sender_role,
                          sender_title, sender_level, timestamp, reply_to,
                          content, content_type, content_segments, images,
                          delivery_state, delivery_error
                   FROM chat_messages
                   WHERE session_key=?
                     AND (
                         {CHAT_MESSAGE_SORT_KEY_SQL} > ?
                         OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id >= ?)
                     )
                   ORDER BY {CHAT_MESSAGE_ORDER_ASC_SQL}
                   LIMIT ?""",
                (session_key, anchor_sort, anchor_sort, top_db_id, page_size),
            ).fetchall()
            entries = [_row_to_entry(r) for r in rows]
            _hydrate_history_quote_extras(session, conn, session_key, entries)
    except Exception:
        logger.exception("[history_window] 查询失败 session=%s top=%d", session_key, top_db_id)
        return []

    return entries


def has_previous_messages(
    session,
    *,
    browsing: bool | None = None,
    top_db_id: int | None = None,
) -> bool:
    """判断当前聊天窗口上方是否还有更早记录。"""
    from database import DB_PATH

    session_key = _session_key(session)
    if not session_key:
        return False

    history_mode = session.is_browsing_history() if browsing is None else browsing

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            if history_mode:
                anchor_db_id = top_db_id if top_db_id is not None else int(session.chat_window_view.get("top_db_id") or 0)
            else:
                anchor_db_id = _oldest_context_db_id(session, conn, session_key) or 0
            return _has_rows_before(conn, session_key, int(anchor_db_id or 0))
    except Exception:
        logger.exception("[history_window] has_previous_messages 失败 session=%s", session_key)
        return False


def scroll_up(session) -> dict:
    """向更早翻一页。返回 dict 含 ok / moved / message。"""
    page_size = int(session.chat_window_view.get("page_size", 10))
    from database import DB_PATH

    session_key = _session_key(session)
    if not session_key:
        return {"ok": False, "moved": False, "message": "当前会话信息不可用。"}

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row

            # 当前窗口最上方一条消息的 db_id
            if session.is_browsing_history():
                current_top = int(session.chat_window_view.get("top_db_id") or 0)
            else:
                current_top = _oldest_context_db_id(session, conn, session_key) or 0

            if current_top <= 0:
                return {
                    "ok": True,
                    "moved": False,
                    "message": "无法定位当前聊天窗口边界，未发生滚动。",
                }
            current_sort = _anchor_sort_key(conn, session_key, current_top)
            if current_sort is None:
                return {
                    "ok": True,
                    "moved": False,
                    "message": "无法定位当前聊天窗口边界，未发生滚动。",
                }

            # 取当前窗口上方更早的最新 page_size 条，再反转得到时间正序。
            rows = conn.execute(
                f"""SELECT id FROM chat_messages
                   WHERE session_key=?
                     AND (
                         {CHAT_MESSAGE_SORT_KEY_SQL} < ?
                         OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id < ?)
                     )
                   ORDER BY {CHAT_MESSAGE_ORDER_DESC_SQL}
                   LIMIT ?""",
                (session_key, current_sort, current_sort, current_top, page_size),
            ).fetchall()

            if not rows:
                # 已无更早消息：保持当前模式，不切到 history
                return {
                    "ok": True,
                    "moved": False,
                    "message": "已经到达最早的聊天记录，无法继续向上。",
                }

            new_top = int(rows[-1]["id"])  # 反转后最早一条的 id
    except Exception:
        logger.exception("[history_window] scroll_up 失败 session=%s", session_key)
        return {"ok": False, "moved": False, "message": "滚动聊天窗口时发生内部错误。"}

    session.chat_window_view = {
        "mode": "history",
        "top_db_id": new_top,
        "page_size": page_size,
    }
    return {"ok": True, "moved": True, "message": "聊天窗口已向上滚动。"}


def scroll_down(session) -> dict:
    """向更新方向翻一页。若已贴到最新则自动回到 live。"""
    if not session.is_browsing_history():
        return {
            "ok": True,
            "moved": False,
            "message": "当前已经在最新聊天窗口，无需向下滚动。",
        }

    page_size = int(session.chat_window_view.get("page_size", 10))
    current_top = int(session.chat_window_view.get("top_db_id") or 0)

    from database import DB_PATH

    session_key = _session_key(session)
    if not session_key:
        return {"ok": False, "moved": False, "message": "当前会话信息不可用。"}

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            current_sort = _anchor_sort_key(conn, session_key, current_top)
            if current_sort is None:
                session.reset_chat_window_view()
                return {
                    "ok": True,
                    "moved": True,
                    "snapped_to_latest": True,
                    "message": "聊天窗口已向下滚动并回到最新。",
                }

            # 新窗口的 top = 当前窗口 top 之后的第 page_size+1 条（即向下推一页）
            rows = conn.execute(
                f"""SELECT id FROM chat_messages
                   WHERE session_key=?
                     AND (
                         {CHAT_MESSAGE_SORT_KEY_SQL} > ?
                         OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id > ?)
                     )
                   ORDER BY {CHAT_MESSAGE_ORDER_ASC_SQL}
                   LIMIT ?""",
                (session_key, current_sort, current_sort, current_top, page_size),
            ).fetchall()

            if not rows:
                # 已经在最末位置，直接回 live
                session.reset_chat_window_view()
                return {
                    "ok": True,
                    "moved": True,
                    "snapped_to_latest": True,
                    "message": "聊天窗口已向下滚动并回到最新。",
                }

            new_top = int(rows[-1]["id"])
            new_sort = _anchor_sort_key(conn, session_key, new_top)
            if new_sort is None:
                session.reset_chat_window_view()
                return {
                    "ok": True,
                    "moved": True,
                    "snapped_to_latest": True,
                    "message": "聊天窗口已向下滚动并回到最新。",
                }

            # 探测：下一页起点之后是否还有更新的消息
            tail = conn.execute(
                f"""SELECT COUNT(*) AS c FROM chat_messages
                    WHERE session_key=?
                      AND (
                          {CHAT_MESSAGE_SORT_KEY_SQL} > ?
                          OR ({CHAT_MESSAGE_SORT_KEY_SQL} = ? AND id > ?)
                      )""",
                (session_key, new_sort, new_sort, new_top),
            ).fetchone()
            remaining = int(tail["c"]) if tail else 0
    except Exception:
        logger.exception("[history_window] scroll_down 失败 session=%s", session_key)
        return {"ok": False, "moved": False, "message": "滚动聊天窗口时发生内部错误。"}

    if remaining <= 0:
        session.reset_chat_window_view()
        return {
            "ok": True,
            "moved": True,
            "snapped_to_latest": True,
            "message": "聊天窗口已向下滚动并回到最新。",
        }

    session.chat_window_view = {
        "mode": "history",
        "top_db_id": new_top,
        "page_size": page_size,
    }
    return {"ok": True, "moved": True, "message": "聊天窗口已向下滚动。"}


def scroll_to_latest(session) -> dict:
    """直接跳到最新聊天窗口。"""
    if not session.is_browsing_history():
        return {
            "ok": True,
            "moved": False,
            "message": "当前已经在最新聊天窗口。",
        }
    session.reset_chat_window_view()
    return {
        "ok": True,
        "moved": True,
        "message": "聊天窗口已跳回最新。",
    }
