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

logger = logging.getLogger("AICQ.llm.history")


def _row_to_entry(row: sqlite3.Row) -> dict:
    """sqlite Row → 与 session.context_messages 一致的 entry dict。"""
    entry: dict = {
        "role": row["role"],
        "message_id": row["message_id"],
        "sender_id": row["sender_id"],
        "sender_name": row["sender_name"],
        "sender_role": row["sender_role"],
        "timestamp": row["timestamp"],
        "content": row["content"],
        "content_type": row["content_type"],
        "content_segments": json.loads(row["content_segments"] or "[]"),
    }
    images_raw = json.loads(row["images"] or "[]")
    if images_raw:
        entry["images"] = images_raw
    return entry


def _session_key(session) -> str:
    return f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""


def _oldest_context_db_id(session, conn: sqlite3.Connection, session_key: str) -> int | None:
    """定位 session.context_messages 中最早一条消息在 DB 中的自增 id。"""
    msg_ids = [
        m.get("message_id", "")
        for m in session.context_messages
        if m.get("message_id", "").strip()
    ]
    if not msg_ids:
        return None
    placeholders = ",".join("?" * len(msg_ids))
    row = conn.execute(
        f"SELECT MIN(id) AS min_id FROM chat_messages "
        f"WHERE session_key=? AND message_id IN ({placeholders})",
        [session_key] + msg_ids,
    ).fetchone()
    if row and row["min_id"] is not None:
        return int(row["min_id"])
    return None


def _latest_db_id(conn: sqlite3.Connection, session_key: str) -> int | None:
    """返回当前会话在 DB 中最新一条消息的自增 id。"""
    row = conn.execute(
        "SELECT MAX(id) AS max_id FROM chat_messages WHERE session_key=?",
        (session_key,),
    ).fetchone()
    if row and row["max_id"] is not None:
        return int(row["max_id"])
    return None


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
            rows = conn.execute(
                """SELECT role, message_id, sender_id, sender_name, sender_role,
                          timestamp, content, content_type, content_segments, images
                   FROM chat_messages
                   WHERE session_key=? AND id >= ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (session_key, top_db_id, page_size),
            ).fetchall()
    except Exception:
        logger.exception("[history_window] 查询失败 session=%s top=%d", session_key, top_db_id)
        return []

    return [_row_to_entry(r) for r in rows]


def count_unread_below(session, page_window: list[dict]) -> int:
    """计算窗口最下方一条消息之后，DB 中还有多少条更新的消息。"""
    if not page_window:
        return 0
    last_id_str = page_window[-1].get("message_id", "")
    if not last_id_str:
        return 0

    from database import DB_PATH

    session_key = _session_key(session)
    if not session_key:
        return 0

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id FROM chat_messages WHERE session_key=? AND message_id=? LIMIT 1",
                (session_key, last_id_str),
            ).fetchone()
            if not row:
                return 0
            last_db_id = int(row["id"])
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages WHERE session_key=? AND id > ?",
                (session_key, last_db_id),
            ).fetchone()
            return int(cnt["c"]) if cnt else 0
    except Exception:
        logger.exception("[history_window] 未读计数失败 session=%s", session_key)
        return 0


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

            # 取 id < current_top 的最新 page_size 条，再反转得到时间正序
            rows = conn.execute(
                """SELECT id FROM chat_messages
                   WHERE session_key=? AND id < ?
                   ORDER BY id DESC
                   LIMIT ?""",
                (session_key, current_top, page_size),
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

            # 新窗口的 top = 当前窗口 top 之后的第 page_size+1 条（即向下推一页）
            rows = conn.execute(
                """SELECT id FROM chat_messages
                   WHERE session_key=? AND id > ?
                   ORDER BY id ASC
                   LIMIT ?""",
                (session_key, current_top, page_size),
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

            # 探测：下一页起点之后是否还有更新的消息
            tail = conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages WHERE session_key=? AND id > ?",
                (session_key, new_top),
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
