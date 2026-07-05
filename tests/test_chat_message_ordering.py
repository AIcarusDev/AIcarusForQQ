from __future__ import annotations

import asyncio
import sqlite3

import database
from platforms.chat.history_window import (
    has_previous_messages,
    load_history_window,
    scroll_down,
    scroll_to_message,
    scroll_up,
)
from platforms.qq.tools.qq.qq_chat_view.scroll_chat_log import make_handler as make_scroll_chat_log_handler
from platforms.qq.tools.qq.qq_chat_view.search_history import make_handler as make_search_history_handler


SESSION_KEY = "qq:group:42"


class FakeSession:
    conv_type = "group"
    conv_id = "42"
    conv_name = "Chronology Test"
    quoted_extra: dict = {}

    def __init__(self, context_messages: list[dict] | None = None):
        self.context_messages = context_messages or []
        self.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 2}

    def is_browsing_history(self) -> bool:
        return self.chat_window_view.get("mode") == "history"

    def reset_chat_window_view(self) -> None:
        self.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 2}


def _setup_db(monkeypatch, tmp_path):
    db_path = tmp_path / "AICQ.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    asyncio.run(database.init_db())
    return db_path


async def _save(message_id: str, timestamp: str, content: str) -> None:
    await database.save_chat_message(
        SESSION_KEY,
        {
            "role": "user",
            "message_id": message_id,
            "sender_id": "10001",
            "sender_name": "Alice",
            "timestamp": timestamp,
            "content": content,
            "content_type": "text",
            "content_segments": [{"type": "text", "text": content}],
        },
    )


async def _seed_out_of_order_messages() -> None:
    # Live rows arrive first and therefore get low DB ids.
    await _save("recent-1", "2026-06-25T08:00:00+08:00", "recent one")
    await _save("recent-2", "2026-06-25T08:01:00+08:00", "recent two")

    # Recovery backfills older rows later, so they get higher DB ids.
    await _save("old-1", "2026-06-22T00:00:00+08:00", "old one")
    await _save("old-2", "2026-06-22T00:01:00+08:00", "old two")

    await _save("latest", "2026-06-26T09:48:15+08:00", "latest")


async def _seed_linear_messages(count: int) -> None:
    for i in range(1, count + 1):
        await _save(
            f"msg-{i:02d}",
            f"2026-06-25T08:{i:02d}:00+08:00",
            f"message {i:02d}",
        )


def _id_for(db_path, message_id: str) -> int:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM chat_messages WHERE session_key=? AND message_id=?",
            (SESSION_KEY, message_id),
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_chat_message_latest_window_and_edges_use_message_time(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)

    async def scenario():
        await _seed_out_of_order_messages()
        all_messages = await database.load_chat_messages(SESSION_KEY, limit=10)
        latest_three = await database.load_chat_messages(SESSION_KEY, limit=3)
        earliest = await database.get_chat_message_edge(SESSION_KEY, newest=False)
        newest = await database.get_chat_message_edge(SESSION_KEY, newest=True)
        return all_messages, latest_three, earliest, newest

    all_messages, latest_three, earliest, newest = asyncio.run(scenario())

    assert [m["message_id"] for m in all_messages] == [
        "old-1",
        "old-2",
        "recent-1",
        "recent-2",
        "latest",
    ]
    assert [m["message_id"] for m in latest_three] == [
        "recent-1",
        "recent-2",
        "latest",
    ]
    assert earliest["message_id"] == "old-1"
    assert newest["message_id"] == "latest"


def test_chat_message_delivery_state_round_trips(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)

    async def scenario():
        await database.save_chat_message(
            SESSION_KEY,
            {
                "role": "bot",
                "message_id": "pending_abc",
                "sender_id": "bot",
                "sender_name": "Bot",
                "timestamp": "2026-06-26T09:48:15+08:00",
                "content": "pending hello",
                "content_type": "text",
                "content_segments": [{"type": "text", "text": "pending hello"}],
                "delivery_state": "pending",
                "delivery_error": "adapter timeout",
            },
        )
        pending = await database.load_chat_messages(SESSION_KEY, limit=10)
        await database.update_chat_message_id(SESSION_KEY, "pending_abc", "-100")
        confirmed = await database.load_chat_messages(SESSION_KEY, limit=10)
        return pending, confirmed

    pending, confirmed = asyncio.run(scenario())

    assert pending[0]["message_id"] == "pending_abc"
    assert pending[0]["delivery_state"] == "pending"
    assert pending[0]["delivery_error"] == "adapter timeout"
    assert confirmed[0]["message_id"] == "-100"
    assert "delivery_state" not in confirmed[0]
    assert "delivery_error" not in confirmed[0]


def test_history_window_scrolls_by_message_time_not_insert_id(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)

    async def scenario():
        await _seed_out_of_order_messages()
        return await database.load_chat_messages(SESSION_KEY, limit=3)

    live_context = asyncio.run(scenario())
    session = FakeSession(live_context)

    assert has_previous_messages(session, browsing=False) is True

    up_result = scroll_up(session)
    assert up_result["moved"] is True
    assert session.chat_window_view["top_db_id"] == _id_for(db_path, "old-1")

    older_window = load_history_window(session, session.chat_window_view["top_db_id"], 2)
    assert [m["message_id"] for m in older_window] == ["old-1", "old-2"]
    assert has_previous_messages(session, browsing=True, top_db_id=session.chat_window_view["top_db_id"]) is False

    down_result = scroll_down(session)
    assert down_result["moved"] is True
    assert session.chat_window_view["mode"] == "history"
    assert session.chat_window_view["top_db_id"] == _id_for(db_path, "recent-1")

    newer_window = load_history_window(session, session.chat_window_view["top_db_id"], 2)
    assert [m["message_id"] for m in newer_window] == ["recent-1", "recent-2"]


def test_history_window_can_scroll_by_smaller_count(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)

    async def scenario():
        await _seed_linear_messages(20)
        return await database.load_chat_messages(SESSION_KEY, limit=10)

    live_context = asyncio.run(scenario())
    session = FakeSession(live_context)
    session.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 10}

    up_result = scroll_up(session, count=6)
    assert up_result["moved"] is True
    assert up_result["count"] == 6
    assert session.chat_window_view["top_db_id"] == _id_for(db_path, "msg-05")

    older_window = load_history_window(session, session.chat_window_view["top_db_id"], 10)
    assert [m["message_id"] for m in older_window] == [
        "msg-05",
        "msg-06",
        "msg-07",
        "msg-08",
        "msg-09",
        "msg-10",
        "msg-11",
        "msg-12",
        "msg-13",
        "msg-14",
    ]

    down_result = scroll_down(session, count=3)
    assert down_result["moved"] is True
    assert down_result["count"] == 3
    assert session.chat_window_view["mode"] == "history"
    assert session.chat_window_view["top_db_id"] == _id_for(db_path, "msg-08")

    newer_window = load_history_window(session, session.chat_window_view["top_db_id"], 10)
    assert [m["message_id"] for m in newer_window] == [
        "msg-08",
        "msg-09",
        "msg-10",
        "msg-11",
        "msg-12",
        "msg-13",
        "msg-14",
        "msg-15",
        "msg-16",
        "msg-17",
    ]


def test_scroll_chat_log_count_defaults_and_clamps_to_ten(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)

    async def scenario():
        await _seed_linear_messages(25)
        return await database.load_chat_messages(SESSION_KEY, limit=10)

    live_context = asyncio.run(scenario())
    session = FakeSession(live_context)
    session.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 10}

    result = make_scroll_chat_log_handler(session)(action="up", count=99)
    assert result["moved"] is True
    assert result["count"] == 10
    assert session.chat_window_view["top_db_id"] == _id_for(db_path, "msg-06")


def test_scroll_chat_log_can_jump_to_message_as_window_top(monkeypatch, tmp_path):
    db_path = _setup_db(monkeypatch, tmp_path)

    async def scenario():
        await _seed_linear_messages(12)
        return await database.load_chat_messages(SESSION_KEY, limit=10)

    live_context = asyncio.run(scenario())
    session = FakeSession(live_context)
    session.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 10}

    result = make_scroll_chat_log_handler(session)(action="jump", message_id="msg-04")

    assert result["ok"] is True
    assert result["moved"] is True
    assert result["message_id"] == "msg-04"
    assert session.chat_window_view["mode"] == "history"
    assert session.chat_window_view["top_db_id"] == _id_for(db_path, "msg-04")

    window = load_history_window(session, session.chat_window_view["top_db_id"], 10)
    assert [m["message_id"] for m in window] == [
        "msg-04",
        "msg-05",
        "msg-06",
        "msg-07",
        "msg-08",
        "msg-09",
        "msg-10",
        "msg-11",
        "msg-12",
    ]


def test_scroll_to_message_reports_missing_target_without_moving(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)
    asyncio.run(_seed_linear_messages(3))
    session = FakeSession()
    session.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 10}

    result = scroll_to_message(session, "missing")

    assert result["ok"] is False
    assert result["moved"] is False
    assert result["message_id"] == "missing"
    assert session.chat_window_view == {"mode": "live", "top_db_id": None, "page_size": 10}


def test_search_history_context_uses_message_time(monkeypatch, tmp_path):
    _setup_db(monkeypatch, tmp_path)
    asyncio.run(_seed_out_of_order_messages())

    session = FakeSession()
    result = make_search_history_handler(session)(
        keywords=["recent two"],
        context_window=1,
        limit=1,
    )

    assert result["total_hits"] == 1
    assert [m["content"] for m in result["results"][0]["context"]] == [
        "recent one",
        "recent two",
        "latest",
    ]
    assert result["results"][0]["context"][1]["is_hit"] is True

