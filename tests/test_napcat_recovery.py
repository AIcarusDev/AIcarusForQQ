import os
import sqlite3
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
import database
from database import init_db, save_chat_message
from llm.session import get_or_create_session, sessions
from napcat.recovery import RecoveryConfig, RecoveryTarget, _build_targets, _recover_target


class _FakeVisionBridge:
    def process_entry(self, entry: dict) -> None:
        return None


class _FakeNapcatClient:
    def __init__(self, responses: dict[tuple[str, str, bool], list[dict]], bot_id: str = "999") -> None:
        self.responses = responses
        self.bot_id = bot_id
        self.connected = True
        self.calls: list[tuple[str, dict]] = []

    async def send_api(self, action: str, params: dict, timeout: float = 20.0):
        del timeout
        self.calls.append((action, dict(params)))
        anchor = str(params.get("message_seq", "") or "")
        reverse = bool(params.get("reverse_order", False))
        return {"messages": list(self.responses.get((action, anchor, reverse), []))}

    async def send_api_raw(self, action: str, params: dict, timeout: float = 20.0):
        del timeout
        self.calls.append((action, dict(params)))
        anchor = str(params.get("message_seq", "") or "")
        reverse = bool(params.get("reverse_order", False))
        return {"status": "ok", "data": {"messages": list(self.responses.get((action, anchor, reverse), []))}}


def _group_message(message_id: str, sender_id: str, text: str, *, group_id: str = "1") -> dict:
    return {
        "message_id": message_id,
        "self_id": 999,
        "time": 1710000000 + int(message_id),
        "message_type": "group",
        "group_id": int(group_id),
        "sender": {
            "user_id": int(sender_id),
            "nickname": f"user_{sender_id}",
            "card": f"card_{sender_id}",
            "role": "member",
        },
        "message": [{"type": "text", "data": {"text": text}}],
    }


def _entry_from_message(message: dict) -> dict:
    sender = message["sender"]
    return {
        "role": "user",
        "message_id": str(message["message_id"]),
        "sender_id": str(sender["user_id"]),
        "sender_name": sender.get("card") or sender.get("nickname") or str(sender["user_id"]),
        "sender_role": sender.get("role", "member"),
        "timestamp": f"2026-04-29T00:00:{int(message['message_id']):02d}+08:00",
        "content": message["message"][0]["data"]["text"],
        "content_type": "text",
        "content_segments": [{"type": "text", "text": message["message"][0]["data"]["text"]}],
    }


class NapcatRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(self._tmpdir.name, "test_recovery.db")
        await init_db()

        self._old_timezone = app_state.TIMEZONE
        self._old_bot_name = app_state.BOT_NAME
        self._old_napcat_cfg = app_state.napcat_cfg
        self._old_vision_bridge = app_state.vision_bridge

        app_state.TIMEZONE = ZoneInfo("Asia/Shanghai")
        app_state.BOT_NAME = "AIcarus"
        app_state.napcat_cfg = {"whitelist": {"private_users": [], "group_ids": []}}
        app_state.vision_bridge = _FakeVisionBridge()

        sessions.clear()

    async def asyncTearDown(self) -> None:
        sessions.clear()
        database.DB_PATH = self._old_db_path
        app_state.TIMEZONE = self._old_timezone
        app_state.BOT_NAME = self._old_bot_name
        app_state.napcat_cfg = self._old_napcat_cfg
        app_state.vision_bridge = self._old_vision_bridge
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def _count_messages(self, session_key: str) -> int:
        with sqlite3.connect(database.DB_PATH) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM chat_messages WHERE session_key=? AND role='user'",
                (session_key,),
            ).fetchone()
        return int(row[0]) if row else 0

    async def test_save_chat_message_is_idempotent_for_same_message_id(self) -> None:
        entry = {
            "role": "user",
            "message_id": "101",
            "sender_id": "42",
            "sender_name": "Alice",
            "sender_role": "member",
            "timestamp": "2026-04-29T00:00:00+08:00",
            "content": "hello",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "hello"}],
        }

        await save_chat_message("group_1", entry)
        await save_chat_message("group_1", entry)

        self.assertEqual(self._count_messages("group_1"), 1)

    async def test_recover_target_backfills_older_history_without_touching_live_context(self) -> None:
        existing_messages = [
            _group_message("4", "42", "m4"),
            _group_message("5", "42", "m5"),
        ]
        for message in existing_messages:
            await save_chat_message("group_1", _entry_from_message(message))

        session = get_or_create_session("group_1")
        session.set_conversation_meta("group", "1", "测试群")
        session.context_messages = [_entry_from_message(message) for message in existing_messages]

        responses = {
            ("get_group_msg_history", "", False): existing_messages,
            ("get_group_msg_history", "4", False): [
                _group_message("1", "42", "m1"),
                _group_message("2", "42", "m2"),
                _group_message("3", "42", "m3"),
                _group_message("4", "42", "m4"),
            ],
            ("get_group_msg_history", "1", False): [_group_message("1", "42", "m1")],
        }
        client = _FakeNapcatClient(responses)

        recent_count, older_count = await _recover_target(
            client,
            RecoveryTarget("group_1", "group", "1", "测试群"),
            RecoveryConfig(page_size=4, max_pages_per_session=4, backfill_history=True),
        )

        self.assertEqual(recent_count, 0)
        self.assertEqual(older_count, 3)
        self.assertEqual(self._count_messages("group_1"), 5)
        self.assertEqual([msg["message_id"] for msg in session.context_messages], ["4", "5"])

    async def test_recover_target_recent_messages_inject_live_context(self) -> None:
        base_message = _group_message("3", "42", "m3")
        await save_chat_message("group_1", _entry_from_message(base_message))

        session = get_or_create_session("group_1")
        session.set_conversation_meta("group", "1", "测试群")
        session.context_messages = [_entry_from_message(base_message)]

        responses = {
            ("get_group_msg_history", "", False): [
                base_message,
                _group_message("4", "42", "m4"),
                _group_message("5", "42", "m5"),
            ]
        }
        client = _FakeNapcatClient(responses)

        recent_count, older_count = await _recover_target(
            client,
            RecoveryTarget("group_1", "group", "1", "测试群"),
            RecoveryConfig(page_size=5, max_pages_per_session=0, backfill_history=False),
        )

        self.assertEqual(recent_count, 2)
        self.assertEqual(older_count, 0)
        self.assertEqual(self._count_messages("group_1"), 3)
        self.assertEqual([msg["message_id"] for msg in session.context_messages], ["3", "4", "5"])
        self.assertEqual(session.unread_count, 2)

    async def test_recover_target_recent_messages_skip_context_while_browsing_history(self) -> None:
        base_message = _group_message("3", "42", "m3")
        await save_chat_message("group_1", _entry_from_message(base_message))

        session = get_or_create_session("group_1")
        session.set_conversation_meta("group", "1", "测试群")
        session.context_messages = [_entry_from_message(base_message)]
        session.chat_window_view = {"mode": "history", "top_db_id": 1, "page_size": 10}

        responses = {
            ("get_group_msg_history", "", False): [
                base_message,
                _group_message("4", "42", "m4"),
                _group_message("5", "42", "m5"),
            ]
        }
        client = _FakeNapcatClient(responses)

        recent_count, older_count = await _recover_target(
            client,
            RecoveryTarget("group_1", "group", "1", "测试群"),
            RecoveryConfig(page_size=5, max_pages_per_session=0, backfill_history=False),
        )

        self.assertEqual(recent_count, 2)
        self.assertEqual(older_count, 0)
        self.assertEqual(self._count_messages("group_1"), 3)
        self.assertEqual([msg["message_id"] for msg in session.context_messages], ["3"])
        self.assertEqual(session.unread_count, 2)

    async def test_build_targets_includes_whitelist_sessions(self) -> None:
        app_state.napcat_cfg = {
            "whitelist": {
                "private_users": ["42"],
                "group_ids": ["1"],
            }
        }

        targets = await _build_targets(
            RecoveryConfig(seed_from_whitelist=True, backfill_history=False)
        )

        keys = {target.session_key for target in targets}
        self.assertIn("private_42", keys)
        self.assertIn("group_1", keys)


if __name__ == "__main__":
    unittest.main()