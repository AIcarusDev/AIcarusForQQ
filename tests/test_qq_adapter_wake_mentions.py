import asyncio
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
import database
from database import init_db, is_bot_chat_message, save_chat_message
from llm.session import get_or_create_session, sessions
from qq_adapter_handler import (
    _dispatch_wake_signals,
    _is_at_bot,
    _is_reply_to_bot_message,
    _is_reply_to_bot,
)


def _reply_segment(message_id: str) -> list[dict]:
    return [{"type": "reply", "data": {"id": message_id}}, {"type": "text", "data": {"text": "reply"}}]


class QQAdapterWakeMentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_focus = app_state.current_focus
        sessions.clear()
        app_state.current_focus = "group_1"

    def tearDown(self) -> None:
        sessions.clear()
        app_state.current_focus = self._old_focus

    def _session(self):
        session = get_or_create_session("group_1")
        session.context_messages = [
            {"role": "bot", "message_id": "bot-msg", "content": "bot said this"},
            {"role": "user", "message_id": "user-msg", "content": "user said this"},
        ]
        return session

    def test_reply_to_other_user_message_is_not_mentioned(self) -> None:
        session = self._session()

        self.assertFalse(_is_reply_to_bot(_reply_segment("user-msg"), session))

        session.wait_event = asyncio.Event()
        session.wait_early_trigger = {"scope": "global", "condition": "mentioned"}
        session.sleep_wake_event = asyncio.Event()

        _dispatch_wake_signals(
            session,
            "group_1",
            is_mention=False,
            wake_remark="被动激活",
        )

        self.assertFalse(session.wait_event.is_set())
        self.assertFalse(session.sleep_wake_event.is_set())
        self.assertEqual(session.last_wake_reason, "")

    def test_reply_to_bot_message_is_mentioned(self) -> None:
        session = self._session()

        self.assertTrue(_is_reply_to_bot(_reply_segment("bot-msg"), session))

        session.wait_event = asyncio.Event()
        session.wait_early_trigger = {"scope": "global", "condition": "mentioned"}
        session.sleep_wake_event = asyncio.Event()

        _dispatch_wake_signals(
            session,
            "group_1",
            is_mention=True,
            wake_remark="收到回复，被动激活",
        )

        self.assertTrue(session.wait_event.is_set())
        self.assertTrue(session.sleep_wake_event.is_set())
        self.assertEqual(session.last_wake_reason, "收到回复，被动激活")

    def test_reply_to_bot_while_waiting_does_not_wake_future_sleep(self) -> None:
        session = self._session()
        session.wait_event = asyncio.Event()
        session.wait_early_trigger = {"scope": "session", "condition": "any_message"}
        session.sleep_wake_event = None
        session.sleep_arming = False

        _dispatch_wake_signals(
            session,
            "group_1",
            is_mention=True,
            wake_remark="收到回复，被动激活",
        )

        self.assertTrue(session.wait_event.is_set())
        self.assertFalse(session.sleep_pending_wake)
        self.assertEqual(session.last_wake_reason, "")

    def test_reply_to_bot_during_sleep_arming_is_preserved(self) -> None:
        session = self._session()
        session.sleep_wake_event = None
        session.sleep_arming = True

        _dispatch_wake_signals(
            session,
            "group_1",
            is_mention=True,
            wake_remark="收到回复，被动激活",
        )

        self.assertTrue(session.sleep_pending_wake)
        self.assertEqual(session.last_wake_reason, "收到回复，被动激活")

    def test_at_bot_is_mentioned(self) -> None:
        self.assertTrue(_is_at_bot([{"type": "at", "data": {"qq": "999"}}], "999"))
        self.assertFalse(_is_at_bot([{"type": "at", "data": {"qq": "42"}}], "999"))


class QQAdapterWakeMentionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(self._tmpdir.name, "wake_mentions.db")
        await init_db()
        sessions.clear()

    async def asyncTearDown(self) -> None:
        sessions.clear()
        database.DB_PATH = self._old_db_path
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    async def test_reply_to_persisted_bot_message_is_mentioned(self) -> None:
        await save_chat_message("group_1", {
            "role": "bot",
            "message_id": "old-bot-msg",
            "sender_id": "999",
            "sender_name": "Iccc",
            "sender_role": "",
            "timestamp": "2026-05-18T23:00:00+08:00",
            "content": "old reply target",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "old reply target"}],
        })
        session = get_or_create_session("group_1")
        session.context_messages = []

        self.assertTrue(await is_bot_chat_message("group_1", "old-bot-msg"))
        self.assertTrue(
            await _is_reply_to_bot_message(_reply_segment("old-bot-msg"), session, "group_1")
        )
        self.assertFalse(
            await _is_reply_to_bot_message(_reply_segment("old-bot-msg"), session, "group_2")
        )


if __name__ == "__main__":
    unittest.main()
