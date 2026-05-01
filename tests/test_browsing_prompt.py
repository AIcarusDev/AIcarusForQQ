import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import database
from database import init_db, save_chat_message
from llm.prompt.final_reminder import build_browsing_reminder
from llm.prompt.user_prompt_builder import _build_browsing_chat_log, build_main_user_prompt
from llm.session import create_session


def _entry(message_id: str, text: str) -> dict:
    return {
        "role": "user",
        "message_id": message_id,
        "sender_id": "42",
        "sender_name": "Alice",
        "sender_role": "member",
        "timestamp": "2026-04-29T00:00:00+08:00",
        "content": text,
        "content_type": "text",
        "content_segments": [{"type": "text", "text": text}],
    }


class BrowsingPromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(self._tmpdir.name, "test_browsing_prompt.db")
        await init_db()

    async def asyncTearDown(self) -> None:
        database.DB_PATH = self._old_db_path
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    async def test_browsing_chat_log_marks_history_without_fake_unread_count(self) -> None:
        session = create_session()
        session.set_conversation_meta("group", "1", "测试群")
        session._qq_id = "999"
        session._qq_name = "Bot"
        session.chat_window_view = {"mode": "history", "top_db_id": 1, "page_size": 10}
        session.unread_count = 3

        await save_chat_message("group_1", _entry("101", "hello"))

        chat_log = _build_browsing_chat_log(session)

        self.assertIsInstance(chat_log, str)
        self.assertIn('<chat_logs mode="history" has_previous="false">', chat_log)
        self.assertIn('<bubble>当前会话有 3 条未读新消息</bubble>', chat_log)

    async def test_browsing_chat_log_consumes_visible_unread_messages(self) -> None:
        session = create_session()
        session.set_conversation_meta("group", "1", "测试群")
        session._qq_id = "999"
        session._qq_name = "Bot"
        session.chat_window_view = {"mode": "history", "top_db_id": 3, "page_size": 2}

        for index in range(1, 6):
            await save_chat_message("group_1", _entry(str(100 + index), f"m{index}"))

        session.mark_unread_message("104")
        session.mark_unread_message("105")

        chat_log = _build_browsing_chat_log(session)

        self.assertIn('<chat_logs mode="history" has_previous="true">', chat_log)
        self.assertIn('<bubble>当前会话有 1 条未读新消息</bubble>', chat_log)
        self.assertEqual(session.unread_count, 1)

    async def test_current_chat_log_exposes_has_previous_without_bubble(self) -> None:
        session = create_session()
        session.set_conversation_meta("group", "1", "测试群")
        session._qq_id = "999"
        session._qq_name = "Bot"

        entries = [_entry("101", "m1"), _entry("102", "m2"), _entry("103", "m3")]
        for entry in entries:
            await save_chat_message("group_1", entry)

        session.add_to_context(entries[1])
        session.add_to_context(entries[2])

        prompt = build_main_user_prompt(session, consume_unread=False)

        self.assertIsInstance(prompt, str)
        self.assertIn('<chat_logs mode="current" has_previous="true">', prompt)
        self.assertNotIn('<bubble>', prompt)

    def test_browsing_reminder_keeps_original_template(self) -> None:
        session = create_session()
        session.chat_window_view = {"mode": "history", "top_db_id": 1, "page_size": 10}

        reminder = build_browsing_reminder(session)

        self.assertIn("你正在翻阅该聊天窗口的历史记录", reminder)
        self.assertNotIn("未读新消息", reminder)


if __name__ == "__main__":
    unittest.main()