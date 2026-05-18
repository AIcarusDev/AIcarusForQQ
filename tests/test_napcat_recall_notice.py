import os
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
import database
from database import (
    init_db,
    load_chat_messages,
    save_chat_message,
    upsert_membership,
)
from llm.prompt.xml_builder import build_chat_log_xml
from llm.session import get_or_create_session, sessions
from napcat_handler import _handle_napcat_recall


class _FakeNapcatClient:
    bot_id = "999"
    connected = True

    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def send_api(self, action: str, params: dict, timeout: float = 15.0):
        self.calls.append((action, params))
        return self.responses.get(action)


class NapcatRecallNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        self._old_timezone = app_state.TIMEZONE
        self._old_bot_name = app_state.BOT_NAME
        self._old_napcat_client = app_state.napcat_client
        database.DB_PATH = os.path.join(self._tmpdir.name, "recall_notice.db")
        app_state.TIMEZONE = ZoneInfo("Asia/Shanghai")
        app_state.BOT_NAME = "AIcarus"
        app_state.napcat_client = None
        sessions.clear()
        await init_db()

    async def asyncTearDown(self) -> None:
        sessions.clear()
        database.DB_PATH = self._old_db_path
        app_state.TIMEZONE = self._old_timezone
        app_state.BOT_NAME = self._old_bot_name
        app_state.napcat_client = self._old_napcat_client
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    async def test_group_recall_persists_without_active_session_and_uses_operator_id(self) -> None:
        await upsert_membership("qq", "456", "123", nickname="AdminNick", cardname="AdminCard", permission_level="admin")
        await upsert_membership("qq", "999", "123", nickname="BotNick", cardname="BotCard")
        await save_chat_message("group_123", {
            "role": "bot",
            "message_id": "bot-msg",
            "sender_id": "999",
            "sender_name": "BotNick",
            "sender_role": "",
            "timestamp": "2026-05-18T23:00:00+08:00",
            "content": "old bot message",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "old bot message"}],
        })

        await _handle_napcat_recall({
            "notice_type": "group_recall",
            "group_id": 123,
            "user_id": 999,
            "operator_id": 456,
            "message_id": "bot-msg",
            "time": 1779120000,
        })

        rows = await load_chat_messages("group_123", limit=10)
        self.assertEqual(len(rows), 1)
        recalled = rows[0]
        self.assertEqual(recalled["role"], "note")
        self.assertEqual(recalled["content_type"], "recall")
        self.assertEqual(recalled["content"], "管理员 AdminCard 撤回了一条成员消息")
        self.assertEqual(recalled["message_id"], "bot-msg")
        self.assertEqual(recalled["content_segments"][0]["type"], "recall_notice")
        self.assertEqual(recalled["content_segments"][0]["operator"]["id"], "456")
        self.assertNotIn("sender", recalled["content_segments"][0])
        self.assertNotIn("message_id", recalled["content_segments"][0])
        self.assertNotIn("notice_type", recalled["content_segments"][0])

        xml = build_chat_log_xml(
            rows,
            {"type": "group", "id": "123", "name": "test", "bot_id": "999", "bot_name": "AIcarus"},
        )
        self.assertIn('<note timestamp="', xml)
        self.assertNotIn("group_recall", xml)
        self.assertNotIn('message_id="bot-msg"', xml)
        self.assertIn('<operator id="456" card="AdminCard" nickname="AdminNick"/>', xml)
        self.assertNotIn("<sender", xml)
        self.assertIn('<content type="recall">管理员 AdminCard 撤回了一条成员消息</content>', xml)

    async def test_group_recall_updates_active_session_when_present(self) -> None:
        await upsert_membership("qq", "456", "123", nickname="AdminNick", cardname="AdminCard", permission_level="admin")
        await upsert_membership("qq", "789", "123", nickname="UserNick", cardname="UserCard")
        session = get_or_create_session("group_123")
        session.context_messages = [{
            "role": "user",
            "message_id": "user-msg",
            "sender_id": "789",
            "sender_name": "UserNick",
            "sender_role": "member",
            "timestamp": "2026-05-18T23:00:00+08:00",
            "content": "old user message",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "old user message"}],
        }]
        await save_chat_message("group_123", session.context_messages[0])

        await _handle_napcat_recall({
            "notice_type": "group_recall",
            "group_id": 123,
            "user_id": 789,
            "operator_id": 456,
            "message_id": "user-msg",
            "time": 1779120000,
        })

        self.assertEqual(session.context_messages[0]["role"], "note")
        self.assertEqual(session.context_messages[0]["content"], "管理员 AdminCard 撤回了一条成员消息")
        self.assertEqual(session.context_messages[0]["content_segments"][0]["operator"]["id"], "456")
        self.assertNotIn("sender", session.context_messages[0]["content_segments"][0])

    async def test_group_self_recall_renders_plain_recall_operator(self) -> None:
        await upsert_membership("qq", "456", "123", nickname="UserNick", cardname="UserCard")
        await save_chat_message("group_123", {
            "role": "user",
            "message_id": "self-msg",
            "sender_id": "456",
            "sender_name": "UserNick",
            "sender_role": "member",
            "timestamp": "2026-05-18T23:00:00+08:00",
            "content": "old self message",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "old self message"}],
        })

        await _handle_napcat_recall({
            "notice_type": "group_recall",
            "group_id": 123,
            "user_id": 456,
            "operator_id": 456,
            "message_id": "self-msg",
            "time": 1779120000,
        })

        rows = await load_chat_messages("group_123", limit=10)
        xml = build_chat_log_xml(
            rows,
            {"type": "group", "id": "123", "name": "test", "bot_id": "999", "bot_name": "AIcarus"},
        )
        self.assertEqual(rows[0]["content"], "UserCard 撤回了一条消息")
        self.assertIn('<note timestamp="', xml)
        self.assertIn('<operator id="456" card="UserCard" nickname="UserNick"/>', xml)
        self.assertIn('<content type="recall">UserCard 撤回了一条消息</content>', xml)
        self.assertNotIn("group_recall", xml)
        self.assertNotIn('message_id="self-msg"', xml)

    async def test_group_owner_recall_renders_owner_label(self) -> None:
        await upsert_membership("qq", "111", "123", nickname="OwnerNick", cardname="OwnerCard", permission_level="owner")
        await upsert_membership("qq", "789", "123", nickname="UserNick", cardname="UserCard")
        await save_chat_message("group_123", {
            "role": "user",
            "message_id": "member-msg",
            "sender_id": "789",
            "sender_name": "UserNick",
            "sender_role": "member",
            "timestamp": "2026-05-18T23:00:00+08:00",
            "content": "old member message",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "old member message"}],
        })

        await _handle_napcat_recall({
            "notice_type": "group_recall",
            "group_id": 123,
            "user_id": 789,
            "operator_id": 111,
            "message_id": "member-msg",
            "time": 1779120000,
        })

        rows = await load_chat_messages("group_123", limit=10)
        xml = build_chat_log_xml(
            rows,
            {"type": "group", "id": "123", "name": "test", "bot_id": "999", "bot_name": "AIcarus"},
        )
        self.assertEqual(rows[0]["content"], "群主 OwnerCard 撤回了一条成员消息")
        self.assertIn('<operator id="111" card="OwnerCard" nickname="OwnerNick"/>', xml)
        self.assertIn('<content type="recall">群主 OwnerCard 撤回了一条成员消息</content>', xml)

    async def test_group_recall_fetches_missing_operator_role_from_napcat(self) -> None:
        app_state.napcat_client = _FakeNapcatClient({
            "get_group_member_info": {
                "group_id": 123,
                "user_id": 111,
                "nickname": "OwnerNick",
                "card": "OwnerCard",
                "role": "owner",
            },
        })
        await upsert_membership("qq", "789", "123", nickname="UserNick", cardname="UserCard")
        await save_chat_message("group_123", {
            "role": "user",
            "message_id": "remote-owner-msg",
            "sender_id": "789",
            "sender_name": "UserNick",
            "sender_role": "member",
            "timestamp": "2026-05-18T23:00:00+08:00",
            "content": "old member message",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "old member message"}],
        })

        await _handle_napcat_recall({
            "notice_type": "group_recall",
            "group_id": 123,
            "user_id": 789,
            "operator_id": 111,
            "message_id": "remote-owner-msg",
            "time": 1779120000,
        })

        rows = await load_chat_messages("group_123", limit=10)
        xml = build_chat_log_xml(
            rows,
            {"type": "group", "id": "123", "name": "test", "bot_id": "999", "bot_name": "AIcarus"},
        )
        self.assertEqual(rows[0]["content"], "群主 OwnerCard 撤回了一条成员消息")
        self.assertIn('<operator id="111" card="OwnerCard" nickname="OwnerNick"/>', xml)
        self.assertEqual(
            app_state.napcat_client.calls[0],
            ("get_group_member_info", {"group_id": "123", "user_id": "111", "no_cache": True}),
        )


if __name__ == "__main__":
    unittest.main()
