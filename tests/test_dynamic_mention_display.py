import os
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
import database
from database import init_db, upsert_membership
from llm.prompt.xml_builder import build_chat_log_xml
from llm.session import sessions
from qq_adapter_handler import _handle_qq_adapter_group_notice


class _FakeQQAdapterClient:
    bot_id = "999"
    connected = True

    async def send_api(self, action: str, params: dict, timeout: float = 8.0):
        del timeout
        if action == "get_group_member_info":
            return {
                "user_id": params["user_id"],
                "nickname": "Nick42",
                "card": "NoticeCard42",
                "role": "member",
            }
        return None


def _message_with_mention(uid: str, display: str) -> dict:
    return {
        "role": "user",
        "message_id": "101",
        "sender_id": "7",
        "sender_name": "Sender",
        "sender_role": "member",
        "timestamp": "2026-05-19T12:00:00+08:00",
        "content": f"{display} hello",
        "content_type": "text",
        "content_segments": [
            {"type": "mention", "uid": uid, "display": display},
            {"type": "text", "text": " hello"},
        ],
    }


class DynamicMentionDisplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        self._old_qq_adapter_cfg = app_state.qq_adapter_cfg
        self._old_qq_adapter_client = app_state.qq_adapter_client
        self._old_timezone = app_state.TIMEZONE
        database.DB_PATH = os.path.join(self._tmpdir.name, "mentions.db")
        app_state.qq_adapter_cfg = {"whitelist": {"enabled": False}}
        app_state.qq_adapter_client = _FakeQQAdapterClient()
        app_state.TIMEZONE = ZoneInfo("Asia/Shanghai")
        sessions.clear()
        await init_db()

    async def asyncTearDown(self) -> None:
        sessions.clear()
        database.DB_PATH = self._old_db_path
        app_state.qq_adapter_cfg = self._old_qq_adapter_cfg
        app_state.qq_adapter_client = self._old_qq_adapter_client
        app_state.TIMEZONE = self._old_timezone
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    async def test_group_mentions_are_rendered_with_current_member_card(self) -> None:
        await upsert_membership(
            "qq",
            "42",
            "1",
            nickname="Nick42",
            cardname="NewCard42",
            permission_level="member",
        )
        entry = _message_with_mention("42", "@OldCard42")

        xml = build_chat_log_xml([entry], {"type": "group", "id": "1"})

        self.assertIn('<at uid="42">@NewCard42</at>', xml)
        self.assertEqual(entry["content_segments"][0]["display"], "@OldCard42")

    async def test_self_mentions_are_rendered_with_current_bot_card(self) -> None:
        entry = _message_with_mention("self", "@OldBotCard")

        xml = build_chat_log_xml(
            [entry],
            {
                "type": "group",
                "id": "1",
                "bot_id": "999",
                "bot_name": "BotNick",
                "bot_card": "BotNewCard",
            },
        )

        self.assertIn('<at uid="self">@BotNewCard</at>', xml)

    async def test_legacy_bot_id_mentions_are_rendered_as_self_display(self) -> None:
        entry = _message_with_mention("999", "@OldBotCard")

        xml = build_chat_log_xml(
            [entry],
            {
                "type": "group",
                "id": "1",
                "bot_id": "999",
                "bot_name": "BotNick",
                "bot_card": "BotNewCard",
            },
        )

        self.assertIn('<at uid="999">@BotNewCard</at>', xml)

    async def test_group_card_notice_refreshes_future_mention_rendering(self) -> None:
        entry = _message_with_mention("42", "@OldCard42")

        await _handle_qq_adapter_group_notice({
            "notice_type": "group_card",
            "group_id": 1,
            "user_id": 42,
            "card_old": "OldCard42",
            "card_new": "NoticeCard42",
        })
        xml = build_chat_log_xml([entry], {"type": "group", "id": "1"})

        self.assertIn('<at uid="42">@NoticeCard42</at>', xml)

    async def test_group_sender_state_renders_title_and_level(self) -> None:
        await upsert_membership(
            "qq",
            "7",
            "1",
            nickname="SenderNick",
            cardname="SenderCard",
            title="专属头衔",
            level="42",
            permission_level="admin",
        )
        entry = {
            "role": "user",
            "message_id": "102",
            "sender_id": "7",
            "sender_name": "SenderCard",
            "sender_role": "admin",
            "timestamp": "2026-05-19T12:00:00+08:00",
            "content": "hello",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "hello"}],
        }

        xml = build_chat_log_xml([entry], {"type": "group", "id": "1"})

        self.assertIn(
            '<sender id="7" nickname="SenderCard" role="admin" title="专属头衔" level="42"/>',
            xml,
        )
        self.assertNotIn("honors=", xml)
