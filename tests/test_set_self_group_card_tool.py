import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import database
from database import get_group_info, get_group_member_display_info, init_db, upsert_group
from llm.session import create_session, sessions
from tools import build_tools
from tools.set_self_group_card import make_handler, sanitize_semantic_args


class _FakeLoop:
    def is_running(self) -> bool:
        return True


class _FakeQQAdapter:
    def __init__(
        self,
        *,
        connected: bool = True,
        bot_id: str = "999",
        raw_response: dict | None = None,
        member_card: str = "新群名片",
    ) -> None:
        self.connected = connected
        self.bot_id = bot_id
        self._loop = _FakeLoop()
        self.raw_response = raw_response or {"status": "ok"}
        self.member_card = member_card
        self.raw_calls: list[tuple[str, dict]] = []
        self.api_calls: list[tuple[str, dict]] = []

    async def send_api_raw(self, action: str, params: dict, timeout: float = 15.0) -> dict | None:
        self.raw_calls.append((action, params))
        return self.raw_response

    async def send_api(self, action: str, params: dict, timeout: float = 15.0):
        self.api_calls.append((action, params))
        if action == "get_group_member_list":
            return [
                {
                    "user_id": int(self.bot_id) if str(self.bot_id).isdigit() else self.bot_id,
                    "nickname": "AIcarus",
                    "card": self.member_card,
                    "role": "admin",
                }
            ]
        if action == "get_group_info":
            return {"group_id": 12345, "group_name": "测试群", "member_count": 42}
        return None


def _run_coroutine_sync(coro, *_args, **_kwargs):
    return asyncio.run(coro)


class SetSelfGroupCardToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(self._tmpdir.name, "set_self_group_card.db")
        sessions.clear()
        asyncio.run(init_db())

    def tearDown(self) -> None:
        sessions.clear()
        database.DB_PATH = self._old_db_path
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    def _group_session(self):
        session = create_session()
        session.set_conversation_meta("group", "12345", "旧测试群", 10)
        session._qq_card = "旧群名片"
        return session

    def test_group_tool_is_latent_and_private_session_cannot_see_it(self) -> None:
        group_tools = build_tools(
            {},
            qq_adapter_client=_FakeQQAdapter(),
            session=self._group_session(),
            group_id="12345",
        )
        self.assertIn("set_self_group_card", group_tools.latent_names())
        self.assertNotIn("set_self_group_card", group_tools.active_names())

        private_session = create_session()
        private_session.set_conversation_meta("private", "12345", "测试私聊")
        private_tools = build_tools(
            {},
            qq_adapter_client=_FakeQQAdapter(),
            session=private_session,
            group_id=None,
        )
        self.assertNotIn("set_self_group_card", private_tools.latent_names())

    def test_semantic_sanitizer_trims_card_and_allows_blank_clear(self) -> None:
        repaired, changes, error = sanitize_semantic_args({"card": "  新群名片  "})
        self.assertIsNone(error)
        self.assertEqual(repaired["card"], "新群名片")
        self.assertEqual(changes, ["card: trimmed surrounding whitespace"])

        repaired, changes, error = sanitize_semantic_args({"card": "   "})
        self.assertIsNone(error)
        self.assertEqual(repaired["card"], "")
        self.assertEqual(changes, ["card: trimmed surrounding whitespace"])

    def test_success_confirms_remote_card_and_syncs_database_and_sessions(self) -> None:
        from tools import set_self_group_card

        session = self._group_session()
        sessions["group_12345"] = session
        other_session = self._group_session()
        sessions["group_12345_duplicate"] = other_session
        asyncio.run(upsert_group("12345", "旧测试群", "旧群名片", 10))

        client = _FakeQQAdapter(member_card="新群名片")
        handler = make_handler(client, session, "12345")

        with patch.object(
            set_self_group_card,
            "run_coroutine_sync",
            side_effect=_run_coroutine_sync,
        ):
            result = handler(card=" 新群名片 ", motivation="群里需要换一个自称")

        self.assertEqual(result["success"], True)
        self.assertTrue(result["verified"])
        self.assertTrue(result["synced"])
        self.assertEqual(
            client.raw_calls,
            [
                (
                    "set_group_card",
                    {"group_id": 12345, "user_id": 999, "card": "新群名片"},
                )
            ],
        )
        self.assertEqual(session._qq_card, "新群名片")
        self.assertEqual(other_session._qq_card, "新群名片")
        self.assertEqual(session.conv_name, "测试群")
        self.assertEqual(session.conv_member_count, 42)

        group_name, member_count, bot_card = asyncio.run(get_group_info("12345"))
        self.assertEqual((group_name, member_count, bot_card), ("测试群", 42, "新群名片"))
        member_info = asyncio.run(get_group_member_display_info("qq", "999", "12345"))
        self.assertEqual(member_info["card"], "新群名片")
        self.assertEqual(member_info["nickname"], "AIcarus")
        self.assertEqual(member_info["permission_level"], "admin")

    def test_clear_card_confirms_remote_empty_card_and_syncs_local_state(self) -> None:
        from tools import set_self_group_card

        session = self._group_session()
        sessions["group_12345"] = session
        asyncio.run(upsert_group("12345", "旧测试群", "旧群名片", 10))

        client = _FakeQQAdapter(member_card="")
        handler = make_handler(client, session, "12345")

        with patch.object(
            set_self_group_card,
            "run_coroutine_sync",
            side_effect=_run_coroutine_sync,
        ):
            result = handler(card="", motivation="清空群名片")

        self.assertEqual(result["success"], True)
        self.assertTrue(result["verified"])
        self.assertTrue(result["synced"])
        self.assertEqual(
            client.raw_calls,
            [
                (
                    "set_group_card",
                    {"group_id": 12345, "user_id": 999, "card": ""},
                )
            ],
        )
        self.assertEqual(session._qq_card, "")

        group_name, member_count, bot_card = asyncio.run(get_group_info("12345"))
        self.assertEqual((group_name, member_count, bot_card), ("测试群", 42, ""))
        member_info = asyncio.run(get_group_member_display_info("qq", "999", "12345"))
        self.assertEqual(member_info["card"], "")
        self.assertEqual(member_info["display"], "AIcarus")
        self.assertEqual(member_info["permission_level"], "admin")

    def test_unconfirmed_remote_card_does_not_sync_local_state(self) -> None:
        from tools import set_self_group_card

        session = self._group_session()
        asyncio.run(upsert_group("12345", "旧测试群", "旧群名片", 10))
        client = _FakeQQAdapter(member_card="旧群名片")
        handler = make_handler(client, session, "12345")

        with patch.object(
            set_self_group_card,
            "run_coroutine_sync",
            side_effect=_run_coroutine_sync,
        ):
            result = handler(card="新群名片", motivation="测试未生效")

        self.assertEqual(result["status"], "unconfirmed")
        self.assertFalse(result["synced"])
        self.assertEqual(session._qq_card, "旧群名片")
        self.assertEqual(asyncio.run(get_group_info("12345"))[2], "旧群名片")

    def test_failure_paths_do_not_call_QQAdapter_or_sync(self) -> None:
        disconnected = _FakeQQAdapter(connected=False)
        result = make_handler(disconnected, self._group_session(), "12345")(
            card="新群名片",
            motivation="测试",
        )
        self.assertIn("QQ adapter 未连接", result["error"])
        self.assertEqual(disconnected.raw_calls, [])

        no_bot_id = _FakeQQAdapter(bot_id="")
        result = make_handler(no_bot_id, self._group_session(), "12345")(
            card="新群名片",
            motivation="测试",
        )
        self.assertIn("bot_id 未初始化", result["error"])
        self.assertEqual(no_bot_id.raw_calls, [])

        private_session = create_session()
        private_session.set_conversation_meta("private", "12345", "测试私聊")
        connected = _FakeQQAdapter()
        result = make_handler(connected, private_session, "12345")(
            card="新群名片",
            motivation="测试",
        )
        self.assertIn("仅能在群聊会话中使用", result["error"])
        self.assertEqual(connected.raw_calls, [])

        missing_card = _FakeQQAdapter()
        result = make_handler(missing_card, self._group_session(), "12345")(
            motivation="测试",
        )
        self.assertIn("缺少 card 参数", result["error"])
        self.assertEqual(missing_card.raw_calls, [])


if __name__ == "__main__":
    unittest.main()
