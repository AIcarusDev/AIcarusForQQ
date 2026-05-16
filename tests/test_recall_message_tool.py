import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.core.tool_calling import process_tool_arguments
from tools.recall_message import DECLARATION, make_handler


class _FakeLoop:
    def is_running(self) -> bool:
        return True


class _FakeSession:
    def __init__(self, conv_type: str = "group", conv_id: str = "12345") -> None:
        self.conv_type = conv_type
        self.conv_id = conv_id
        self.conv_name = "测试会话"
        self._qq_id = "10000"
        self._qq_name = "AIcarus"
        self.context_messages: list[dict] = []

    def add_to_context(self, entry: dict) -> None:
        self.context_messages.append(entry)


class _FakeNapcat:
    def __init__(
        self,
        *,
        delete_response: dict | None = None,
        send_response: dict | None = None,
    ) -> None:
        self.connected = True
        self._loop = _FakeLoop()
        self.delete_response = delete_response or {"status": "ok"}
        self.send_response = send_response or {"message_id": 9001}
        self.raw_calls: list[tuple[str, dict]] = []
        self.api_calls: list[tuple[str, dict]] = []

    async def send_api_raw(self, action: str, params: dict) -> dict | None:
        self.raw_calls.append((action, params))
        return self.delete_response

    async def send_api(self, action: str, params: dict) -> dict | None:
        self.api_calls.append((action, params))
        return self.send_response


def _run_coroutine_sync(coro, *_args, **_kwargs):
    return asyncio.run(coro)


def _close_background_coro(coro, _loop):
    coro.close()
    return SimpleNamespace()


class RecallMessageToolTests(unittest.TestCase):
    def test_schema_accepts_legacy_arguments(self) -> None:
        result = process_tool_arguments(
            '{"message_id": "123", "motivation": "写错了，撤回"}',
            "recall_message",
            "test",
            DECLARATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["message_id"], 123)
        self.assertNotIn("edited_text", result.args)

    def test_schema_accepts_optional_edited_text(self) -> None:
        result = process_tool_arguments(
            '{"message_id": 123, "motivation": "修正错字", "edited_text": "修正后的内容"}',
            "recall_message",
            "test",
            DECLARATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["edited_text"], "修正后的内容")

    @patch("tools.recall_message.run_coroutine_sync", side_effect=_run_coroutine_sync)
    def test_blank_edited_text_does_not_send_replacement(self, _mock_run) -> None:
        session = _FakeSession()
        client = _FakeNapcat()
        handler = make_handler(session, client)

        result = handler(message_id=101, motivation="撤回", edited_text="   ")

        self.assertEqual(result["success"], True)
        self.assertFalse(result["edited_message_sent"])
        self.assertEqual(client.raw_calls, [("delete_msg", {"message_id": 101})])
        self.assertEqual(client.api_calls, [])
        self.assertEqual(session.context_messages, [])

    @patch("tools.recall_message.run_coroutine_sync", side_effect=_run_coroutine_sync)
    def test_delete_failure_does_not_send_replacement(self, _mock_run) -> None:
        session = _FakeSession()
        client = _FakeNapcat(delete_response={"status": "failed", "message": "too late"})
        handler = make_handler(session, client)

        result = handler(message_id=101, motivation="撤回", edited_text="修正后")

        self.assertIn("撤回消息失败", result["error"])
        self.assertEqual(client.raw_calls, [("delete_msg", {"message_id": 101})])
        self.assertEqual(client.api_calls, [])
        self.assertEqual(session.context_messages, [])

    @patch("tools.recall_message.asyncio.run_coroutine_threadsafe", side_effect=_close_background_coro)
    @patch("tools.recall_message.run_coroutine_sync", side_effect=_run_coroutine_sync)
    def test_successful_group_replacement_sends_and_records_text(self, _mock_run, _mock_bg) -> None:
        import app_state

        session = _FakeSession("group", "12345")
        client = _FakeNapcat(send_response={"message_id": 9001})
        handler = make_handler(session, client)

        with patch.object(app_state, "TIMEZONE", ZoneInfo("Asia/Shanghai")):
            result = handler(message_id=101, motivation="修正错字", edited_text="  修正后的内容  ")

        self.assertEqual(result["success"], True)
        self.assertTrue(result["edited_message_sent"])
        self.assertEqual(result["edited_message_id"], "9001")
        self.assertEqual(client.raw_calls, [("delete_msg", {"message_id": 101})])
        self.assertEqual(
            client.api_calls,
            [
                (
                    "send_msg",
                    {
                        "message_type": "group",
                        "group_id": 12345,
                        "message": [{"type": "text", "data": {"text": "修正后的内容"}}],
                    },
                )
            ],
        )
        self.assertEqual(session.context_messages[-1]["message_id"], "9001")
        self.assertEqual(session.context_messages[-1]["content"], "修正后的内容")
        self.assertEqual(
            session.context_messages[-1]["content_segments"],
            [{"type": "text", "text": "修正后的内容"}],
        )

    @patch("tools.recall_message.asyncio.run_coroutine_threadsafe", side_effect=_close_background_coro)
    @patch("tools.recall_message.run_coroutine_sync", side_effect=_run_coroutine_sync)
    def test_successful_private_replacement_uses_user_id(self, _mock_run, _mock_bg) -> None:
        import app_state

        session = _FakeSession("private", "54321")
        client = _FakeNapcat(send_response={"message_id": 9002})
        handler = make_handler(session, client)

        with patch.object(app_state, "TIMEZONE", ZoneInfo("Asia/Shanghai")):
            result = handler(message_id=102, motivation="修正错字", edited_text="私聊修正")

        self.assertTrue(result["edited_message_sent"])
        self.assertEqual(
            client.api_calls,
            [
                (
                    "send_msg",
                    {
                        "message_type": "private",
                        "user_id": 54321,
                        "message": [{"type": "text", "data": {"text": "私聊修正"}}],
                    },
                )
            ],
        )
        self.assertEqual(session.context_messages[-1]["message_id"], "9002")


if __name__ == "__main__":
    unittest.main()
