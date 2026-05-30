import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import app_state
import database
from database import init_db, upsert_chat_session
from llm.prompt.unread_builder import build_unread_info_xml
from llm.prompt.xml_builder import build_chat_log_xml
from llm.session import create_session, sessions
from qq_adapter.client import QQAdapterClient
from qq_adapter.events import get_conversation_id
from tools.get_contact_list import make_handler as make_contact_list_handler
from tools.send_message.send_message import make_handler as make_send_message_handler
from tools.shift import DECLARATION as SHIFT_DECLARATION
from tools.shift import _list_shift_type_candidates, _resolve_shift_target, _resolve_temp_target, execute as execute_shift


class TempSessionIdentityTests(unittest.TestCase):
    def test_group_temp_private_event_uses_user_unique_session_key(self) -> None:
        event = {
            "post_type": "message",
            "message_type": "private",
            "sub_type": "group",
            "group_id": 10001,
            "sender": {"user_id": 42},
        }

        self.assertEqual(get_conversation_id(event), "temp_42")

    def test_friend_private_event_stays_private_session_key(self) -> None:
        event = {
            "post_type": "message",
            "message_type": "private",
            "sub_type": "friend",
            "sender": {"user_id": 42},
        }

        self.assertEqual(get_conversation_id(event), "private_42")

    def test_temp_xml_and_unread_are_marked_as_temp(self) -> None:
        session = create_session()
        session.set_conversation_meta(
            "temp",
            "42",
            "Alice",
            temp_source_group_id="10001",
            temp_source_group_name="测试群",
        )
        session.add_to_context({
            "role": "user",
            "message_id": "msg_1",
            "sender_id": "42",
            "sender_name": "Alice",
            "timestamp": "2026-05-30T12:00:00+08:00",
            "content": "hello",
            "content_type": "text",
            "content_segments": [{"type": "text", "text": "hello"}],
        })
        session.mark_unread_message("msg_1")

        xml = build_chat_log_xml(session.context_messages, session._get_conv_meta())
        self.assertIn('<conversation type="temp" id="42" user_id="42"', xml)
        self.assertIn('source_group_id="10001"', xml)
        self.assertIn('source_group_name="测试群"', xml)
        self.assertIn('<other id="42" name="Alice"/>', xml)
        self.assertIn('from="other"', xml)

        unread = build_unread_info_xml({"temp_42": session}, current_key="group_10001")
        self.assertIn('<session type="temp" id="42" user_id="42"', unread)
        self.assertIn('nickname="Alice"', unread)
        self.assertIn('source_group_id="10001"', unread)


class TempSessionAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self._old_db_path = database.DB_PATH
        self._old_focus = app_state.current_focus
        self._old_cfg = app_state.qq_adapter_cfg
        self._old_client = app_state.qq_adapter_client
        database.DB_PATH = os.path.join(self._tmpdir.name, "temp_sessions.db")
        app_state.qq_adapter_cfg = {"whitelist": {"enabled": False}}
        app_state.current_focus = None
        app_state.qq_adapter_client = None
        sessions.clear()
        await init_db()

    async def asyncTearDown(self) -> None:
        sessions.clear()
        database.DB_PATH = self._old_db_path
        app_state.current_focus = self._old_focus
        app_state.qq_adapter_cfg = self._old_cfg
        app_state.qq_adapter_client = self._old_client
        try:
            self._tmpdir.cleanup()
        except PermissionError:
            pass

    async def test_shift_temp_first_open_uses_current_group_only(self) -> None:
        class FakeClient:
            connected = True

            async def send_api(self, action: str, params: dict, timeout: float = 15.0):
                self.last_api_error = None
                self.last_call = (action, params)
                return {"user_id": params["user_id"], "nickname": "Alice"}

        group_session = create_session()
        group_session.set_conversation_meta("group", "10001", "测试群")
        sessions["group_10001"] = group_session
        app_state.current_focus = "group_10001"
        app_state.qq_adapter_client = FakeClient()

        resolved = await _resolve_temp_target("42")

        self.assertEqual(resolved["key"], "temp_42")
        self.assertEqual(resolved["type"], "temp")
        self.assertEqual(resolved["id"], "42")
        self.assertEqual(resolved["temp_source_group_id"], "10001")
        self.assertEqual(resolved["temp_source_group_name"], "测试群")

    async def test_shift_temp_first_open_fails_outside_group(self) -> None:
        resolved = await _resolve_temp_target("42")

        self.assertIn("当前焦点不是群聊", resolved["error"])

    async def test_shift_infers_registered_temp_as_private(self) -> None:
        await upsert_chat_session(
            "temp_42",
            "temp",
            "42",
            "Alice",
            temp_source_group_id="10001",
            temp_source_group_name="测试群",
        )

        candidates = await _list_shift_type_candidates("42")

        self.assertEqual(candidates, {"private"})

    async def test_shift_private_can_resolve_registered_temp_without_adapter(self) -> None:
        await upsert_chat_session(
            "temp_42",
            "temp",
            "42",
            "Alice",
            temp_source_group_id="10001",
            temp_source_group_name="测试群",
        )

        resolved = await _resolve_shift_target("private", "42")

        self.assertEqual(resolved["key"], "temp_42")
        self.assertEqual(resolved["type"], "temp")
        self.assertEqual(resolved["id"], "42")


class TempSessionToolTests(unittest.TestCase):
    def test_shift_schema_exposes_temp_as_private(self) -> None:
        type_schema = SHIFT_DECLARATION["parameters"]["properties"]["type"]

        self.assertEqual(type_schema["enum"], ["private", "group"])
        self.assertIn("包含临时会话", type_schema["description"])

    @patch("tools.shift.run_coroutine_sync")
    def test_shift_private_temp_target_reports_private(self, mock_run_coroutine_sync) -> None:
        class FakeLoop:
            def is_running(self) -> bool:
                return True

        def _run_and_close(coro, *_args, **_kwargs):
            if hasattr(coro, "close"):
                coro.close()
            if not hasattr(_run_and_close, "called"):
                _run_and_close.called = True
                return {
                    "key": "temp_42",
                    "type": "temp",
                    "id": "42",
                    "name": "Alice",
                    "temp_source_group_id": "10001",
                    "temp_source_group_name": "测试群",
                }
            return None

        old_loop = getattr(app_state, "main_loop", None)
        old_focus = app_state.current_focus
        old_sessions = dict(sessions)
        app_state.main_loop = FakeLoop()
        app_state.current_focus = "group_10001"
        sessions.clear()
        mock_run_coroutine_sync.side_effect = _run_and_close
        try:
            result = execute_shift("private", "42")
        finally:
            sessions.clear()
            sessions.update(old_sessions)
            app_state.main_loop = old_loop
            app_state.current_focus = old_focus

        self.assertTrue(result["ok"])
        self.assertEqual(result["now_focusing"]["type"], "private")
        self.assertEqual(result["focus_transition"]["to"], "qq_private_42")
        self.assertEqual(result["focus_transition"]["summary"], "qq_group_10001 -> qq_private_42")

    def test_get_contact_list_temp_uses_registered_sessions_without_adapter(self) -> None:
        class FakeLoop:
            def is_running(self) -> bool:
                return True

        old_loop = getattr(app_state, "main_loop", None)
        old_sessions = dict(sessions)
        app_state.main_loop = FakeLoop()
        sessions.clear()
        live = create_session()
        live.set_conversation_meta(
            "temp",
            "42",
            "Alice",
            temp_source_group_id="10001",
            temp_source_group_name="测试群",
        )
        sessions["temp_42"] = live
        def _close_and_return_empty(coro, *_args, **_kwargs):
            if hasattr(coro, "close"):
                coro.close()
            return []

        try:
            handler = make_contact_list_handler(None, {"qq_adapter": {"whitelist": {"enabled": False}}})
            with patch("tools.get_contact_list.run_coroutine_sync", side_effect=_close_and_return_empty):
                result = handler(type="temp")
        finally:
            sessions.clear()
            sessions.update(old_sessions)
            app_state.main_loop = old_loop

        self.assertEqual(result["temps"], [{
            "name": "Alice",
            "qqid": "42",
            "source_group_id": "10001",
            "source_group_name": "测试群",
        }])

    @patch("asyncio.run_coroutine_threadsafe")
    @patch("tools.send_message.send_message.run_coroutine_sync")
    def test_send_message_temp_passes_user_and_source_group(
        self,
        mock_run_coroutine_sync,
        mock_run_coroutine_threadsafe,
    ) -> None:
        class FakeLoop:
            def is_running(self) -> bool:
                return True

        class FakeFuture:
            pass

        class FakeCoro:
            def close(self) -> None:
                pass

        class FakeQQAdapter:
            connected = True
            last_api_error = None

            def __init__(self) -> None:
                self.calls: list[dict] = []

            def send_message(self, **kwargs):
                self.calls.append(kwargs)
                return FakeCoro()

        def _close_fire_and_forget(coro, *_args, **_kwargs):
            if hasattr(coro, "close"):
                coro.close()
            return FakeFuture()

        session = create_session()
        session.set_conversation_meta(
            "temp",
            "42",
            "Alice",
            temp_source_group_id="10001",
            temp_source_group_name="测试群",
        )
        fake_adapter = FakeQQAdapter()
        old_loop = getattr(app_state, "main_loop", None)
        app_state.main_loop = FakeLoop()
        mock_run_coroutine_sync.return_value = {"message_id": "sent_1"}
        mock_run_coroutine_threadsafe.side_effect = _close_fire_and_forget
        try:
            handler = make_send_message_handler(session, fake_adapter)
            result = handler(messages=[{
                "segments": [{"command": "text", "params": {"content": "hi"}}],
            }])
        finally:
            app_state.main_loop = old_loop

        self.assertEqual(result["sent_count"], 1)
        self.assertEqual(fake_adapter.calls[0]["user_id"], 42)
        self.assertIsNone(fake_adapter.calls[0]["group_id"])
        self.assertEqual(fake_adapter.calls[0]["temp_source_group_id"], 10001)

    def test_qq_adapter_client_temp_send_msg_params(self) -> None:
        class CapturingClient(QQAdapterClient):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[tuple[str, dict]] = []

            async def send_api(self, action: str, params: dict, timeout: float = 15.0):
                self.calls.append((action, params))
                return {"message_id": "1"}

        async def _run() -> dict:
            client = CapturingClient()
            await client.send_message(
                user_id=42,
                temp_source_group_id=10001,
                message=[{"type": "text", "data": {"text": "hi"}}],
                llm_elapsed=999,
            )
            return client.calls[0][1]

        params = asyncio.run(_run())
        self.assertEqual(params["message_type"], "private")
        self.assertEqual(params["user_id"], 42)
        self.assertEqual(params["group_id"], 10001)


class TempSessionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_load_chat_sessions_includes_temp_source_metadata(self) -> None:
        tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        old_db_path = database.DB_PATH
        database.DB_PATH = os.path.join(tmpdir.name, "temp_db.db")
        try:
            await init_db()
            await upsert_chat_session(
                "temp_42",
                "temp",
                "42",
                "Alice",
                temp_source_group_id="10001",
                temp_source_group_name="测试群",
            )
            rows = await database.load_chat_sessions()
        finally:
            database.DB_PATH = old_db_path
            tmpdir.cleanup()

        self.assertEqual(rows[0]["session_key"], "temp_42")
        self.assertEqual(rows[0]["temp_source_group_id"], "10001")
        self.assertEqual(rows[0]["temp_source_group_name"], "测试群")


if __name__ == "__main__":
    unittest.main()
