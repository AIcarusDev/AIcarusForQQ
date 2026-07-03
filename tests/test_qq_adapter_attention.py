from __future__ import annotations

import asyncio
from types import SimpleNamespace

import app_state
import qq_adapter_handler
from llm.session import ChatSession, sessions
from qq_adapter.events import should_respond


def _group_event(message: list[dict]) -> dict:
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": 123,
        "sender": {"user_id": 456},
        "message": message,
    }


def test_group_at_all_is_mention_level():
    event = _group_event([{"type": "at", "data": {"qq": "all"}}])

    assert qq_adapter_handler._is_mention_level_message(
        event,
        event["message"],
        bot_id="bot",
    ) is True


def test_group_at_other_is_not_mention_level():
    event = _group_event([{"type": "at", "data": {"qq": "someone_else"}}])

    assert qq_adapter_handler._is_mention_level_message(
        event,
        event["message"],
        bot_id="bot",
    ) is False


def test_group_reply_to_bot_is_mention_level_without_double_counting_at_bot():
    event = _group_event([
        {"type": "reply", "data": {"id": "bot_msg_1"}},
        {"type": "at", "data": {"qq": "bot"}},
    ])

    assert qq_adapter_handler._is_mention_level_message(
        event,
        event["message"],
        bot_id="bot",
        reply_to_bot=True,
    ) is True


def test_private_message_is_mention_level_even_with_reply():
    event = {
        "post_type": "message",
        "message_type": "private",
        "sender": {"user_id": 456},
        "message": [{"type": "reply", "data": {"id": "any_msg"}}],
    }

    assert qq_adapter_handler._is_mention_level_message(
        event,
        event["message"],
        bot_id="bot",
    ) is True


def test_should_respond_self_name_can_be_disabled():
    event = _group_event([
        {"type": "text", "data": {"text": "AIcarus 看一下这个"}},
    ])

    assert should_respond(event, bot_id="bot", bot_name="AIcarus") is True
    assert should_respond(
        event,
        bot_id="bot",
        bot_name="AIcarus",
        respond_to_self_name=False,
    ) is False


def test_should_respond_at_all_even_when_self_name_disabled():
    event = _group_event([{"type": "at", "data": {"qq": "all"}}])

    assert should_respond(
        event,
        bot_id="bot",
        bot_name="AIcarus",
        respond_to_self_name=False,
    ) is True


def test_poke_to_bot_records_note_without_attention_wake(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        session = ChatSession()
        session.set_conversation_meta("group", "123", "Sandbox")
        sessions["group_123"] = session
        monkeypatch.setattr(app_state, "current_focus", "group_123")
        monkeypatch.setattr(app_state, "qq_adapter_cfg", {"whitelist": {"enabled": False}})
        monkeypatch.setattr(app_state, "qq_adapter_client", SimpleNamespace(bot_id="bot"))

        async def fake_get_display_name(platform, user_id, group_id=None):
            return {"456": "Alice", "bot": "Bot"}.get(str(user_id), str(user_id))

        async def fake_save_chat_message(*args, **kwargs):
            return None

        async def fake_upsert_chat_session(*args, **kwargs):
            return None

        def fail_dispatch(*args, **kwargs):
            raise AssertionError("poke should not dispatch mention-level wake")

        monkeypatch.setattr(qq_adapter_handler, "get_display_name", fake_get_display_name)
        monkeypatch.setattr(qq_adapter_handler, "save_chat_message", fake_save_chat_message)
        monkeypatch.setattr(qq_adapter_handler, "upsert_chat_session", fake_upsert_chat_session)
        monkeypatch.setattr(qq_adapter_handler, "_dispatch_wake_signals", fail_dispatch)

        asyncio.run(
            qq_adapter_handler._handle_qq_adapter_poke({
                "group_id": "123",
                "user_id": "456",
                "target_id": "bot",
                "action": "戳了戳",
            })
        )

        assert session.context_messages[-1]["content_type"] == "poke"
        assert session.sleep_pending_wake is False
        assert session.last_wake_reason == ""
    finally:
        sessions.clear()
        sessions.update(original_sessions)
