from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

from llm.core.tool_calling.pipeline import process_tool_arguments
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from platforms.qq.tools.qq_social.send_message import send_message as send_mod


def test_get_declaration_switches_between_array_and_single_shapes():
    array_decl = send_mod.get_declaration(config={"tools": {"send_message": "array"}})
    single_decl = send_mod.get_declaration(config={"tools": {"send_message": {"shape": "single"}}})

    assert array_decl["parameters"]["required"] == ["messages"]
    assert "messages" in array_decl["parameters"]["properties"]
    assert single_decl["parameters"]["required"] == ["segments"]
    assert "segments" in single_decl["parameters"]["properties"]


def test_repair_schema_args_splits_nested_message_objects_from_segments():
    args = {
        "messages": [
            {
                "segments": [
                    {"command": "text", "content": "first"},
                    {"segments": [{"command": "text", "content": "second"}]},
                ]
            }
        ]
    }

    repaired, notes = send_mod.make_schema_repairer(
        {"tools": {"send_message": {"message_shape": "array"}}}
    )(args)

    assert repaired["messages"] == [
        {"segments": [{"command": "text", "content": "first"}]},
        {"segments": [{"command": "text", "content": "second"}]},
    ]
    assert notes == ["split leaked message objects from messages[0].segments"]


def test_sanitize_semantic_args_splits_consecutive_text_segments():
    args = {
        "messages": [
            {
                "segments": [
                    {"command": "text", "content": "one"},
                    {"command": "text", "content": "two"},
                    {"command": "at", "user_id": "u_alice"},
                    {"command": "text", "content": "three"},
                ]
            }
        ]
    }

    repaired, changes, error = send_mod.sanitize_semantic_args(args)

    assert error is None
    assert len(repaired["messages"]) == 2
    assert repaired["messages"][0]["segments"] == [{"command": "text", "content": "one"}]
    assert changes == ["expanded messages by splitting consecutive text segments (1 -> 2)"]


def test_array_shape_repairs_root_single_message_arguments_before_schema_validation():
    declaration = send_mod.get_declaration(config={"tools": {"send_message": "array"}})
    raw_arguments = json.dumps(
        {
            "segments": [{"command": "text", "content": "07.21元这个折扣价好可爱"}],
            "quote": "-7549",
        },
        ensure_ascii=False,
    )

    result = process_tool_arguments(
        raw_arguments,
        "send_message",
        "test",
        tool_declaration=declaration,
        schema_repairer=send_mod.make_schema_repairer(
            {"tools": {"send_message": {"message_shape": "array"}}}
        ),
        semantic_sanitizer=send_mod.sanitize_semantic_args,
    )

    assert result.ok is True
    assert result.args == {
        "messages": [
            {
                "quote": "-7549",
                "segments": [{"command": "text", "content": "07.21元这个折扣价好可爱"}],
            }
        ]
    }
    assert result.schema_changes == ("wrapped root single-message fields into messages[0]",)


def test_array_shape_repairs_nested_numeric_quote_through_refs():
    declaration = send_mod.get_declaration(config={"tools": {"send_message": "array"}})
    raw_arguments = json.dumps(
        {
            "messages": [
                {
                    "quote": 12345,
                    "segments": [{"command": "text", "content": "hi"}],
                }
            ]
        },
        ensure_ascii=False,
    )

    result = process_tool_arguments(
        raw_arguments,
        "send_message",
        "test",
        tool_declaration=declaration,
        schema_repairer=send_mod.make_schema_repairer(
            {"tools": {"send_message": {"message_shape": "array"}}}
        ),
        semantic_sanitizer=send_mod.sanitize_semantic_args,
    )

    assert result.ok is True
    assert result.args["messages"][0]["quote"] == "12345"
    assert result.schema_changes == ("messages[0].quote: 12345 -> '12345' (string id)",)


def test_build_tools_single_shape_preserves_root_single_message_arguments():
    state = NamespaceRuntimeState()
    state.open("qq_social", load_namespace_registry(), 1)
    collection = build_tools(
        {"tools": {"send_message": {"message_shape": "single"}}},
        namespace_state=state,
        current_round=1,
        current_platform="qq",
        session=SimpleNamespace(conv_type="group"),
        qq_client=object(),
    )
    spec = collection.active_specs["qq_social.send_message"]
    raw_arguments = json.dumps(
        {
            "segments": [{"command": "text", "content": "我在"}],
        },
        ensure_ascii=False,
    )

    result = process_tool_arguments(
        raw_arguments,
        "send_message",
        "test",
        tool_declaration=spec.declaration,
        schema_repairer=spec.schema_repairer,
        semantic_sanitizer=spec.semantic_sanitizer,
    )

    assert result.ok is True
    assert result.args == {"segments": [{"command": "text", "content": "我在"}]}
    assert result.schema_changes == ()


def test_coerce_execute_messages_accepts_single_message_shape():
    messages, error = send_mod._coerce_execute_messages(
        messages=None,
        segments=[{"command": "text", "content": "hello"}],
        quote="msg-1",
    )

    assert error is None
    assert messages == [{"segments": [{"command": "text", "content": "hello"}], "quote": "msg-1"}]


def test_send_message_from_history_snaps_chat_window_to_latest():
    class BrowsingSession:
        def __init__(self):
            self.chat_window_view = {"mode": "history", "top_db_id": 42, "page_size": 10}

        def is_browsing_history(self):
            return self.chat_window_view.get("mode") == "history"

        def reset_chat_window_view(self):
            self.chat_window_view = {"mode": "live", "top_db_id": None, "page_size": 10}

    session = BrowsingSession()

    assert send_mod._snap_chat_window_to_latest_for_send(session) is True
    assert session.chat_window_view == {"mode": "live", "top_db_id": None, "page_size": 10}
    assert send_mod._snap_chat_window_to_latest_for_send(session) is False


def test_resolve_send_target_formats_group_private_and_temp_targets():
    assert send_mod._resolve_send_target(SimpleNamespace(conv_type="group", conv_id="1234")) == (
        1234,
        None,
        None,
        None,
    )
    assert send_mod._resolve_send_target(SimpleNamespace(conv_type="private", conv_id="4321")) == (
        None,
        4321,
        None,
        None,
    )
    assert send_mod._resolve_send_target(
        SimpleNamespace(conv_type="temp", conv_id="77", temp_source_group_id="1234")
    ) == (None, 77, 1234, None)


def test_adapter_failed_send_returns_error_without_local_chat_entry(fake_session, monkeypatch):
    fake_session.key = "qq:group:1234"

    class FakeClient:
        connected = True
        adapter = "llonebot"

        def __init__(self):
            self.last_api_error = None

        async def send_message(self, **_kwargs):
            self.last_api_error = {
                "action": "send_msg",
                "status": "failed",
                "retcode": 1200,
                "message": "no such column: NaN",
                "wording": "no such column: NaN",
            }
            return None

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    import app_state

    old_loop = getattr(app_state, "main_loop", None)
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        handler = send_mod.make_handler(lambda: fake_session, FakeClient())
        result = handler(
            messages=[
                {
                    "quote": "零一万物是哪家的",
                    "segments": [{"command": "text", "content": "李开复的"}],
                }
            ]
        )
    finally:
        monkeypatch.setattr(app_state, "main_loop", old_loop)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    assert result["sent_count"] == 0
    assert result["failed_count"] == 1
    assert result["total_count"] == 1
    assert result["error"] == "QQ adapter 返回错误: send_msg / failed / retcode=1200 / no such column: NaN"
    assert result["failed_messages"] == [
        {
            "index": 0,
            "reason": "QQ adapter 返回错误: send_msg / failed / retcode=1200 / no such column: NaN",
        }
    ]
    assert fake_session.context_messages == []


def test_prepare_sendable_segments_rejects_empty_or_unknown_sticker(fake_session):
    prepared, error, warnings = send_mod._prepare_sendable_segments([], fake_session)
    assert prepared is None
    assert error
    assert warnings == []

    prepared, error, warnings = send_mod._prepare_sendable_segments(
        [{"command": "sticker", "sticker_id": "missing-sticker"}],
        fake_session,
    )
    assert prepared is None
    assert "missing-sticker" in error
    assert warnings == []


def test_history_confirmation_match_requires_self_quote_text_and_new_id():
    event = {
        "message_id": "-1174946",
        "time": 1782571,
        "user_id": "2136288",
        "sender": {"user_id": "2136288"},
        "message": [
            {"type": "reply", "data": {"id": "3263136"}},
            {"type": "text", "data": {"text": "重启好了"}},
        ],
    }

    assert send_mod._history_message_matches_pending_send(
        event,
        bot_sender_id="2136288",
        bot_sender_name="Icc",
        expected_text="重启好了",
        reply_id="3263136",
        sent_started_at=1782569,
        known_bot_message_ids={"-8683226"},
    )
    assert not send_mod._history_message_matches_pending_send(
        event,
        bot_sender_id="2136288",
        bot_sender_name="Icc",
        expected_text="重启好了",
        reply_id="3263136",
        sent_started_at=1782569,
        known_bot_message_ids={"-1174946"},
    )
    assert not send_mod._history_message_matches_pending_send(
        event,
        bot_sender_id="2136288",
        bot_sender_name="Icc",
        expected_text="重启好了",
        reply_id="different",
        sent_started_at=1782569,
        known_bot_message_ids=set(),
    )



