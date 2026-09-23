from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

from PIL import Image

from llm.core.tool_calling.pipeline import process_tool_arguments
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from platforms.qq.adapter.conversation import format_adapter_error
from platforms.qq.adapter.segments import llm_segments_to_qq_adapter
from platforms.qq.tools.qq_social.send_message import send_message as send_mod


def test_get_declaration_always_returns_array_shape():
    decl_default = send_mod.get_declaration()
    decl_single_cfg = send_mod.get_declaration(config={"tools": {"send_message": {"shape": "single"}}})

    assert decl_default["parameters"]["required"] == ["messages"]
    assert "messages" in decl_default["parameters"]["properties"]
    assert decl_single_cfg["parameters"]["required"] == ["messages"]
    assert "messages" in decl_single_cfg["parameters"]["properties"]


def test_image_segment_accepts_one_agent_linux_path():
    declaration = send_mod.get_declaration(config={"tools": {"send_message": "array"}})
    result = process_tool_arguments(
        json.dumps({
            "messages": [{
                "segments": [{
                    "command": "image",
                    "path": "/home/agent/output/result.png",
                }],
            }],
        }),
        "send_message",
        "test",
        tool_declaration=declaration,
    )

    assert result.ok is True
    assert result.args["messages"][0]["segments"][0]["path"] == "/home/agent/output/result.png"


def test_image_segment_rejects_host_and_outside_agent_paths():
    declaration = send_mod.get_declaration(config={"tools": {"send_message": "array"}})
    for path in (r"C:\temp\result.png", "/tmp/result.png"):
        result = process_tool_arguments(
            json.dumps({
                "messages": [{
                    "segments": [{"command": "image", "path": path}],
                }],
            }),
            "send_message",
            "test",
            tool_declaration=declaration,
        )
        assert result.ok is False


def test_face_segment_schema_conversion_and_local_history():
    declaration = send_mod.get_declaration()
    result = process_tool_arguments(
        json.dumps({"messages": [{"segments": [
            {"command": "text", "content": "你好"},
            {"command": "face", "id": 14},
        ]}]}),
        "send_message",
        "test",
        tool_declaration=declaration,
    )
    assert result.ok is True
    segments = result.args["messages"][0]["segments"]
    prepared, error, _ = send_mod._prepare_sendable_segments(segments, SimpleNamespace())
    assert error is None
    assert llm_segments_to_qq_adapter(prepared) == [
        {"type": "text", "data": {"text": "你好"}},
        {"type": "face", "data": {"id": "14"}},
    ]
    assert llm_segments_to_qq_adapter([
        {"command": "face", "id": 14},
        {"command": "image", "_local_image_base64": "cG5nLWJ5dGVz"},
    ]) == [
        {"type": "face", "data": {"id": "14"}},
        {"type": "image", "data": {"file": "base64://cG5nLWJ5dGVz"}},
    ]
    text, content_segments, content_type = send_mod._extract_message_text(prepared)
    assert text == "你好[微笑]"
    assert content_segments == [
        {"type": "text", "text": "你好"},
        {"type": "face", "id": "14", "des": "/微笑"},
    ]
    assert content_type == "text"
    assert send_mod._extract_message_text([{"command": "face", "id": 364}])[2] == "face"
    super_text, super_segments, _ = send_mod._extract_message_text(
        [{"command": "face", "id": 364}], {364: "/超级赞"},
    )
    assert super_text == "[超级赞]"
    assert super_segments == [{"type": "face", "id": "364", "des": "/超级赞"}]

    rejected = process_tool_arguments(
        json.dumps({"messages": [{"segments": [{"command": "face", "id": "not-an-id"}]}]}),
        "send_message",
        "test",
        tool_declaration=declaration,
    )
    assert rejected.ok is False
    assert send_mod._prepare_sendable_segments(
        [{"command": "face", "id": -1}], SimpleNamespace(),
    )[1] == "face segment 需要非负整数 id。"
    assert send_mod._prepare_sendable_segments(
        [{"command": "face", "id": "bad"}], SimpleNamespace(),
    )[1] == "face segment 需要非负整数 id。"


def test_pending_face_send_matches_echo_by_id_even_if_name_changes():
    sent = [{"type": "face", "data": {"id": "364"}}]
    expected = send_mod._delivery_match_text(sent)
    echo = {
        "message_id": "qq-echo-1",
        "user_id": "bot-1",
        "time": time.time(),
        "message": [{"type": "face", "data": {
            "id": "364", "raw": {"faceText": "/超级赞"},
        }}],
    }
    assert send_mod._history_message_matches_pending_send(
        echo,
        bot_sender_id="bot-1",
        bot_sender_name="Bot",
        expected_text=expected,
        reply_id=None,
        sent_started_at=time.time() - 1,
        known_bot_message_ids=set(),
    )
    echo["message"][0]["data"]["id"] = "365"
    assert not send_mod._history_message_matches_pending_send(
        echo,
        bot_sender_id="bot-1",
        bot_sender_name="Bot",
        expected_text=expected,
        reply_id=None,
        sent_started_at=time.time() - 1,
        known_bot_message_ids=set(),
    )


def test_materialize_local_image_path_validates_and_embeds_bytes(monkeypatch, tmp_path):
    output = io.BytesIO()
    Image.new("RGB", (12, 8), (1, 2, 3)).save(output, format="PNG")
    raw = output.getvalue()
    host_path = tmp_path / "result.png"
    host_path.write_bytes(raw)

    class WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path):
            assert path == "/home/agent/output/result.png"
            yield SimpleNamespace(
                size=len(raw),
                host_path=str(host_path),
                workspace_path=path,
                name="result.png",
            )

    messages, error = asyncio.run(send_mod._materialize_local_image_paths(
        [{"segments": [{"command": "image", "path": "/home/agent/output/result.png"}]}],
        WorkspaceService(),
    ))

    assert error is None
    segment = messages[0]["segments"][0]
    assert "path" not in segment
    from llm.media.image_store import read_image
    assert read_image(segment["_local_image_ref"])["data"] == raw
    assert base64.b64decode(segment["_local_image_base64"]) == raw


def test_materialize_local_image_path_rejects_ambiguous_sources():
    messages, error = asyncio.run(send_mod._materialize_local_image_paths(
        [{"segments": [{
            "command": "image",
            "path": "/home/agent/output/result.png",
            "image_ref": "img_existing",
        }]}],
        object(),
    ))

    assert messages is None
    assert "只能提供一个" in error


def test_adapter_error_exposes_only_bounded_metadata():
    error = format_adapter_error(
        {
            "action": "send_msg",
            "status": "failed",
            "retcode": 1200,
            "message": r"C:\Users\private\AICQ.db",
            "wording": "/app/napcat/private/payload.bin",
        }
    )

    assert "send_msg" in error
    assert "retcode=1200" in error
    assert "private" not in error
    assert "AICQ.db" not in error
    assert "payload.bin" not in error


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
    assert len(notes) == 1


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
    assert len(changes) == 1


def test_array_shape_strictly_rejects_root_single_message_arguments_before_schema_validation():
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

    assert result.ok is False


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
    assert len(result.schema_changes) == 1


def test_build_tools_always_builds_array_declaration():
    state = NamespaceRuntimeState()
    state.open("qq_social", load_namespace_registry(), 1)
    collection = build_tools(
        {
            "platforms": {"qq": {"enabled": True}},
            "tools": {"send_message": {"message_shape": "single"}},
        },
        namespace_state=state,
        current_round=1,
        current_platform="qq",
        session=SimpleNamespace(conv_type="group"),
        qq_client=object(),
    )
    spec = collection.active_specs["qq_social.send_message"]
    assert spec.declaration["parameters"]["required"] == ["messages"]
    assert "messages" in spec.declaration["parameters"]["properties"]


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
    assert result["error"]
    assert result["failed_messages"][0]["index"] == 0
    assert result["failed_messages"][0]["reason"] == result["error"]
    assert "send_msg" in result["error"]
    assert "retcode=1200" in result["error"]
    assert "no such column" not in repr(result)
    assert fake_session.context_messages == []


def test_face_send_persists_a_typed_history_entry_without_real_qq(fake_session, monkeypatch, tmp_path):
    import app_state
    import database
    from platforms.chat.xml_builder import _render_content_xml
    from web import debug_server

    catalog = tmp_path / "face_config.json"
    catalog.write_text(
        json.dumps({"sysface": [{"QSid": "364", "QDes": "/超级赞", "AniStickerType": 1}]}),
        encoding="utf-8",
    )
    fake_session.key = "qq:group:1234"
    sent_messages = []

    class FakeClient:
        connected = True
        adapter = "napcat"

        async def send_message(self, **kwargs):
            sent_messages.append(kwargs["message"])
            return {"message_id": 12345}

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr(database, "save_chat_message", no_op)
    monkeypatch.setattr(debug_server, "broadcast_chat_event", no_op)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        handler = send_mod.make_handler(
            lambda: fake_session,
            FakeClient(),
            {"platforms": {"qq": {"adapter": {"face_config_path": str(catalog)}}}},
        )
        result = handler(messages=[{"segments": [{"command": "face", "id": 364}]}])
        rejected = handler(messages=[{"segments": [
            {"command": "text", "content": "一起发送"},
            {"command": "face", "id": 364},
        ]}])
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=2)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    assert result["sent_count"] == 1
    assert rejected["sent_count"] == 0
    assert rejected["failed_count"] == 1
    assert "超级表情只能单独" in rejected["error"]
    assert sent_messages == [[{"type": "face", "data": {"id": "364"}}]]
    entry = fake_session.context_messages[0]
    assert entry["content"] == "[超级赞]"
    assert entry["content_type"] == "face"
    assert entry["content_segments"] == [{"type": "face", "id": "364", "des": "/超级赞"}]
    assert _render_content_xml(entry) == '    <content type="face" id="364">/超级赞</content>'


def test_dual_use_face_can_be_mixed_while_super_only_face_cannot():
    assert send_mod._validate_face_composition(
        [{"command": "text", "content": "哭"}, {"command": "face", "id": 5}],
        {364},
    ) is None
    assert send_mod._validate_face_composition(
        [{"command": "face", "id": 364}], {364},
    ) is None
    assert send_mod._validate_face_composition(
        [{"command": "text", "content": "赞"}, {"command": "face", "id": 364}],
        {364},
    ) is not None


def test_prepare_sendable_segments_rejects_empty_or_unknown_sticker(fake_session):
    prepared, error, warnings = send_mod._prepare_sendable_segments([], fake_session)
    assert prepared is None
    assert error
    assert warnings == []
    prepared, error, warnings = send_mod._prepare_sendable_segments(
        [{"command": "sticker", "image_ref": "missing-sticker"}],
        fake_session,
    )
    assert prepared is None
    assert "missing-sticker" in error
    assert warnings == []


def test_high_risk_browser_image_stages_exact_message_without_sending(
    fake_session,
    monkeypatch,
):
    import app_state
    import browser
    from browser.image_confirmation import current_pending

    monkeypatch.setattr(
        browser,
        "materialize_browser_resources",
        lambda refs: [{
            "resource_ref": refs[0],
            "image_ref": "img_" + "1" * 32,
            "sha256": "2" * 64,
            "confirmation_reasons": ["very_small_preview"],
        }],
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    old_loop = getattr(app_state, "main_loop", None)
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        handler = send_mod.make_handler(
            lambda: fake_session,
            SimpleNamespace(connected=False),
            {"browser_control": {"image_send_confirmation": "high_risk"}},
        )
        result = handler(messages=[{
            "quote": "message-1",
            "segments": [
                {"command": "text", "content": "caption"},
                {"command": "image", "resource_ref": "br_" + "a" * 20},
            ],
        }])
    finally:
        monkeypatch.setattr(app_state, "main_loop", old_loop)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    assert result["confirmation_required"] is True
    assert result["sent_count"] == 0
    assert result["confirmation_reasons"] == ["very_small_preview"]
    pending = current_pending(fake_session)
    assert pending is not None
    assert pending.target == ("group", "1234", "")
    assert pending.inbound_revision == 0
    assert pending.messages == ({
        "quote": "message-1",
        "segments": [
            {"command": "text", "content": "caption"},
            {"command": "image", "image_ref": "img_" + "1" * 32},
        ],
    },)
    assert pending.artifacts[0]["sha256"] == "2" * 64
    assert fake_session.context_messages == []


def test_default_confirmation_off_does_not_create_an_extra_round(
    fake_session,
    monkeypatch,
):
    import app_state
    import browser
    from browser.image_confirmation import current_pending

    monkeypatch.setattr(
        browser,
        "materialize_browser_resources",
        lambda refs: [{
            "resource_ref": refs[0],
            "image_ref": "img_" + "3" * 32,
            "sha256": "4" * 64,
            "confirmation_reasons": ["resource_identity_unproven"],
        }],
    )
    fake_session.key = "qq:group:1234"
    monkeypatch.setattr(
        send_mod,
        "_prepare_sendable_segments",
        lambda _segments, _session: (None, "stop-before-adapter", []),
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    old_loop = getattr(app_state, "main_loop", None)
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        handler = send_mod.make_handler(
            lambda: fake_session,
            SimpleNamespace(connected=False),
            {},
        )
        result = handler(messages=[{
            "segments": [
                {"command": "image", "resource_ref": "br_" + "b" * 20},
            ],
        }])
    finally:
        monkeypatch.setattr(app_state, "main_loop", old_loop)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    assert result.get("confirmation_required") is not True
    assert result["error"] == "stop-before-adapter"
    assert current_pending(fake_session) is None


def test_new_inbound_revision_during_materialization_invalidates_send(
    fake_session,
    monkeypatch,
):
    import app_state
    import browser
    from browser.image_confirmation import current_pending

    fake_session.inbound_revision = 0

    def materialize(refs):
        fake_session.inbound_revision = 1
        return [{
            "resource_ref": refs[0],
            "image_ref": "img_" + "5" * 32,
            "sha256": "6" * 64,
            "confirmation_reasons": ["very_small_preview"],
        }]

    monkeypatch.setattr(browser, "materialize_browser_resources", materialize)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    old_loop = getattr(app_state, "main_loop", None)
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        handler = send_mod.make_handler(
            lambda: fake_session,
            SimpleNamespace(connected=False),
            {"browser_control": {"image_send_confirmation": "high_risk"}},
            round_inbound_revision=0,
        )
        result = handler(messages=[{
            "segments": [
                {"command": "image", "resource_ref": "br_" + "c" * 20},
            ],
        }])
    finally:
        monkeypatch.setattr(app_state, "main_loop", old_loop)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()

    assert result["interrupted"] is True
    assert result["error"]
    assert current_pending(fake_session) is None

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



