from __future__ import annotations

from types import SimpleNamespace

from tools.send_message import send_message as send_mod


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

    repaired, notes = send_mod.repair_schema_args(args)

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


def test_coerce_execute_messages_accepts_single_message_shape():
    messages, error = send_mod._coerce_execute_messages(
        messages=None,
        segments=[{"command": "text", "content": "hello"}],
        quote="msg-1",
    )

    assert error is None
    assert messages == [{"segments": [{"command": "text", "content": "hello"}], "quote": "msg-1"}]


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
