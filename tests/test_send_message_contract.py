from __future__ import annotations

import json
from types import SimpleNamespace

from llm.core.tool_calling.pipeline import process_tool_arguments
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
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
        schema_repairer=send_mod.repair_schema_args,
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


def test_build_tools_single_shape_preserves_root_single_message_arguments():
    state = NamespaceRuntimeState()
    state.open("qq_social", load_namespace_registry(), 1)
    collection = build_tools(
        {"tools": {"send_message": {"message_shape": "single"}}},
        namespace_state=state,
        current_round=1,
        session=SimpleNamespace(conv_type="group"),
        qq_adapter_client=object(),
    )
    spec = collection.active_specs["send_message"]
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
