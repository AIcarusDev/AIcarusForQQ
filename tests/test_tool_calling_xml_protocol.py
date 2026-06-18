from __future__ import annotations

import json

from llm.core.tool_calling.xml_protocol import (
    XML_TOOL_CALL_ERROR_NAME,
    build_tools_xml_message,
    extract_cognition_text,
    parse_xml_tool_calls,
    strip_schema_extensions,
)


def _arguments(call) -> dict:
    return json.loads(call.function.arguments)


def test_build_tools_xml_message_strips_schema_extensions_and_escapes_hidden_names():
    declaration = {
        "name": "wait",
        "description": "wait safely",
        "parameters": {
            "type": "object",
            "x-internal": True,
            "properties": {
                "seconds": {"type": "integer", "x-coerce-integer": True},
            },
        },
    }

    xml = build_tools_xml_message([declaration], hidden_names=["secret<tool>"])

    assert xml.startswith("<tools>")
    assert '"name":"wait"' in xml
    assert "x-internal" not in xml
    assert "x-coerce-integer" not in xml
    assert "secret&lt;tool&gt;" in xml


def test_strip_schema_extensions_is_recursive_without_mutating_original():
    schema = {"x-root": 1, "items": [{"x-child": 2, "type": "string"}]}

    stripped = strip_schema_extensions(schema)

    assert stripped == {"items": [{"type": "string"}]}
    assert "x-root" in schema


def test_parse_xml_tool_calls_extracts_cognition_and_ordered_calls():
    raw = """
    <cognition>Check the current surface.</cognition>
    <action>
      <tool_call>{"name":"wait","arguments":{"seconds":1}}</tool_call>
      <tool_call>{"function":{"name":"shift","arguments":{"type":"group","id":"sandbox"}}}</tool_call>
    </action>
    """

    result = parse_xml_tool_calls(raw)

    assert result.found_blocks is True
    assert result.errors == []
    assert result.cognition == "Check the current surface."
    assert [call.function.name for call in result.tool_calls] == ["wait", "shift"]
    assert _arguments(result.tool_calls[0]) == {"seconds": 1}
    assert _arguments(result.tool_calls[1]) == {"type": "group", "id": "sandbox"}


def test_parse_xml_tool_calls_recovers_top_level_arguments_for_known_tools():
    raw = '<tool_call>{"name":"open_forward_message","id":"forward-demo"}</tool_call>'

    result = parse_xml_tool_calls(raw)

    assert result.errors == []
    assert len(result.repairs) == 1
    assert "arguments: id" in result.repairs[0]
    assert result.tool_calls[0].function.name == "open_forward_message"
    assert _arguments(result.tool_calls[0]) == {"id": "forward-demo"}


def test_parse_xml_tool_calls_repairs_missing_json_closers():
    raw = (
        '<tool_call>{"name":"wait","arguments":{"early_trigger":'
        '{"scope":"world","condition":"any_change"}</tool_call>'
    )

    result = parse_xml_tool_calls(raw)

    assert result.errors == []
    assert result.tool_calls[0].function.name == "wait"
    assert _arguments(result.tool_calls[0]) == {
        "early_trigger": {"scope": "world", "condition": "any_change"}
    }
    assert any("closer" in note.lower() or "json" in note.lower() for note in result.repairs)


def test_parse_xml_tool_calls_returns_protocol_error_call_for_bad_json():
    result = parse_xml_tool_calls('<tool_call>{"name": </tool_call>')

    assert result.found_blocks is True
    assert result.errors
    assert result.tool_calls[0].function.name == XML_TOOL_CALL_ERROR_NAME
    assert "raw" in _arguments(result.tool_calls[0])


def test_extract_cognition_text_joins_multiple_blocks():
    raw = "<cognition>first</cognition> ignored <cognition>second</cognition>"

    assert extract_cognition_text(raw) == "first\n\nsecond"
