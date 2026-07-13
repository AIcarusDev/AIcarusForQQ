from __future__ import annotations

import json

from llm.core.tool_calling.aic_action import (
    AIC_ACTION_ERROR_NAME,
    build_aic_action_message,
    extract_cognition_text,
    parse_aic_action_calls,
    strip_schema_extensions,
)


def _arguments(call) -> dict:
    return json.loads(call.function.arguments)


def test_build_aic_action_message_strips_schema_extensions_and_escapes_namespace_names():
    declaration = {
        "name": "runtime_manage",
        "description": "manage runtime safely",
        "parameters": {
            "type": "object",
            "x-internal": True,
            "properties": {
                "seconds": {"type": "integer", "x-coerce-integer": True},
            },
        },
    }

    action_message = build_aic_action_message(
        [],
        namespace_blocks=[
            {
                "name": "core",
                "active": True,
                "declarations": [declaration],
            },
            {
                "name": "secret<tool>",
                "description": "",
                "active": False,
            },
        ],
    )

    assert action_message.startswith("<tools>")
    assert '"name":"runtime_manage"' in action_message
    assert "x-internal" not in action_message
    assert "x-coerce-integer" not in action_message
    assert '<namespace name="secret&lt;tool&gt;" description="" active="false"/>' in action_message
    assert "<hidden>" not in action_message


def test_build_aic_action_message_prefers_prompt_signatures_for_active_namespaces():
    action_message = build_aic_action_message(
        [],
        namespace_blocks=[
            {
                "name": "core",
                "active": True,
                "declarations": [
                    {
                        "name": "runtime_manage",
                        "description": "manage runtime safely",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "seconds": {
                                    "type": "integer",
                                    "description": "等待秒数",
                                },
                            },
                            "required": ["seconds"],
                        },
                    }
                ],
                "signatures": [
                    "// manage runtime safely\nruntime_manage(args: {\n  action: \"wait\";\n  seconds: number; // 等待秒数\n})"
                ],
            },
        ],
    )

    assert "runtime_manage(args:" in action_message
    assert "seconds: number" in action_message
    assert '"parameters"' not in action_message
    assert '"type":"object"' not in action_message


def test_strip_schema_extensions_is_recursive_without_mutating_original():
    schema = {"x-root": 1, "items": [{"x-child": 2, "type": "string"}]}

    stripped = strip_schema_extensions(schema)

    assert stripped == {"items": [{"type": "string"}]}
    assert "x-root" in schema


def test_parse_aic_action_calls_extracts_cognition_and_ordered_calls():
    raw = """
    <cognition>Check the current surface.</cognition>
    <motive>Confirm what changed before acting.</motive>
    <action>
      <tool_call>{"namespace":"core","name":"runtime_manage","arguments":{"action":"wait","seconds":1}}</tool_call>
      <tool_call>{"function":{"namespace":"core","name":"enter_qq_session","arguments":{"type":"group","id":"sandbox"}}}</tool_call>
    </action>
    """

    result = parse_aic_action_calls(raw)

    assert result.found_blocks is True
    assert result.errors == []
    assert result.cognition == "Check the current surface."
    assert result.motive == "Confirm what changed before acting."
    assert [call.function.namespace for call in result.tool_calls] == ["core", "core"]
    assert [call.function.name for call in result.tool_calls] == ["runtime_manage", "enter_qq_session"]
    assert _arguments(result.tool_calls[0]) == {"action": "wait", "seconds": 1}
    assert _arguments(result.tool_calls[1]) == {"type": "group", "id": "sandbox"}


def test_parse_aic_action_calls_joins_multiple_motive_blocks_and_allows_missing():
    multiple = parse_aic_action_calls(
        "<cognition>check</cognition>"
        "<motive>first</motive><motive>second</motive>"
        '<action><tool_call>{"name":"runtime_manage","arguments":{"action":"wait"}}</tool_call></action>'
    )
    missing = parse_aic_action_calls(
        "<cognition>check</cognition>"
        '<action><tool_call>{"name":"runtime_manage","arguments":{"action":"wait"}}</tool_call></action>'
    )

    assert multiple.motive == "first\n\nsecond"
    assert missing.motive == ""
    assert len(missing.tool_calls) == 1


def test_parse_aic_action_calls_recovers_top_level_arguments_for_known_tools():
    raw = '<tool_call>{"name":"browse_forward","id":"forward-demo"}</tool_call>'

    result = parse_aic_action_calls(raw)

    assert result.errors == []
    assert len(result.repairs) == 1
    assert "arguments: id" in result.repairs[0]
    assert result.tool_calls[0].function.name == "browse_forward"
    assert _arguments(result.tool_calls[0]) == {"id": "forward-demo"}


def test_parse_aic_action_calls_repairs_missing_json_closers():
    raw = (
        '<tool_call>{"name":"runtime_manage","arguments":{"action":"wait","seconds":'
        '{"scope":"world","condition":"any_change"}</tool_call>'
    )

    result = parse_aic_action_calls(raw)

    assert result.errors == []
    assert result.tool_calls[0].function.name == "runtime_manage"
    assert _arguments(result.tool_calls[0]) == {
        "action": "wait",
        "seconds": {"scope": "world", "condition": "any_change"},
    }
    assert any("closer" in note.lower() or "json" in note.lower() for note in result.repairs)


def test_parse_aic_action_calls_unescapes_xml_entities_in_string_arguments():
    raw = (
        '<tool_call>{"name":"send_message","arguments":{"messages":[{"segments":['
        '{"command":"text","content":"2 &gt; 1 &amp;&amp; 1 &lt; 2, say &quot;ok&quot;"}'
        ']}]}}</tool_call>'
    )

    result = parse_aic_action_calls(raw)

    assert result.errors == []
    assert result.repairs == []
    assert result.tool_calls[0].function.name == "send_message"
    assert _arguments(result.tool_calls[0]) == {
        "messages": [
            {
                "segments": [
                    {
                        "command": "text",
                        "content": '2 > 1 && 1 < 2, say "ok"',
                    }
                ]
            }
        ]
    }


def test_parse_aic_action_calls_accepts_entity_escaped_json_body():
    raw = (
        "<tool_call>"
        "{&quot;name&quot;:&quot;runtime_manage&quot;,&quot;arguments&quot;:{&quot;action&quot;:&quot;wait&quot;,&quot;seconds&quot;:1}}"
        "</tool_call>"
    )

    result = parse_aic_action_calls(raw)

    assert result.errors == []
    assert result.repairs == []
    assert result.tool_calls[0].function.name == "runtime_manage"
    assert _arguments(result.tool_calls[0]) == {"action": "wait", "seconds": 1}


def test_parse_aic_action_calls_returns_aic_action_error_call_for_bad_json():
    result = parse_aic_action_calls('<tool_call>{"name": </tool_call>')

    assert result.found_blocks is True
    assert result.errors
    assert result.tool_calls[0].function.name == AIC_ACTION_ERROR_NAME
    assert "raw" in _arguments(result.tool_calls[0])


def test_extract_cognition_text_joins_multiple_blocks():
    raw = "<cognition>first</cognition> ignored <cognition>second</cognition>"

    assert extract_cognition_text(raw) == "first\n\nsecond"
