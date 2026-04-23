import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.core.tool_calling import process_tool_arguments
from tools.get_tools import DECLARATION as GET_TOOLS_DECLARATION
from tools.get_tools import sanitize_semantic_args as sanitize_get_tools_args
from tools.send_message.send_message import (
    get_declaration,
    repair_schema_args as repair_send_message_schema_args,
    sanitize_semantic_args as sanitize_send_message_args,
)
from tools.specs import ToolCollection, ToolSpec


class _PrivateSession:
    conv_type = "private"


class ToolCallingPipelineTests(unittest.TestCase):
    def test_integer_schema_repair(self) -> None:
        declaration = {
            "name": "set_volume",
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                    }
                },
                "required": ["level"],
            },
        }

        result = process_tool_arguments('{"level": "12"}', "set_volume", "test", declaration)

        self.assertTrue(result.ok)
        self.assertEqual(result.args["level"], 10)
        self.assertIn("level: '12' -> 12 (int)", result.schema_changes)

    def test_send_message_hoists_nested_motivation(self) -> None:
        declaration = get_declaration()
        raw_arguments = json.dumps(
            {
                "messages": [
                    {
                        "motivation": "reply now",
                        "segments": [
                            {
                                "command": "text",
                                "params": {"content": "hello"},
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )

        result = process_tool_arguments(
            raw_arguments,
            "send_message",
            "test",
            declaration,
            repair_send_message_schema_args,
            sanitize_send_message_args,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["motivation"], "reply now")
        self.assertNotIn("motivation", result.args["messages"][0])

    def test_private_send_message_rejects_at_segment(self) -> None:
        declaration = get_declaration(_PrivateSession())
        raw_arguments = json.dumps(
            {
                "motivation": "reply",
                "messages": [
                    {
                        "segments": [
                            {
                                "command": "at",
                                "params": {"user_id": "42"},
                            }
                        ]
                    }
                ],
            },
            ensure_ascii=False,
        )

        result = process_tool_arguments(
            raw_arguments,
            "send_message",
            "test",
            declaration,
            repair_send_message_schema_args,
            sanitize_send_message_args,
        )

        self.assertFalse(result.ok)
        self.assertIsNotNone(result.failure)
        self.assertEqual(result.failure.stage, "schema")

    def test_parse_failure_stops_pipeline(self) -> None:
        result = process_tool_arguments(
            "not-json",
            "send_message",
            "test",
            get_declaration(),
            repair_send_message_schema_args,
            sanitize_send_message_args,
        )

        self.assertFalse(result.ok)
        self.assertIsNotNone(result.failure)
        self.assertEqual(result.failure.stage, "parse")

    def test_get_tools_semantic_normalization(self) -> None:
        result = process_tool_arguments(
            '{"tool_names": [" get_user_avatar ", "get_user_avatar", "", "get_weather"]}',
            "get_tools",
            "test",
            GET_TOOLS_DECLARATION,
            semantic_sanitizer=sanitize_get_tools_args,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["tool_names"], ["get_user_avatar", "get_weather"])
        self.assertGreaterEqual(len(result.semantic_changes), 2)

    def test_tool_collection_activation_moves_spec(self) -> None:
        active_spec = ToolSpec(
            name="active_tool",
            declaration={"name": "active_tool"},
            handler=lambda **kwargs: {"ok": True},
            module_name="tests.active_tool",
        )
        latent_spec = ToolSpec(
            name="latent_tool",
            declaration={"name": "latent_tool"},
            handler=lambda **kwargs: {"ok": True},
            module_name="tests.latent_tool",
            always_available=False,
        )
        collection = ToolCollection(
            active_specs={"active_tool": active_spec},
            latent_specs={"latent_tool": latent_spec},
        )

        activated = collection.activate("latent_tool")

        self.assertIs(activated, latent_spec)
        self.assertIn("latent_tool", collection.active_specs)
        self.assertNotIn("latent_tool", collection.latent_specs)


if __name__ == "__main__":
    unittest.main()