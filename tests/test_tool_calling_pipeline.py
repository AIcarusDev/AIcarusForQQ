import _thread
import json
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm.core.internal_tool import InternalToolSpec
from llm.core.tool_calling import process_tool_arguments
from tools.delete_memory import DECLARATION as DELETE_MEMORY_DECLARATION
from tools.get_tools import DECLARATION as GET_TOOLS_DECLARATION
from tools.get_tools import sanitize_semantic_args as sanitize_get_tools_args
from tools.shift import DECLARATION as SHIFT_DECLARATION
from tools.shift import repair_schema_args as repair_shift_schema_args
from tools.send_message.send_message import (
    get_declaration,
    repair_schema_args as repair_send_message_schema_args,
    sanitize_semantic_args as sanitize_send_message_args,
)
from tools.specs import ToolCollection, ToolSpec


class _PrivateSession:
    conv_type = "private"


class _FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response) -> None:
        self.completions = _FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_forced_tool_response(arguments: str):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments=arguments),
                        )
                    ]
                )
            )
        ],
    )


def _make_tool_call_response(*tool_calls):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=list(tool_calls))
            )
        ],
    )


def _make_tool_call(name: str, arguments: str = "{}", call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


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

    def test_string_identifier_schema_repair(self) -> None:
        result = process_tool_arguments(
            '{"memory_id": 2221, "reason": "remove"}',
            "delete_memory",
            "test",
            DELETE_MEMORY_DECLARATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["memory_id"], "2221")
        self.assertIn("memory_id: 2221 -> '2221' (string id)", result.schema_changes)

    def test_string_identifier_schema_repair_for_plain_id_field(self) -> None:
        result = process_tool_arguments(
            '{"type": "group", "id": 12345, "motivation": "switch"}',
            "shift",
            "test",
            SHIFT_DECLARATION,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["id"], "12345")
        self.assertIn("id: 12345 -> '12345' (string id)", result.schema_changes)

    @patch("tools.shift._infer_missing_shift_type", return_value=("group", None))
    def test_shift_schema_repair_infers_missing_type(self, _mock_infer_type) -> None:
        result = process_tool_arguments(
            '{"id": 12345, "motivation": "switch"}',
            "shift",
            "test",
            SHIFT_DECLARATION,
            repair_shift_schema_args,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.args["id"], "12345")
        self.assertEqual(result.args["type"], "group")
        self.assertIn("inferred type='group' from id '12345'", result.schema_changes)

    @patch(
        "tools.shift._infer_missing_shift_type",
        return_value=(None, "会话 ID 12345 同时匹配好友和群，必须显式提供 type"),
    )
    def test_shift_schema_repair_failure_keeps_original_schema_error(self, _mock_infer_type) -> None:
        result = process_tool_arguments(
            '{"id": 12345, "motivation": "switch"}',
            "shift",
            "test",
            SHIFT_DECLARATION,
            repair_shift_schema_args,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.failure.message, "arguments do not satisfy schema")
        self.assertIn("$: 'type' is a required property", result.failure.details)
        self.assertNotIn(
            "会话 ID 12345 同时匹配好友和群，必须显式提供 type",
            result.failure.details,
        )

    def test_send_message_quote_repair_uses_description_hint(self) -> None:
        declaration = get_declaration()
        raw_arguments = json.dumps(
            {
                "motivation": "reply",
                "messages": [
                    {
                        "quote": 123456,
                        "segments": [
                            {
                                "command": "text",
                                "params": {"content": "hello"},
                            }
                        ],
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

        self.assertTrue(result.ok)
        self.assertEqual(result.args["messages"][0]["quote"], "123456")
        self.assertIn(
            "messages[0].quote: 123456 -> '123456' (string id)",
            result.schema_changes,
        )

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

    def test_send_message_hoists_nested_motivation_even_when_root_exists(self) -> None:
        declaration = get_declaration()
        raw_arguments = json.dumps(
            {
                "motivation": "晚安",
                "messages": [
                    {
                        "motivation": "晚安",
                        "segments": [
                            {
                                "command": "text",
                                "params": {"content": "晚安啦"},
                            }
                        ],
                    },
                    {
                        "motivation": "结束对话",
                        "segments": [
                            {
                                "command": "text",
                                "params": {"content": "你也早点睡呀"},
                            }
                        ],
                    },
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

        self.assertTrue(result.ok)
        self.assertEqual(result.args["motivation"], "晚安\n\n结束对话")
        self.assertNotIn("motivation", result.args["messages"][0])
        self.assertNotIn("motivation", result.args["messages"][1])
        self.assertIn(
            "hoisted messages[0].motivation, messages[1].motivation -> motivation",
            result.schema_changes,
        )

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

    def test_forced_tool_accepts_internal_tool_spec_processors(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        declaration = {
            "name": "lookup_topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        }

        def repairer(args: dict[str, object]) -> tuple[dict[str, object], list[str]]:
            if "query" in args or "search" not in args:
                return args, []
            repaired = dict(args)
            repaired["query"] = repaired.pop("search")
            return repaired, ["search -> query"]

        def sanitizer(args: dict[str, object]) -> tuple[dict[str, object], list[str], str | None]:
            repaired = dict(args)
            repaired["query"] = str(repaired["query"]).strip()
            return repaired, ["trimmed query"], None

        spec = InternalToolSpec(
            declaration=declaration,
            schema_repairer=repairer,
            semantic_sanitizer=sanitizer,
        )
        fake_client = _FakeClient(_make_forced_tool_response('{"search": "  topic  "}'))

        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = fake_client
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter._call_forced_tool(
            "system",
            "user",
            {},
            spec,
            log_tag="unit",
        )

        self.assertEqual(result, {"query": "topic"})
        self.assertEqual(
            fake_client.completions.calls[0]["tools"][0]["function"]["name"],
            "lookup_topic",
        )

    def test_call_one_round_ctrl_c_during_parallel_tool_returns_immediately(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        release_tool = threading.Event()

        def slow_tool(**kwargs):
            release_tool.wait(timeout=1.0)
            return {"ok": True}

        declaration = {
            "name": "slow_tool",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
        collection = ToolCollection(
            active_specs={
                "slow_tool": ToolSpec(
                    name="slow_tool",
                    declaration=declaration,
                    handler=slow_tool,
                    module_name="tests.slow_tool",
                )
            }
        )

        fake_client = _FakeClient(
            _make_tool_call_response(_make_tool_call("slow_tool"))
        )
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = fake_client
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        interrupt_timer = threading.Timer(0.05, _thread.interrupt_main)
        interrupt_timer.daemon = True
        release_timer = threading.Timer(0.5, release_tool.set)
        release_timer.daemon = True
        interrupt_timer.start()
        release_timer.start()

        started_at = time.perf_counter()
        try:
            with self.assertRaises(KeyboardInterrupt):
                adapter.call_one_round(
                    lambda active, latent: "system",
                    "user",
                    {},
                    collection,
                )
        finally:
            interrupt_timer.cancel()
            release_timer.cancel()
            release_tool.set()

        elapsed = time.perf_counter() - started_at
        self.assertLess(
            elapsed,
            0.3,
            msg=f"ctrl+c 应在工具仍阻塞时立刻返回，实际耗时 {elapsed:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()