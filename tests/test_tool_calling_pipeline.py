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

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.core.internal_tool import InternalToolSpec
from llm.core.tool_calling import process_tool_arguments
from llm.core.tool_calling.xml_protocol import parse_xml_tool_calls
from tools.not_used.delete_memory import DECLARATION as DELETE_MEMORY_DECLARATION
from tools.get_tools import DECLARATION as GET_TOOLS_DECLARATION
from tools.get_tools import sanitize_semantic_args as sanitize_get_tools_args
from tools.shift import DECLARATION as SHIFT_DECLARATION
from tools.shift import execute as execute_shift
from tools.shift import repair_schema_args as repair_shift_schema_args
from tools.resolve_goal import get_declaration as get_resolve_goal_declaration
from tools.send_message.send_message import (
    get_declaration,
    make_handler as make_send_message_handler,
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


def _make_text_response(content: str | None):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None)
            )
        ],
    )


def _make_tool_call(name: str, arguments: str = "{}", call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeStream:
    def __init__(self, chunks) -> None:
        self.chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


class _FakeStreamingCompletions:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.stream


class _FakeStreamingClient:
    def __init__(self, stream) -> None:
        self.completions = _FakeStreamingCompletions(stream)
        self.chat = SimpleNamespace(completions=self.completions)


def _make_stream_tool_call_delta(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=index,
                            id=call_id,
                            function=SimpleNamespace(name=name, arguments=arguments),
                        )
                    ],
                )
            )
        ],
    )


def _make_stream_content_delta(content: str | None):
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=None)
            )
        ],
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

    @patch("llm.session.get_or_create_session")
    @patch("tools.shift.run_coroutine_sync")
    def test_shift_execute_keeps_motivation_out_of_result(
        self,
        mock_run_coroutine_sync,
        mock_get_or_create_session,
    ) -> None:
        import app_state

        class FakeLoop:
            def is_running(self) -> bool:
                return True

        class FakeSession:
            conv_type = "group"
            conv_id = "12345"
            conv_name = "目标群"
            last_wake_reason = ""

            def set_conversation_meta(self, conv_type: str, conv_id: str) -> None:
                self.conv_type = conv_type
                self.conv_id = conv_id

        target = FakeSession()

        def _run_and_close(coro, *_args, **_kwargs):
            coro.close()
            return None

        mock_run_coroutine_sync.side_effect = _run_and_close
        mock_get_or_create_session.return_value = target

        old_loop = getattr(app_state, "main_loop", None)
        old_focus = getattr(app_state, "current_focus", None)
        from llm.session import create_session, sessions
        old_session = create_session()
        old_session.chat_window_view = {"mode": "history", "top_db_id": 1, "page_size": 10}
        old_session.forward_browser_stack.append({"forward_id": "old-fwd"})
        old_session.forward_virtual_registry["fwd:old:1"] = {"forward_id": "child-fwd"}
        replaced_session = sessions.get("group_999")
        sessions["group_999"] = old_session
        app_state.main_loop = FakeLoop()
        app_state.current_focus = "group_999"
        try:
            result = execute_shift("group", "12345", "切过去看一下")
        finally:
            app_state.main_loop = old_loop
            app_state.current_focus = old_focus
            if replaced_session is None:
                sessions.pop("group_999", None)
            else:
                sessions["group_999"] = replaced_session

        self.assertEqual(result["ok"], True)
        self.assertEqual(result["now_focusing"]["id"], "12345")
        self.assertNotIn("motivation", result)
        self.assertEqual(
            result["focus_transition"],
            {
                "from": "qq_group_999",
                "to": "qq_group_12345",
                "summary": "qq_group_999 -> qq_group_12345",
            },
        )
        self.assertEqual(target.last_wake_reason, "shift 自 group_999（动机：切过去看一下）")
        self.assertFalse(old_session.is_browsing_history())
        self.assertFalse(old_session.is_browsing_forward())
        self.assertEqual(old_session.forward_virtual_registry, {})

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

    def test_private_send_message_schema_keeps_at_segment_stable(self) -> None:
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

        self.assertTrue(result.ok)
        variants = declaration["parameters"]["properties"]["messages"]["items"]["properties"]["segments"]["items"]["oneOf"]
        commands = [
            variant["properties"]["command"]["enum"][0]
            for variant in variants
        ]
        self.assertEqual(commands, ["at", "text", "sticker"])

    def test_private_send_message_handler_fails_at_segment_without_sending(self) -> None:
        import app_state

        class FakeLoop:
            def is_running(self) -> bool:
                return True

        class FakeSession(_PrivateSession):
            conv_id = "12345"
            conv_name = "私聊"
            _qq_id = "bot"
            _qq_name = "Bot"
            context_messages: list[dict] = []

            def add_to_context(self, entry: dict) -> None:
                self.context_messages.append(entry)

        class FakeNapcat:
            connected = True

            async def send_message(self, **_kwargs):
                raise AssertionError("private at message must not reach NapCat")

        old_loop = getattr(app_state, "main_loop", None)
        app_state.main_loop = FakeLoop()
        try:
            handler = make_send_message_handler(FakeSession(), FakeNapcat())
            result = handler(
                motivation="test private at rejection",
                messages=[
                    {
                        "segments": [
                            {
                                "command": "at",
                                "params": {"user_id": "42"},
                            }
                        ],
                    }
                ],
            )
        finally:
            app_state.main_loop = old_loop

        self.assertEqual(result["sent_count"], 0)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["total_count"], 1)
        self.assertIn("私聊不支持 at", result["error"])
        self.assertEqual(result["failed_messages"][0]["index"], 0)

    def test_resolve_goal_schema_does_not_embed_active_goal_ids(self) -> None:
        from llm.prompt import goals

        old_goals = goals.get_all()
        try:
            goals.restore(
                [
                    {
                        "goal_id": "goal_live_1",
                        "created_at": 1,
                        "updated_at": 1,
                        "title": "t",
                        "content": "c",
                        "reason": "r",
                        "conv_type": "",
                        "conv_id": "",
                        "conv_name": "",
                        "status": "active",
                        "resolution": "",
                    }
                ]
            )

            declaration = get_resolve_goal_declaration()
        finally:
            goals.restore(old_goals)

        items_schema = declaration["parameters"]["properties"]["goal_ids"]["items"]
        self.assertEqual(items_schema, {"type": "string"})
        self.assertNotIn("goal_live_1", json.dumps(declaration, ensure_ascii=False))

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

    def test_tool_collection_uses_canonical_prompt_order(self) -> None:
        def spec(name: str, *, always_available: bool = True) -> ToolSpec:
            return ToolSpec(
                name=name,
                declaration={"name": name},
                handler=lambda **kwargs: {"ok": True},
                module_name=f"tests.{name}",
                always_available=always_available,
            )

        collection = ToolCollection(
            active_specs={
                "send_voice_message": spec("send_voice_message"),
                "unknown_b": spec("unknown_b"),
                "send_message": spec("send_message"),
                "sleep": spec("sleep"),
                "unknown_a": spec("unknown_a"),
            },
            latent_specs={
                "set_self_group_card": spec("set_self_group_card", always_available=False),
                "get_contact_list": spec("get_contact_list", always_available=False),
                "search_current_session_chat_history": spec(
                    "search_current_session_chat_history",
                    always_available=False,
                ),
            },
        )

        self.assertEqual(
            collection.active_names(),
            ["send_message", "sleep", "send_voice_message", "unknown_a", "unknown_b"],
        )
        self.assertEqual(
            [decl["name"] for decl in collection.active_declarations()],
            collection.active_names(),
        )
        self.assertEqual(
            collection.latent_names(),
            [
                "search_current_session_chat_history",
                "get_contact_list",
                "set_self_group_card",
            ],
        )

        collection.activate("get_contact_list")

        self.assertEqual(
            collection.active_names(),
            [
                "send_message",
                "sleep",
                "get_contact_list",
                "send_voice_message",
                "unknown_a",
                "unknown_b",
            ],
        )

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

        fake_client = _FakeClient(_make_text_response(
            '<tool_call>{"name":"slow_tool","arguments":{}}</tool_call>'
        ))
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

    def test_call_one_round_uses_xml_tools_message_without_native_tools(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        executed_args: list[dict] = []

        def record_tool(**kwargs):
            executed_args.append(kwargs)
            return {"ok": True}

        collection = ToolCollection(
            active_specs={
                "record_tool": ToolSpec(
                    name="record_tool",
                    declaration={
                        "name": "record_tool",
                        "description": "Record one value.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "integer", "x-note": "hidden"},
                            },
                            "required": ["value"],
                        },
                    },
                    handler=record_tool,
                    module_name="tests.record_tool",
                )
            },
            latent_specs={
                "latent_tool": ToolSpec(
                    name="latent_tool",
                    declaration={
                        "name": "latent_tool",
                        "description": "Hidden full schema should not be shown.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "secret": {"type": "string"},
                            },
                        },
                    },
                    handler=lambda **kwargs: {"ok": True},
                    module_name="tests.latent_tool",
                    always_available=False,
                )
            }
        )
        fake_client = _FakeClient(_make_text_response(
            '<tool_call>{"name":"record_tool","arguments":{"value":1}}</tool_call>'
        ))
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = fake_client
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
        )

        self.assertTrue(result.had_tool_call)
        self.assertEqual(executed_args, [{"value": 1}])
        call_kwargs = fake_client.completions.calls[0]
        self.assertNotIn("tools", call_kwargs)
        self.assertNotIn("tool_choice", call_kwargs)
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("<tools>", messages[1]["content"])
        self.assertIn("<activated>", messages[1]["content"])
        self.assertIn("<hidden>", messages[1]["content"])
        self.assertIn('"name": "record_tool"', messages[1]["content"])
        self.assertIn('<tool name="latent_tool" />', messages[1]["content"])
        self.assertNotIn("Hidden full schema should not be shown", messages[1]["content"])
        self.assertNotIn('"secret"', messages[1]["content"])
        self.assertNotIn("x-note", messages[1]["content"])
        self.assertEqual(messages[-1], {"role": "user", "content": "user"})

    def test_xml_tool_call_id_is_system_assigned(self) -> None:
        parsed = parse_xml_tool_calls(
            '<tool_call>{"id":"model_hallucinated_id",'
            '"name":"record_tool","arguments":{"value":1}}</tool_call>'
        )

        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertEqual(parsed.tool_calls[0].id, "call_1")

    def test_consciousness_flow_renders_xml_tool_history(self) -> None:
        flow = ConsciousnessFlow()
        flow.append_round(
            [ToolCall(name="record_tool", args={"value": 1}, call_id="call_1")],
            [ToolResponse(name="record_tool", response={"ok": True}, call_id="call_1")],
            cognition="先记录上下文。",
        )

        messages = flow.to_xml_messages()

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertIn("<cognition>先记录上下文。</cognition>", messages[0]["content"])
        self.assertIn("<tool_call>", messages[0]["content"])
        self.assertNotIn("tool_calls", messages[0])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("<tool_response>", messages[1]["content"])

        data, timestamps = flow.dump()
        restored = ConsciousnessFlow()
        restored.restore(data, timestamps)
        restored_messages = restored.to_xml_messages()
        self.assertIn("<cognition>先记录上下文。</cognition>", restored_messages[0]["content"])

    def test_call_one_round_without_xml_tool_call_keeps_had_tool_call_false(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        collection = ToolCollection(
            active_specs={
                "noop": ToolSpec(
                    name="noop",
                    declaration={"name": "noop", "parameters": {"type": "object"}},
                    handler=lambda **kwargs: {"ok": True},
                    module_name="tests.noop",
                )
            }
        )
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeClient(_make_text_response("我需要再想想。"))
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
        )

        self.assertFalse(result.had_tool_call)
        self.assertEqual(result.tool_calls_log, [])

    def test_call_one_round_ignores_provider_native_tool_calls_without_xml(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        executed = False

        def noop(**kwargs):
            nonlocal executed
            executed = True
            return {"ok": True}

        collection = ToolCollection(
            active_specs={
                "noop": ToolSpec(
                    name="noop",
                    declaration={"name": "noop", "parameters": {"type": "object"}},
                    handler=noop,
                    module_name="tests.noop",
                )
            }
        )
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeClient(_make_tool_call_response(_make_tool_call("noop")))
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
        )

        self.assertFalse(result.had_tool_call)
        self.assertEqual(result.tool_calls_log, [])
        self.assertFalse(executed)

    def test_malformed_xml_tool_call_is_recorded_as_protocol_error(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        collection = ToolCollection(
            active_specs={
                "noop": ToolSpec(
                    name="noop",
                    declaration={"name": "noop", "parameters": {"type": "object"}},
                    handler=lambda **kwargs: {"ok": True},
                    module_name="tests.noop",
                )
            }
        )
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeClient(_make_text_response("<tool_call>{bad</tool_call>"))
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
        )

        self.assertTrue(result.had_tool_call)
        self.assertEqual(result.tool_calls_log[0]["function"], "__xml_tool_call_error__")
        self.assertIn("工具调用协议错误", result.tool_calls_log[0]["result"]["error"])

    def test_xml_cognition_is_captured_without_affecting_tool_call(self) -> None:
        parsed = parse_xml_tool_calls(
            "<cognition>我先确认上下文。</cognition>\n"
            '<tool_call>{"name":"noop","arguments":{"ok":true}}</tool_call>'
        )

        self.assertEqual(parsed.cognition, "我先确认上下文。")
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertEqual(parsed.tool_calls[0].function.name, "noop")

    def test_call_one_round_exposes_cognition_to_output_tool_context(self) -> None:
        from llm.core.provider import OpenAICompatAdapter
        from llm.core.round_context import get_current_inner_state

        seen_inner_states: list[dict] = []

        def send_message(**kwargs):
            del kwargs
            seen_inner_states.append(get_current_inner_state())
            return {"ok": True}

        collection = ToolCollection(
            active_specs={
                "send_message": ToolSpec(
                    name="send_message",
                    declaration={
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                    handler=send_message,
                    module_name="tests.send_message",
                )
            }
        )
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeClient(_make_text_response(
            "<cognition>需要先回应用户。</cognition>\n"
            '<tool_call>{"name":"send_message","arguments":{}}</tool_call>'
        ))
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False
        flow = ConsciousnessFlow()

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
            flow,
        )

        self.assertTrue(result.had_tool_call)
        self.assertEqual(result.cognition, "需要先回应用户。")
        self.assertEqual(seen_inner_states, [
            {"cognition": "需要先回应用户。", "think": "需要先回应用户。"}
        ])
        self.assertIn("<cognition>需要先回应用户。</cognition>", flow.to_xml_messages()[0]["content"])

    def test_call_one_round_closes_stream_when_new_message_arrives(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        executed = False

        def send_message(**kwargs):
            nonlocal executed
            executed = True
            return {"ok": True}

        collection = ToolCollection(
            active_specs={
                "send_message": ToolSpec(
                    name="send_message",
                    declaration={
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                    handler=send_message,
                    module_name="tests.send_message",
                )
            }
        )

        stream = _FakeStream([
            _make_stream_content_delta(
                '<tool_call>{"name":"send_message","arguments":{"motivation":"old"'
            )
        ])
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeStreamingClient(stream)
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        checks = 0

        def new_message_checker() -> bool:
            nonlocal checks
            checks += 1
            return checks >= 2

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
            new_message_checker=new_message_checker,
        )

        self.assertTrue(result.new_message_during_thinking)
        self.assertFalse(result.had_tool_call)
        self.assertFalse(executed)
        self.assertTrue(stream.closed)
        self.assertTrue(adapter.client.completions.calls[0]["stream"])

    def test_call_one_round_streaming_ignores_native_tool_call_delta(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        executed = False

        def send_message(**kwargs):
            nonlocal executed
            executed = True
            return {"ok": True}

        collection = ToolCollection(
            active_specs={
                "send_message": ToolSpec(
                    name="send_message",
                    declaration={
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "motivation": {"type": "string"},
                            },
                            "required": ["motivation"],
                        },
                    },
                    handler=send_message,
                    module_name="tests.send_message",
                )
            }
        )

        stream = _FakeStream([
            _make_stream_tool_call_delta(
                call_id="native_1",
                name="send_message",
                arguments='{"motivation":"native"}',
            )
        ])
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeStreamingClient(stream)
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
            new_message_checker=lambda: False,
        )

        self.assertFalse(result.had_tool_call)
        self.assertEqual(result.tool_calls_log, [])
        self.assertFalse(executed)
        self.assertTrue(stream.closed)

    def test_call_one_round_streaming_tool_call_executes_without_interrupt(self) -> None:
        from llm.core.provider import OpenAICompatAdapter

        executed_args: list[dict] = []

        def send_message(**kwargs):
            executed_args.append(kwargs)
            return {"ok": True}

        collection = ToolCollection(
            active_specs={
                "send_message": ToolSpec(
                    name="send_message",
                    declaration={
                        "name": "send_message",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "motivation": {"type": "string"},
                            },
                            "required": ["motivation"],
                        },
                    },
                    handler=send_message,
                    module_name="tests.send_message",
                )
            }
        )

        stream = _FakeStream([
            _make_stream_content_delta(
                '<tool_call>{"name":"send_message","arguments":{"mot'
            ),
            _make_stream_content_delta(
                'ivation":"fresh"}}</tool_call>'
            ),
        ])
        adapter = object.__new__(OpenAICompatAdapter)
        adapter.client = _FakeStreamingClient(stream)
        adapter.provider = "test"
        adapter.model = "fake-model"
        adapter._vision_enabled = False

        result = adapter.call_one_round(
            lambda active, latent: "system",
            "user",
            {},
            collection,
            new_message_checker=lambda: False,
        )

        self.assertFalse(result.new_message_during_thinking)
        self.assertTrue(result.had_tool_call)
        self.assertEqual(executed_args, [{"motivation": "fresh"}])
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
