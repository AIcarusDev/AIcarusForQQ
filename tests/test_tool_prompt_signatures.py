from __future__ import annotations

import asyncio

from runtime.events import RuntimeEventHub
from tools import build_tools
import tools as tools_package
from tools.contract import get_contract_from_module
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from tools.prompt_signatures import strip_schema_descriptions
from workspace import WorkspaceService


def _non_null_schema(schema: dict) -> dict:
    return next(option for option in schema.get("anyOf", ()) if option.get("type") != "null")


def test_strip_schema_descriptions_keeps_validation_keywords():
    declaration = {
        "name": "wait",
        "description": "核心等待工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15,
                    "description": "等待秒数。",
                    "x-coerce-integer": True,
                },
                "description": {
                    "type": "string",
                    "description": "业务参数名本身叫 description 时必须保留。",
                },
            },
            "required": ["seconds", "description"],
        },
    }

    stripped = strip_schema_descriptions(declaration)

    assert "核心等待工具" not in repr(stripped)
    assert "等待秒数" not in repr(stripped)
    assert "业务参数名本身叫 description" not in repr(stripped)
    assert "description" in stripped["parameters"]["properties"]
    assert stripped["parameters"]["properties"]["seconds"]["minimum"] == 1
    assert stripped["parameters"]["properties"]["seconds"]["maximum"] == 15
    assert stripped["parameters"]["properties"]["seconds"]["x-coerce-integer"] is True


def test_computer_contract_uses_namespace_local_names_without_model_timeout_controls():
    class Backend:
        async def request(self, method, params, *, timeout=None):
            raise AssertionError(method)

        async def close(self):
            return None

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("computer", registry, 1)
    loop = asyncio.new_event_loop()
    try:
        collection = build_tools(
            {"workspace": {"enabled": True}},
            namespace_state=state,
            current_round=1,
            workspace_service=WorkspaceService(Backend()),
            runtime_event_hub=RuntimeEventHub(),
            main_loop=loop,
        )
        for name in ("command", "read_file", "edit_file", "write_file", "find_files", "search"):
            spec = collection.all_specs[f"computer.{name}"]
            assert spec.name == name
            assert spec.declaration["name"] == name
        read_description = collection.all_specs["computer.read_file"].description
        assert "5000" in read_description
        assert "start_line 和 line_count" in read_description
        command_schema = repr(collection.all_specs["computer.command"].declaration["parameters"])
        assert "timeout_seconds" not in command_schema
        assert "background" not in command_schema
    finally:
        loop.close()


def test_send_voice_declaration_tracks_dynamic_tts_workers(monkeypatch):
    import app_state
    from platforms.qq.tools.qq_social import send_voice

    class TTS:
        def list_plugins(self):
            return [
                {
                    "plugin_id": "main",
                    "llm_schema": {
                        "description": "普通语音",
                        "properties": {
                            "voice": {"type": "string", "description": "音色名称。"},
                            "speed": {
                                "type": "number",
                                "minimum": 0.5,
                                "maximum": 2,
                                "description": "语速。",
                            },
                        },
                    },
                },
                {
                    "plugin_id": "song",
                    "llm_schema": {
                        "description": "歌声合成",
                        "properties": {
                            "style": {
                                "type": "string",
                                "enum": ["pop", "rock"],
                                "description": "演唱风格。",
                            },
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "description": "标签。",
                            },
                        },
                    },
                },
            ]

    monkeypatch.setattr(app_state, "tts_server", TTS(), raising=False)

    declaration = send_voice.get_declaration()
    properties = declaration["parameters"]["properties"]

    assert declaration["parameters"]["required"] == ["plugin_id"]
    assert properties["plugin_id"]["enum"] == ["main", "song"]
    assert properties["speed"]["minimum"] == 0.5
    assert properties["speed"]["maximum"] == 2
    assert properties["style"]["enum"] == ["pop", "rock"]
    assert properties["tags"]["minItems"] == 1


def test_get_weather_uses_python_first_contract():
    from tools.core import get_weather

    contract = getattr(get_weather.execute, "__tool_contract__", None)

    assert contract is not None
    declaration = contract.declaration()
    assert declaration["name"] == "get_weather"
    assert declaration["parameters"]["properties"]["city"]["minLength"] == 1


def test_calculator_contract_keeps_rounding_range():
    from tools.core import calculator

    contract = get_contract_from_module(calculator)

    assert contract is not None
    properties = contract.declaration()["parameters"]["properties"]
    assert properties["expression"]["type"] == "string"
    round_to = _non_null_schema(properties["round_to"])
    assert round_to["minimum"] == 0
    assert round_to["maximum"] == 30


def test_web_extract_contract_requires_nonempty_url():
    from tools.core import web_extract

    contract = get_contract_from_module(web_extract)

    assert contract is not None
    url_schema = contract.declaration()["parameters"]["properties"]["url"]
    assert url_schema["type"] == "string"
    assert url_schema["minLength"] == 1


def test_recall_tool_contracts_require_nonempty_ids():
    from tools.core import recall_memory, recall_skill_resource

    for mod in (recall_memory, recall_skill_resource):
        contract = get_contract_from_module(mod)
        assert contract is not None
        declaration = contract.declaration()
        properties = declaration["parameters"]["properties"]
        assert properties
        assert all(schema.get("minLength") == 1 for schema in properties.values())


def test_web_search_contract_keeps_query_and_result_limits():
    from tools.core import web_search

    contract = get_contract_from_module(web_search)

    assert contract is not None
    declaration = contract.declaration()
    properties = declaration["parameters"]["properties"]
    assert declaration["parameters"]["required"] == ["queries"]
    assert properties["queries"]["minItems"] == 1
    assert properties["queries"]["maxItems"] == 4
    max_results = _non_null_schema(properties["max_results"])
    assert max_results["minimum"] == 1
    assert max_results["maximum"] == 10
    assert _non_null_schema(properties["allowed_domains"])["maxItems"] == 20
    assert _non_null_schema(properties["blocked_domains"])["maxItems"] == 20


def test_restart_contract_has_empty_args():
    from tools.core import restart

    contract = get_contract_from_module(restart)

    assert contract is not None
    parameters = contract.declaration()["parameters"]
    assert parameters["properties"] == {}


def test_runtime_manage_contract_keeps_action_ranges():
    from tools.core import runtime_manage

    contract = get_contract_from_module(runtime_manage)

    assert contract is not None
    parameters = contract.declaration()["parameters"]
    defs = parameters["$defs"]
    wait = _non_null_schema(defs["WaitActionArgs"]["properties"]["seconds"])
    idle = _non_null_schema(defs["IdleActionArgs"]["properties"]["minutes"])
    sleep = _non_null_schema(defs["SleepActionArgs"]["properties"]["minutes"])
    assert (wait["minimum"], wait["maximum"]) == (1, 180)
    assert (idle["minimum"], idle["maximum"]) == (1, 60)
    assert (sleep["minimum"], sleep["maximum"]) == (30, 600)
    assert set(parameters["discriminator"]["mapping"]) == {"wait", "idle", "sleep"}


def test_query_group_members_contract_is_action_specific():
    from platforms.qq.tools.qq_group_info import query_group_members

    contract = get_contract_from_module(query_group_members)

    assert contract is not None
    parameters = contract.declaration()["parameters"]
    defs = parameters["$defs"]
    page = defs["GroupMembersListMembersArgs"]["properties"]["page"]
    query = defs["GroupMembersSearchArgs"]["properties"]["query"]
    assert set(parameters["discriminator"]["mapping"]) == {
        "list_admins",
        "list_members",
        "search",
    }
    assert (page["minimum"], page["maximum"]) == (1, 200)
    assert (query["minLength"], query["maxLength"]) == (1, 32)


def test_goal_manage_contract_preserves_business_title_property():
    from tools.core import goal_manage

    declaration = goal_manage.TOOL_CONTRACT.declaration()
    goal_item = declaration["parameters"]["$defs"]["GoalItem"]

    assert "title" in goal_item["properties"]
    assert goal_item["required"] == ["title", "content", "reason"]


def test_browser_locator_contract_is_operation_specific():
    from tools.browser.browser_use import browser_locator

    declaration = browser_locator.TOOL_CONTRACT.declaration()
    parameters = declaration["parameters"]

    assert len(parameters["anyOf"]) == 10
    defs = parameters["$defs"]
    operations = {
        schema["properties"]["op"]["const"]: schema
        for schema in defs.values()
        if "op" in schema.get("properties", {})
    }
    assert {"fill", "read_attribute", "select_option", "eval"} <= operations.keys()
    assert "input_text" in operations["fill"]["required"]
    assert "attribute" in operations["read_attribute"]["required"]
    assert "select_options" in operations["select_option"]["properties"]
    assert "arg" in defs["BrowserLocatorEvalOptions"]["properties"]


def test_think_deeply_contract_keeps_intent_enum():
    from tools.core import think_deeply

    contract = get_contract_from_module(think_deeply)

    assert contract is not None
    properties = contract.declaration()["parameters"]["properties"]
    assert properties["content"]["minLength"] == 1
    assert _non_null_schema(properties["intent"])["enum"] == [
        "affirmation",
        "criticism",
        "solving",
        "inspiration",
        "simulate",
    ]


def test_enter_qq_session_contract_preserves_enum_and_integer_compatibility():
    from platforms.qq.tools.qq_runtime import enter_qq_session

    contract = get_contract_from_module(enter_qq_session)

    assert contract is not None
    declaration = contract.declaration()
    properties = declaration["parameters"]["properties"]
    assert properties["type"]["enum"] == ["private", "group"]
    assert properties["id"]["type"] == "string"
    assert properties["id"]["minLength"] == 1
    assert properties["id"]["x-coerce-integer"] is True
    assert contract.name == "enter_qq_session"


def test_poke_user_id_contract_is_string_with_integer_compatibility():
    from platforms.qq.tools.qq_social import poke

    contract = get_contract_from_module(poke)

    assert contract is not None
    declaration = contract.declaration()
    user_id_schema = declaration["parameters"]["properties"]["user_id"]
    assert user_id_schema["type"] == "string"
    assert user_id_schema["minLength"] == 1
    assert user_id_schema["x-coerce-integer"] is True


def test_runtime_manage_replaces_wait_family_in_discovered_tools():
    from tools.core import runtime_manage

    contract = get_contract_from_module(runtime_manage)
    assert contract is not None
    assert set(contract.declaration()["parameters"]["discriminator"]["mapping"]) == {
        "wait",
        "idle",
        "sleep",
    }
    assert "wait_qq_event" not in tools_package._discovered_tool_names()
    assert "wait_browser_event" not in tools_package._discovered_tool_names()


def test_scroll_chat_log_contract_is_action_union():
    from platforms.qq.tools.qq_chat_log_view.scroll_chat_log import scroll_chat_log

    contract = get_contract_from_module(scroll_chat_log)

    assert contract is not None
    declaration = contract.declaration()
    assert "oneOf" in declaration["parameters"]
    parameters = declaration["parameters"]
    assert set(parameters["discriminator"]["mapping"]) == {
        "up",
        "down",
        "jump",
        "down_to_latest",
    }
    jump = parameters["$defs"]["ScrollJumpArgs"]
    assert jump["properties"]["message_id"]["minLength"] == 1
    assert jump["required"] == ["action", "message_id"]


def test_core_search_chat_log_contract_stays_lightweight():
    from platforms.core.tools.core_chat import search_chat_log

    contract = get_contract_from_module(search_chat_log)

    assert contract is not None
    properties = contract.declaration()["parameters"]["properties"]
    assert set(properties) == {"query", "sender", "limit", "context_window"}
    assert properties["query"]["minLength"] == 1
    assert properties["sender"]["enum"] == ["any", "guardian", "self"]
    assert (properties["context_window"]["minimum"], properties["context_window"]["maximum"]) == (0, 5)


def test_all_visible_tool_schemas_strip_model_descriptions():
    class Session:
        conv_type = "group"
        conv_id = "demo"
        conv_name = "demo"
        context_messages = []

    class QQAdapter:
        connected = True
        _loop = None

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    for namespace in registry.order:
        spec = registry.get(namespace)
        if spec and spec.visible and spec.openable and not spec.permanent:
            state.open(namespace, registry, 0)

    collection = build_tools(
        {},
        namespace_state=state,
        current_round=0,
        session=Session(),
        qq_client=QQAdapter(),
    )

    assert not [
        spec.name
        for spec in collection.active_specs.values()
        if _has_schema_description_keyword(spec.declaration)
    ]


def _has_schema_description_keyword(value, *, in_properties: bool = False) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "description" and not in_properties:
                return True
            if _has_schema_description_keyword(child, in_properties=(key == "properties")):
                return True
    if isinstance(value, list):
        return any(_has_schema_description_keyword(item) for item in value)
    return False



