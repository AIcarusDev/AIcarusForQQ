from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm.core.tool_calling.aic_action import build_aic_action_message
from llm.core.tool_executor import ToolExecutor
from llm.session import create_session, sessions
from platforms import PlatformRegistry
from platforms.core import CLOSED_PLATFORM_FOCUS, CORE_MAIN_FOCUS, CoreRuntime
from platforms.qq import QQRuntime
from platforms.qq.session_context import HOME_FOCUS, NO_CURRENT_SESSION_ERROR, resolve_current_qq_session
from tools import build_tools
from tools.core import namespace_manage as namespace_manage_mod
from tools.namespaces import NamespaceRuntimeState, load_module_registry, load_namespace_registry
from tools.specs import ToolCollection, ToolSpec


def _declaration(name: str, description: str | None = None) -> dict:
    return {
        "name": name,
        "description": description or f"{name} description",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def _handler(**_kwargs):
    return {"ok": True}


def _tool_call(name: str, args: str = "{}", namespace: str = ""):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, namespace=namespace, arguments=args),
    )


def _namespace_collection() -> ToolCollection:
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_group_info", registry, 1)
    collection = ToolCollection(
        namespace_specs={
            name: spec
            for name, spec in registry.namespaces.items()
            if name in {"core", "qq_group_info", "qq_contacts"}
        },
        namespace_registry=registry,
        namespace_state=state,
        active_namespace_order=["core", "qq_group_info"],
        round_index=1,
    )
    namespace_manage = ToolSpec(
        name="namespace_manage",
        declaration=_declaration("namespace_manage"),
        handler=namespace_manage_mod.make_handler(collection),
        module_name="tools.core.namespace_manage",
        namespace="core",
    )
    query_group_members = ToolSpec(
        name="query_group_members",
        declaration=_declaration("query_group_members", "查询当前群聊成员。"),
        handler=_handler,
        module_name="platforms.qq.tools.qq_group_info.query_group_members",
        namespace="qq_group_info",
    )
    list_contact = ToolSpec(
        name="list_contact",
        declaration=_declaration("list_contact", "获取好友、群聊或临时会话列表。"),
        handler=_handler,
        module_name="platforms.qq.tools.qq_contacts.list_contact",
        namespace="qq_contacts",
    )
    collection.active_specs.update({
        "core.namespace_manage": namespace_manage,
        "qq_group_info.query_group_members": query_group_members,
    })
    collection.latent_specs.update({"qq_contacts.list_contact": list_contact})
    collection.all_specs.update({
        "core.namespace_manage": namespace_manage,
        "qq_group_info.query_group_members": query_group_members,
        "qq_contacts.list_contact": list_contact,
    })
    return collection


def _attached_collection(executed: list[str]) -> ToolCollection:
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_social", registry, 1)

    def _list_stickers(**_kwargs):
        executed.append("list_stickers")
        return {"ok": True}

    collection = ToolCollection(
        namespace_specs={
            name: spec
            for name, spec in registry.namespaces.items()
            if name in {"core", "qq_social", "qq_stickers"}
        },
        namespace_registry=registry,
        namespace_state=state,
        active_namespace_order=["core", "qq_social"],
        round_index=1,
    )
    namespace_manage = ToolSpec(
        name="namespace_manage",
        declaration=_declaration("namespace_manage"),
        handler=namespace_manage_mod.make_handler(collection),
        module_name="tools.core.namespace_manage",
        namespace="core",
    )
    send_message = ToolSpec(
        name="send_message",
        declaration=_declaration("send_message"),
        handler=_handler,
        module_name="platforms.qq.tools.qq_social.send_message",
        namespace="qq_social",
    )
    list_stickers = ToolSpec(
        name="list_stickers",
        declaration=_declaration("list_stickers"),
        handler=_list_stickers,
        module_name="platforms.qq.tools.qq_stickers.list_stickers",
        namespace="qq_stickers",
        attached_to="qq_social",
    )
    collection.active_specs.update({
        "core.namespace_manage": namespace_manage,
        "qq_social.send_message": send_message,
        "qq_social.list_stickers": list_stickers,
    })
    collection.all_specs.update({
        "core.namespace_manage": namespace_manage,
        "qq_social.send_message": send_message,
        "qq_stickers.list_stickers": list_stickers,
    })
    return collection


def test_namespaces_render_active_schema_and_inactive_summary():
    action_message = build_aic_action_message(
        [],
        namespace_blocks=[
            {
                "name": "core",
                "active": True,
                "declarations": [_declaration("wait")],
            },
            {
                "name": "qq_group_info",
                "description": "QQ群信息。",
                "active": False,
            },
        ],
    )

    assert "<namespaces>" in action_message
    assert '<namespace name="core" active="true">' in action_message
    assert '"name":"wait"' in action_message
    assert '<namespace name="qq_group_info" description="QQ群信息。" active="false"/>' in action_message
    assert "<hidden>" not in action_message
    assert "<activated>" not in action_message


def test_build_tools_marks_namespace_manage_parallel_safe():
    collection = build_tools({})
    spec = collection.active_specs["core.namespace_manage"]
    assert spec.execution.parallel_safe is True
    assert spec.execution.parallel_key == "namespace_state"


def test_build_tools_marks_read_only_tools_parallel_safe(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"vision": True, "tts": {"enabled": False}},
        qq_client=FakeClient(),
        session=fake_session,
        vision_bridge=object(),
    )

    expected_parallel = {
        "browser_locator",
        "calculator",
        "examine_image",
        "get_avatar",
        "query_group_members",
        "get_group_notice",
        "get_qq_signature",
        "get_user_info",
        "get_weather",
        "list_contact",
        "list_stickers",
        "namespace_manage",
        "recall_memory",
        "recall_skill_resource",
        "search_history",
        "search_session",
        "think_deeply",
        "view_image_by_ref",
        "web_extract",
        "web_search",
    }
    for name in expected_parallel:
        spec = collection.get_any(name)
        assert spec is not None, name
        assert spec.execution.parallel_safe is True, name

    expected_serial = {
        "browser_control",
        "browse_forward",
        "delete_sticker",
        "enter_qq_session",
        "goal_manage",
        "plus_one",
        "poke",
        "recall_message",
        "restart",
        "return_to_qq_home",
        "runtime_manage",
        "save_sticker",
        "scroll_chat_log",
        "send_message",
        "set_group_card",
        "set_qq_signature",
        "update_sticker",
    }
    for name in expected_serial:
        spec = collection.get_any(name)
        assert spec is not None, name
        assert spec.execution.parallel_safe is False, name


def test_namespace_manage_open_is_next_round_only():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [
            _tool_call("namespace_manage", '{"open":["qq_contacts"]}'),
            _tool_call("list_contact"),
        ],
        inner_state={},
    )

    open_result = outcome.tool_calls_log[0]["result"]
    contact_result = outcome.tool_calls_log[1]["result"]
    assert open_result == {
        "ok": True,
        "tools": [{"namespace": "qq_contacts", "tools": ["list_contact"]}],
    }
    assert "already_open" not in open_result
    assert "closed" not in open_result
    assert contact_result["ok"] is False
    assert contact_result["namespace"] == "qq_contacts"
    assert "next round" in contact_result["error"]
    assert set(contact_result) == {"ok", "error", "namespace"}


def test_namespace_manage_close_blocks_later_tool_same_round():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [
            _tool_call("namespace_manage", '{"close":["qq_group_info"]}'),
            _tool_call("query_group_members"),
        ],
        inner_state={},
    )

    close_result = outcome.tool_calls_log[0]["result"]
    member_result = outcome.tool_calls_log[1]["result"]
    assert close_result["closed"] == ["qq_group_info"]
    assert "opened" not in close_result
    assert member_result["ok"] is False
    assert member_result["namespace"] == "qq_group_info"
    assert "closed earlier" in member_result["error"]
    assert set(member_result) == {"ok", "error", "namespace"}


def test_direct_inactive_tool_call_opens_namespace_for_next_round():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("list_contact")],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["ok"] is False
    assert result["namespace"] == "qq_contacts"
    assert set(result) == {"ok", "error", "namespace"}
    assert collection.namespace_state is not None
    assert "qq_contacts" in collection.namespace_state.open_order


def test_active_namespace_prefixed_tool_name_is_normalized():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("qq_group_info.query_group_members")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["namespace"] == "qq_group_info"
    assert tool_log["function"] == "query_group_members"
    assert tool_log["result"] == {"ok": True}
    assert "original_function" not in tool_log
    assert "repairs" not in tool_log


def test_prefixed_tool_name_is_normalized_without_registry_when_route_key_exists():
    spec = ToolSpec(
        name="send_message",
        declaration=_declaration("send_message"),
        handler=_handler,
        module_name="platforms.qq.tools.qq_social.send_message",
        namespace="qq_social",
    )
    collection = ToolCollection(
        active_specs={"qq_social.send_message": spec},
        all_specs={"qq_social.send_message": spec},
        active_namespace_order=["qq_social"],
    )

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("qq_social.send_message")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["namespace"] == "qq_social"
    assert tool_log["function"] == "send_message"
    assert tool_log["result"] == {"ok": True}


def test_active_namespace_explicit_namespace_tool_call_executes():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("query_group_members", namespace="qq_group_info")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["namespace"] == "qq_group_info"
    assert tool_log["function"] == "query_group_members"
    assert tool_log["result"] == {"ok": True}


def test_unique_bare_tool_name_is_repaired_to_namespace_path():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("query_group_members")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["namespace"] == "qq_group_info"
    assert tool_log["function"] == "query_group_members"
    assert tool_log["result"] == {"ok": True}


def test_ambiguous_bare_tool_name_requires_namespace():
    collection = _namespace_collection()
    core_send = ToolSpec(
        name="query_group_members",
        declaration=_declaration("query_group_members", "core duplicate"),
        handler=_handler,
        module_name="tools.core.query_group_members",
        namespace="core",
        visible_namespace="core",
    )
    collection.active_specs["core.query_group_members"] = core_send
    collection.all_specs["core.query_group_members"] = core_send

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("query_group_members")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["function"] == "query_group_members"
    assert tool_log["result"]["tool_not_executed"] is True
    assert "不明确" in tool_log["result"]["error"]
    assert set(tool_log["result"]["candidates"]) == {
        "qq_group_info.query_group_members",
        "core.query_group_members",
    }


def test_bare_name_does_not_repair_to_hidden_internal_tool():
    collection = _namespace_collection()
    hidden_spec = ToolSpec(
        name="hidden_probe",
        declaration=_declaration("hidden_probe"),
        handler=_handler,
        module_name="tools.hidden_probe",
        namespace="hidden_runtime",
        visibility="internal",
    )
    collection.all_specs["hidden_runtime.hidden_probe"] = hidden_spec

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("hidden_probe")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["namespace"] == ""
    assert tool_log["function"] == "hidden_probe"
    assert tool_log["result"]["error"] == "未知工具: hidden_probe"


def test_attached_tool_allows_host_namespace_prefix():
    executed: list[str] = []
    collection = _attached_collection(executed)

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("qq_social.list_stickers")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["namespace"] == "qq_social"
    assert tool_log["function"] == "list_stickers"
    assert tool_log["result"] == {"ok": True}
    assert "original_function" not in tool_log
    assert "repairs" not in tool_log
    assert executed == ["list_stickers"]


def test_inactive_namespace_prefixed_tool_opens_next_round():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("qq_contacts.list_contact")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    result = tool_log["result"]
    assert tool_log["namespace"] == "qq_contacts"
    assert tool_log["function"] == "list_contact"
    assert result["ok"] is False
    assert result["namespace"] == "qq_contacts"
    assert "inactive namespace" in result["error"]
    assert collection.namespace_state is not None
    assert "qq_contacts" in collection.namespace_state.open_order


def test_namespace_manage_open_reports_tools_attached_tools_and_skills(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"tts": {"enabled": True}, "vision": True},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=object(),
        provider=None,
    )
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("namespace_manage", '{"open":["qq_social"]}')],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["ok"] is True
    assert result["tools"] == [
        {
            "namespace": "qq_social",
            "tools": ["send_message", "send_voice", "recall_message", "poke", "plus_one"],
        }
    ]
    assert result["attached_tools"] == [
        {
            "host_namespace": "qq_social",
            "source_namespace": "qq_stickers",
            "tools": ["list_stickers"],
        },
        {
            "host_namespace": "qq_social",
            "source_namespace": "qq_chat_view",
            "tools": ["scroll_chat_log"],
        },
    ]
    assert result["skills"] == [
        {"namespace": "qq_social", "skill": "qq-social-style"}
    ]
    assert "active_namespaces" not in result
    assert "opened" not in result
    assert "already_open" not in result
    assert "closed" not in result


def test_attached_tool_executes_through_attached_namespace():
    executed: list[str] = []
    collection = _attached_collection(executed)

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("list_stickers")],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result == {"ok": True}
    assert executed == ["list_stickers"]


def test_close_blocks_later_attached_tool_same_round():
    executed: list[str] = []
    collection = _attached_collection(executed)

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [
            _tool_call("namespace_manage", '{"close":["qq_social"]}'),
            _tool_call("list_stickers"),
        ],
        inner_state={},
    )

    result = outcome.tool_calls_log[1]["result"]
    assert result["ok"] is False
    assert result["namespace"] == "qq_social"
    assert set(result) == {"ok", "error", "namespace"}
    assert executed == []


def test_namespace_preview_and_search_do_not_return_schema():
    collection = _namespace_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("namespace_manage", json.dumps({
            "preview": ["qq_contacts"],
            "search": "好友",
        }, ensure_ascii=False))],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["preview"][0]["tools"] == [
        {"name": "list_contact", "description": "获取好友、群聊或临时会话列表。"}
    ]
    assert "parameters" not in result["preview"][0]["tools"][0]
    assert result["search"] == [
        {
            "namespace": "qq_contacts",
            "name": "list_contact",
            "description": "获取好友、群聊或临时会话列表。",
        }
    ]


def test_internal_runtime_namespaces_are_not_model_operable(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("namespace_manage", json.dumps({
            "open": ["qq_runtime"],
            "preview": ["qq_runtime"],
            "search": "QQ 新消息",
        }, ensure_ascii=False))],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["not_found"] == ["qq_runtime"]
    assert result["warnings"] == [{"name": "qq_runtime", "warning": "未找到 namespace。"}]
    assert "preview" not in result
    assert "search" not in result
    inactive_namespaces = {item["name"] for item in collection.inactive_namespace_summaries()}
    assert "qq_runtime" not in inactive_namespaces


def test_runtime_manage_is_core_tool_and_qq_runtime_only_mounts_enter(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    active_names = collection.active_names()
    assert "core.runtime_manage" in active_names
    assert "wait_qq_event" not in active_names
    assert "wait_browser_event" not in active_names
    spec = collection.active_specs["core.enter_qq_session"]
    assert spec.namespace == "qq_runtime"
    assert spec.visible_namespace == "core"
    assert spec.mounted_to == "core"
    assert spec.mounted_by_module == "qq"
    assert "qq_runtime" not in collection.active_namespace_names()

    core_block = next(block for block in collection.namespace_prompt_blocks() if block["name"] == "core")
    core_names = [decl["name"] for decl in core_block["declarations"]]
    assert "runtime_manage" in core_names
    assert "wait_qq_event" not in core_names
    assert all(block["name"] != "qq_runtime" for block in collection.namespace_prompt_blocks())


def test_browser_runtime_no_longer_mounts_wait_when_browser_world_active(monkeypatch):
    import browser.session as browser_session

    monkeypatch.setattr(browser_session, "browser_world_view_state", lambda: {"active": True})
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_client=None,
        vision_bridge=None,
        provider=None,
    )

    active_names = collection.active_names()
    assert "core.runtime_manage" in active_names
    assert "wait_browser_event" not in active_names
    assert "browser_runtime" not in collection.active_namespace_names()

    core_block = next(block for block in collection.namespace_prompt_blocks() if block["name"] == "core")
    core_names = [decl["name"] for decl in core_block["declarations"]]
    assert "runtime_manage" in core_names
    assert "wait_browser_event" not in core_names
    assert all(block["name"] != "browser_runtime" for block in collection.namespace_prompt_blocks())


def test_build_tools_uses_namespace_registry(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    state = NamespaceRuntimeState()
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    assert "core.namespace_manage" in collection.active_names()
    assert "core.recall_skill_resource" in collection.active_names()
    assert "tools_manage" not in collection.all_specs
    inactive_namespaces = {item["name"] for item in collection.inactive_namespace_summaries()}
    assert "qq_group_info" in inactive_namespaces
    assert "qq_contacts" in inactive_namespaces


def test_qq_namespace_manifest_is_platform_owned():
    registry = load_namespace_registry()
    modules = load_module_registry()

    assert registry.get("qq_social").import_path == "platforms.qq.tools.qq_social"
    assert registry.get("qq_social").activation.platform == "qq"
    assert registry.get("qq_social").activation.surfaces == ("session",)
    assert registry.get("qq_runtime").visible is False
    assert registry.get("qq_runtime").activation.surfaces == ("home", "session")
    assert modules.modules["qq"].namespaces == (
        "qq_social",
        "qq_stickers",
        "qq_chat_view",
        "qq_profile",
        "qq_contacts",
        "qq_group_info",
        "qq_runtime",
    )
    assert modules.modules["qq"].mounts[0].source_namespace == "qq_runtime"

    assert "qq_social:" not in Path("src/tools/namespaces.yaml").read_text(encoding="utf-8")
    assert "\n  qq:\n" not in Path("src/tools/modules.yaml").read_text(encoding="utf-8")
    assert "qq_social:" in Path("src/platforms/qq/tools_manifest.yaml").read_text(encoding="utf-8")


def test_qq_surface_defaults_to_session_for_current_runtime(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_social", registry, 1)
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    assert "qq_social.send_message" in collection.active_names()
    assert "qq_social" in collection.active_namespace_names()


def test_qq_home_surface_hides_session_namespaces_but_keeps_runtime_mount(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_social", registry, 1)
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
        qq_surface="home",
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    assert "qq_social.send_message" not in collection.all_specs
    assert "qq_social" not in collection.active_namespace_names()
    inactive_namespaces = {item["name"] for item in collection.inactive_namespace_summaries()}
    assert "qq_social" not in inactive_namespaces

    spec = collection.active_specs["core.enter_qq_session"]
    assert spec.namespace == "qq_runtime"
    assert spec.visible_namespace == "core"
    assert spec.mounted_to == "core"
    assert collection.active_specs["core.return_to_qq_home"].mounted_to == "core"


def test_return_to_qq_home_makes_followup_session_tool_fail_naturally(monkeypatch):
    import app_state

    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    sessions.clear()
    current = create_session("qq:group:1234")
    current.set_conversation_meta("group", "1234", "Sandbox Group")
    sessions[current.key] = current
    monkeypatch.setattr(app_state, "current_focus", current.focus)

    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_chat_view", registry, 1)
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=state,
        current_round=1,
        default_ttl_rounds=5,
        qq_surface="session",
        qq_client=FakeClient(),
        qq_session_provider=resolve_current_qq_session,
        session=current,
        vision_bridge=None,
        provider=None,
    )

    outcome = ToolExecutor(provider_name="test", tool_collection=collection).execute(
        [
            _tool_call("return_to_qq_home"),
            _tool_call("search_history", '{"keywords":["天气"]}'),
        ],
        inner_state={},
    )

    results = {item["function"]: item["result"] for item in outcome.tool_calls_log}
    assert app_state.current_focus == HOME_FOCUS
    assert results["return_to_qq_home"]["ok"] is True
    assert results["search_history"]["error"] == NO_CURRENT_SESSION_ERROR


def test_namespace_manage_cannot_open_surface_hidden_namespace(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_surface="home",
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("namespace_manage", '{"open":["qq_social"]}')],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["not_found"] == ["qq_social"]
    assert "tools" not in result


def test_core_platform_page_tools_are_visible_and_switch_focus(monkeypatch):
    import app_state

    sessions.clear()
    registry = PlatformRegistry()
    registry.register(CoreRuntime({}))
    registry.register(QQRuntime({"enabled": True}))
    monkeypatch.setattr(app_state, "platform_registry", registry)
    monkeypatch.setattr(app_state, "current_focus", CLOSED_PLATFORM_FOCUS)

    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        session=create_session(CLOSED_PLATFORM_FOCUS),
        vision_bridge=None,
        provider=None,
    )

    names = collection.active_names()
    assert "core.list_platforms" in names
    assert "core.enter_platform" in names
    assert "core.close_platform" in names

    enter = collection.active_specs["core.enter_platform"].handler(name="core")
    assert enter["ok"] is True
    assert app_state.current_focus == CORE_MAIN_FOCUS

    listed = collection.active_specs["core.list_platforms"].handler()
    core_row = next(row for row in listed["platforms"] if row["name"] == "core")
    assert core_row["page_open"] is True
    assert core_row["main"]["key"] == CORE_MAIN_FOCUS.key()

    closed = collection.active_specs["core.close_platform"].handler()
    assert closed["ok"] is True
    assert closed["closed_platform"] == "core"
    assert app_state.current_focus == CLOSED_PLATFORM_FOCUS



