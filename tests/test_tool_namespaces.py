from __future__ import annotations

import json
from types import SimpleNamespace

from llm.core.tool_calling.aic_action import build_aic_action_message
from llm.core.tool_executor import ToolExecutor
from tools import build_tools
from tools.core.namespace_manage import execute as namespace_manage_execute
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
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


def _tool_call(name: str, args: str = "{}"):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=args),
    )


def _namespace_collection() -> ToolCollection:
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_group_info", registry, 1)
    return ToolCollection(
        active_specs={
            "namespace_manage": ToolSpec(
                name="namespace_manage",
                declaration=_declaration("namespace_manage"),
                handler=namespace_manage_execute,
                module_name="tools.core.namespace_manage",
                namespace="core",
            ),
            "get_group_members": ToolSpec(
                name="get_group_members",
                declaration=_declaration("get_group_members", "获取当前群聊成员。"),
                handler=_handler,
                module_name="tools.qq.qq_group_info.get_group_members",
                namespace="qq_group_info",
            ),
        },
        latent_specs={
            "list_contact": ToolSpec(
                name="list_contact",
                declaration=_declaration("list_contact", "获取好友、群聊或临时会话列表。"),
                handler=_handler,
                module_name="tools.qq.qq_contacts.list_contact",
                namespace="qq_contacts",
            ),
        },
        all_specs={
            "namespace_manage": ToolSpec(
                name="namespace_manage",
                declaration=_declaration("namespace_manage"),
                handler=namespace_manage_execute,
                module_name="tools.core.namespace_manage",
                namespace="core",
            ),
            "get_group_members": ToolSpec(
                name="get_group_members",
                declaration=_declaration("get_group_members", "获取当前群聊成员。"),
                handler=_handler,
                module_name="tools.qq.qq_group_info.get_group_members",
                namespace="qq_group_info",
            ),
            "list_contact": ToolSpec(
                name="list_contact",
                declaration=_declaration("list_contact", "获取好友、群聊或临时会话列表。"),
                handler=_handler,
                module_name="tools.qq.qq_contacts.list_contact",
                namespace="qq_contacts",
            ),
        },
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


def _attached_collection(executed: list[str]) -> ToolCollection:
    registry = load_namespace_registry()
    state = NamespaceRuntimeState()
    state.open("qq_social", registry, 1)

    def _list_stickers(**_kwargs):
        executed.append("list_stickers")
        return {"ok": True}

    namespace_manage = ToolSpec(
        name="namespace_manage",
        declaration=_declaration("namespace_manage"),
        handler=namespace_manage_execute,
        module_name="tools.core.namespace_manage",
        namespace="core",
    )
    send_message = ToolSpec(
        name="send_message",
        declaration=_declaration("send_message"),
        handler=_handler,
        module_name="tools.qq.qq_social.send_message",
        namespace="qq_social",
    )
    list_stickers = ToolSpec(
        name="list_stickers",
        declaration=_declaration("list_stickers"),
        handler=_list_stickers,
        module_name="tools.qq.qq_stickers.list_stickers",
        namespace="qq_stickers",
        attached_to="qq_social",
    )
    return ToolCollection(
        active_specs={
            "namespace_manage": namespace_manage,
            "send_message": send_message,
            "list_stickers": list_stickers,
        },
        latent_specs={},
        all_specs={
            "namespace_manage": namespace_manage,
            "send_message": send_message,
            "list_stickers": list_stickers,
        },
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
            _tool_call("get_group_members"),
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
        [_tool_call("qq_group_info.get_group_members")],
        inner_state={},
    )

    tool_log = outcome.tool_calls_log[0]
    assert tool_log["function"] == "get_group_members"
    assert tool_log["original_function"] == "qq_group_info.get_group_members"
    assert tool_log["result"] == {"ok": True}
    assert "namespace-qualified tool name" in tool_log["repairs"][0]


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
    assert tool_log["function"] == "list_stickers"
    assert tool_log["original_function"] == "qq_social.list_stickers"
    assert tool_log["result"] == {"ok": True}
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
    assert tool_log["function"] == "list_contact"
    assert tool_log["original_function"] == "qq_contacts.list_contact"
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
        qq_adapter_client=FakeClient(),
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
        }
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
        qq_adapter_client=FakeClient(),
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


def test_qq_runtime_wait_mounts_to_core_at_end(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_adapter_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    active_names = collection.active_names()
    assert active_names.index("wait_qq_event") > active_names.index("get_weather")
    spec = collection.active_specs["wait_qq_event"]
    assert spec.namespace == "qq_runtime"
    assert spec.visible_namespace == "core"
    assert spec.mounted_to == "core"
    assert spec.mounted_by_module == "qq"
    assert "qq_runtime" not in collection.active_namespace_names()

    core_block = next(block for block in collection.namespace_prompt_blocks() if block["name"] == "core")
    core_names = [decl["name"] for decl in core_block["declarations"]]
    assert core_names.index("wait_qq_event") > core_names.index("get_weather")
    assert all(block["name"] != "qq_runtime" for block in collection.namespace_prompt_blocks())


def test_browser_runtime_wait_mounts_to_core_when_browser_world_active(monkeypatch):
    import browser.session as browser_session

    monkeypatch.setattr(browser_session, "browser_world_view_state", lambda: {"active": True})
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=NamespaceRuntimeState(),
        current_round=1,
        default_ttl_rounds=5,
        qq_adapter_client=None,
        vision_bridge=None,
        provider=None,
    )

    active_names = collection.active_names()
    assert active_names.index("wait_browser_event") > active_names.index("get_weather")
    spec = collection.active_specs["wait_browser_event"]
    assert spec.namespace == "browser_runtime"
    assert spec.visible_namespace == "core"
    assert spec.mounted_to == "core"
    assert spec.mounted_by_module == "browser"
    assert "browser_runtime" not in collection.active_namespace_names()

    core_block = next(block for block in collection.namespace_prompt_blocks() if block["name"] == "core")
    core_names = [decl["name"] for decl in core_block["declarations"]]
    assert core_names.index("wait_browser_event") > core_names.index("get_weather")
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
        qq_adapter_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    assert "namespace_manage" in collection.active_names()
    assert "recall_skill_resource" in collection.active_names()
    assert "tools_manage" not in collection.all_specs
    inactive_namespaces = {item["name"] for item in collection.inactive_namespace_summaries()}
    assert "qq_group_info" in inactive_namespaces
    assert "qq_contacts" in inactive_namespaces
