from __future__ import annotations

import json
from types import SimpleNamespace

from llm.core.tool_calling.xml_protocol import build_tools_xml_message
from llm.core.tool_executor import ToolExecutor
from tools import build_tools
from tools.namespace_manage import execute as namespace_manage_execute
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
                module_name="tools.namespace_manage",
                namespace="core",
            ),
            "get_group_members": ToolSpec(
                name="get_group_members",
                declaration=_declaration("get_group_members", "获取当前群聊成员。"),
                handler=_handler,
                module_name="tools.get_group_members",
                namespace="qq_group_info",
            ),
        },
        latent_specs={
            "list_contact": ToolSpec(
                name="list_contact",
                declaration=_declaration("list_contact", "获取好友、群聊或临时会话列表。"),
                handler=_handler,
                module_name="tools.get_contact_list",
                namespace="qq_contacts",
            ),
        },
        all_specs={
            "namespace_manage": ToolSpec(
                name="namespace_manage",
                declaration=_declaration("namespace_manage"),
                handler=namespace_manage_execute,
                module_name="tools.namespace_manage",
                namespace="core",
            ),
            "get_group_members": ToolSpec(
                name="get_group_members",
                declaration=_declaration("get_group_members", "获取当前群聊成员。"),
                handler=_handler,
                module_name="tools.get_group_members",
                namespace="qq_group_info",
            ),
            "list_contact": ToolSpec(
                name="list_contact",
                declaration=_declaration("list_contact", "获取好友、群聊或临时会话列表。"),
                handler=_handler,
                module_name="tools.get_contact_list",
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
        module_name="tools.namespace_manage",
        namespace="core",
    )
    send_message = ToolSpec(
        name="send_message",
        declaration=_declaration("send_message"),
        handler=_handler,
        module_name="tools.send_message",
        namespace="qq_social",
    )
    list_stickers = ToolSpec(
        name="list_stickers",
        declaration=_declaration("list_stickers"),
        handler=_list_stickers,
        module_name="tools.stickers",
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
    xml = build_tools_xml_message(
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

    assert "<namespaces>" in xml
    assert '<namespace name="core" active="true">' in xml
    assert '"name":"wait"' in xml
    assert '<namespace name="qq_group_info" description="QQ群信息。" active="false"/>' in xml
    assert "<hidden>" not in xml
    assert "<activated>" not in xml


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
    assert open_result["opened"] == ["qq_contacts"]
    assert contact_result["tool_not_executed"] is True
    assert contact_result["namespace"] == "qq_contacts"
    assert "next round" in contact_result["error"]


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
    assert member_result["tool_not_executed"] is True
    assert member_result["namespace"] == "qq_group_info"
    assert "closed earlier" in member_result["error"]


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
    assert result["tool_not_executed"] is True
    assert result["opened"] == ["qq_contacts"]
    assert collection.namespace_state is not None
    assert "qq_contacts" in collection.namespace_state.open_order


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
    assert result["tool_not_executed"] is True
    assert result["namespace"] == "qq_social"
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
    assert "tools_manage" not in collection.all_specs
    inactive_namespaces = {item["name"] for item in collection.inactive_namespace_summaries()}
    assert "qq_group_info" in inactive_namespaces
    assert "qq_contacts" in inactive_namespaces
