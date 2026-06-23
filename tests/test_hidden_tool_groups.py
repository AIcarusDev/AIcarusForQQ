from __future__ import annotations

from types import SimpleNamespace

from llm.core.tool_calling.xml_protocol import build_tools_xml_message
from llm.core.tool_executor import ToolExecutor
from tools import build_tools
from tools.specs import ToolCollection, ToolGroupSpec, ToolSpec
from tools.tools_manage import execute as tools_manage_execute


def _declaration(name: str) -> dict:
    return {
        "name": name,
        "description": f"{name} description",
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


def _group_collection() -> ToolCollection:
    group = ToolGroupSpec(
        name="group_info",
        description="群信息能力：查询群成员和群公告。",
        keywords=("群", "成员", "公告"),
    )
    return ToolCollection(
        active_specs={
            "tools_manage": ToolSpec(
                name="tools_manage",
                declaration=_declaration("tools_manage"),
                handler=tools_manage_execute,
                module_name="tools.tools_manage",
            )
        },
        latent_specs={
            "get_group_members": ToolSpec(
                name="get_group_members",
                declaration=_declaration("get_group_members"),
                handler=_handler,
                module_name="tools.get_group_members",
                always_available=False,
                group="group_info",
            ),
            "get_group_notice_list": ToolSpec(
                name="get_group_notice_list",
                declaration=_declaration("get_group_notice_list"),
                handler=_handler,
                module_name="tools.get_group_notice_list",
                always_available=False,
                group="group_info",
            ),
        },
        group_specs={"group_info": group},
    )


def test_hidden_groups_render_as_tool_sets():
    xml = build_tools_xml_message(
        [_declaration("wait")],
        hidden_groups=[
            {
                "name": "group_info",
                "description": "群信息能力：查询群成员和群公告。",
            }
        ],
    )

    assert '<tool_set name="group_info"' in xml
    assert "群信息能力" in xml
    assert "get_group_members" not in xml


def test_tool_collection_activates_whole_group_from_single_tool():
    collection = _group_collection()

    activated = collection.activate_related("get_group_members")

    assert [spec.name for spec in activated] == [
        "get_group_members",
        "get_group_notice_list",
    ]
    assert collection.latent_names() == []
    assert set(collection.active_names()) >= {
        "get_group_members",
        "get_group_notice_list",
    }
    assert collection.hidden_groups() == []


def test_tools_manage_get_group_activates_whole_group():
    collection = _group_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("tools_manage", '{"get":["group_info"]}')],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["activated_groups"] == ["group_info"]
    assert result["activated"] == [
        "get_group_members",
        "get_group_notice_list",
    ]
    assert collection.latent_names() == []


def test_direct_hidden_tool_call_defers_activation_for_whole_group():
    collection = _group_collection()
    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
    ).execute(
        [_tool_call("get_group_members")],
        inner_state={},
    )

    result = outcome.tool_calls_log[0]["result"]
    assert result["activation_deferred"] is True
    assert result["activated_groups"] == ["group_info"]
    assert result["activated"] == [
        "get_group_members",
        "get_group_notice_list",
    ]
    assert collection.latent_names() == [
        "get_group_members",
        "get_group_notice_list",
    ]


def test_build_tools_exposes_hidden_groups_instead_of_names(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        qq_adapter_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    hidden_groups = {group["name"] for group in collection.hidden_groups()}
    assert "group_info" in hidden_groups
    assert "contacts_profile" in hidden_groups
    assert "get_group_members" in collection.latent_names()
