from __future__ import annotations

from types import SimpleNamespace

from llm.core.tool_executor import ToolExecutor
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState, load_namespace_registry
from tools.specs import ToolCollection, ToolSpec


def _declaration(name: str) -> dict:
    return {
        "name": name,
        "description": name,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }


def _tool_call(name: str):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments="{}"),
    )


def _collection(
    names: list[str],
    *,
    externally_perceptible: set[str] | None = None,
    tool_kinds: dict[str, str] | None = None,
    executed: list[str] | None = None,
) -> ToolCollection:
    external = externally_perceptible or set()
    kinds = tool_kinds or {}
    calls = executed if executed is not None else []

    def _handler(tool_name: str):
        def execute(**_kwargs):
            calls.append(tool_name)
            return {"ok": True, "name": tool_name}

        return execute

    return ToolCollection(
        active_specs={
            name: ToolSpec(
                name=name,
                declaration=_declaration(name),
                handler=_handler(name),
                module_name=f"tools.{name}",
                externally_perceptible=name in external,
                tool_kind=kinds.get(name, ""),
            )
            for name in names
        }
    )


def test_tools_execute_in_model_output_order_by_default():
    executed: list[str] = []
    collection = _collection(
        ["ordinary_tool", "plus_one"],
        externally_perceptible={"plus_one"},
        executed=executed,
    )

    ToolExecutor(provider_name="test", tool_collection=collection).execute(
        [_tool_call("ordinary_tool"), _tool_call("plus_one")],
        inner_state={},
    )

    assert executed == ["ordinary_tool", "plus_one"]


def test_focus_switch_before_external_tool_executes_in_model_output_order():
    executed: list[str] = []
    collection = _collection(
        ["enter_qq_session", "plus_one"],
        externally_perceptible={"plus_one"},
        tool_kinds={"enter_qq_session": "focus_switch"},
        executed=executed,
    )

    outcome = ToolExecutor(provider_name="test", tool_collection=collection).execute(
        [_tool_call("enter_qq_session"), _tool_call("plus_one")],
        inner_state={},
    )

    assert executed == ["enter_qq_session", "plus_one"]
    results = {item["function"]: item["result"] for item in outcome.tool_calls_log}
    assert results["enter_qq_session"] == {"ok": True, "name": "enter_qq_session"}
    assert results["plus_one"] == {"ok": True, "name": "plus_one"}


def test_external_tool_before_focus_switch_executes_in_model_output_order():
    executed: list[str] = []
    collection = _collection(
        ["enter_qq_session", "plus_one"],
        externally_perceptible={"plus_one"},
        tool_kinds={"enter_qq_session": "focus_switch"},
        executed=executed,
    )

    outcome = ToolExecutor(provider_name="test", tool_collection=collection).execute(
        [_tool_call("plus_one"), _tool_call("enter_qq_session")],
        inner_state={},
    )

    assert executed == ["plus_one", "enter_qq_session"]
    results = {item["function"]: item["result"] for item in outcome.tool_calls_log}
    assert results["plus_one"] == {"ok": True, "name": "plus_one"}
    assert results["enter_qq_session"] == {"ok": True, "name": "enter_qq_session"}


def test_tool_call_log_includes_call_id_and_elapsed_ms():
    executed: list[str] = []
    collection = _collection(["ordinary_tool"], executed=executed)

    outcome = ToolExecutor(provider_name="test", tool_collection=collection).execute(
        [_tool_call("ordinary_tool")],
        inner_state={},
    )

    entry = outcome.tool_calls_log[0]
    assert entry["function"] == "ordinary_tool"
    assert entry["call_id"] == "call_ordinary_tool"
    assert isinstance(entry["elapsed_ms"], (int, float))
    assert entry["elapsed_ms"] >= 0


def test_build_tools_carries_externally_perceptible_metadata(fake_session):
    class FakeClient:
        connected = True
        bot_id = "10000"
        _loop = None

    state = NamespaceRuntimeState()
    state.open("qq_social", load_namespace_registry(), 1)
    collection = build_tools(
        {"tts": {"enabled": False}, "vision": False},
        namespace_state=state,
        current_round=1,
        qq_client=FakeClient(),
        group_id=fake_session.conv_id,
        user_id=None,
        session=fake_session,
        vision_bridge=None,
        provider=None,
    )

    for name in ("send_message", "recall_message", "poke", "plus_one"):
        spec = collection.get_active(name)
        assert spec is not None
        assert spec.externally_perceptible is True

    enter_spec = collection.get_active("enter_qq_session", "core")
    assert enter_spec is not None
    assert enter_spec.externally_perceptible is False
    assert enter_spec.tool_kind == "focus_switch"
    assert enter_spec.namespace == "qq_runtime"
    assert enter_spec.mounted_to == "core"


