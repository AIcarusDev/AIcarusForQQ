from __future__ import annotations

import json
from types import SimpleNamespace

from llm.core.tool_executor import ToolExecutor
from llm.core.tool_execution_guard import parse_guard_json, world_semantically_changed
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


def _tool_call(name: str, args: dict | None = None):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args or {}, ensure_ascii=False),
        ),
    )


def _collection(
    name: str,
    *,
    externally_perceptible: bool,
    executed: list[str],
) -> ToolCollection:
    def execute(**_kwargs):
        executed.append(name)
        return {"ok": True, "name": name}

    return ToolCollection(
        active_specs={
            name: ToolSpec(
                name=name,
                declaration=_declaration(name),
                handler=execute,
                module_name=f"tools.{name}",
                externally_perceptible=externally_perceptible,
            )
        }
    )


class FakeGuardAdapter:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    def call_simple_text(self, system_prompt, user_content, gen, log_tag):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_content": user_content,
            "gen": gen,
            "log_tag": log_tag,
        })
        return self.response


DECISION_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点0分</current_time>
<qq>
<chat_logs><message id="1" sender="Alice">你现在能过来吗？</message></chat_logs>
</qq>
</world>
"""

BLOCK_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<chat_logs>
<message id="1" sender="Alice">你现在能过来吗？</message>
<message id="2" sender="Alice">不用来了，我已经出门了</message>
</chat_logs>
</qq>
</world>
"""

ALLOW_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<chat_logs>
<message id="1" sender="Alice">你现在能过来吗？</message>
<message id="2" sender="Alice">门口见就行</message>
</chat_logs>
</qq>
</world>
"""


def test_world_signature_ignores_current_time_only_changes():
    later_time_world = DECISION_WORLD.replace("10点0分", "10点1分")

    assert world_semantically_changed(DECISION_WORLD, later_time_world) is False


def test_parse_guard_json_accepts_direct_boolean_and_execute_object():
    assert parse_guard_json("false") == (False, "")
    assert parse_guard_json('{"execute": true, "reason": "ok"}') == (True, "ok")


def test_external_effect_guard_blocks_changed_world_before_handler():
    executed: list[str] = []
    guard = FakeGuardAdapter('{"execute": false, "reason": "对方已经取消请求"}')
    collection = _collection(
        "send_message",
        externally_perceptible=True,
        executed=executed,
    )

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
        decision_world=DECISION_WORLD,
        current_world_provider=lambda: BLOCK_WORLD,
        tool_execution_guard_adapter=guard,
        tool_execution_guard_cfg={"enabled": True},
    ).execute(
        [_tool_call("send_message")],
        inner_state={"cognition": "我准备回复 Alice 说我现在过去。"},
    )

    assert executed == []
    assert len(guard.calls) == 1
    assert "<tool_call_json>" in guard.calls[0]["user_content"]
    assert "不用来了" in guard.calls[0]["user_content"]
    result = outcome.tool_calls_log[0]["result"]
    assert result["tool_not_executed"] is True
    assert result["blocked_by"] == "tool_execution_guard"
    assert result["guard_reason"] == "对方已经取消请求"


def test_external_effect_guard_allows_changed_world_before_handler():
    executed: list[str] = []
    guard = FakeGuardAdapter('{"execute": true, "reason": "新消息与动作兼容"}')
    collection = _collection(
        "send_message",
        externally_perceptible=True,
        executed=executed,
    )

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
        decision_world=DECISION_WORLD,
        current_world_provider=lambda: ALLOW_WORLD,
        tool_execution_guard_adapter=guard,
        tool_execution_guard_cfg={"enabled": True},
    ).execute(
        [_tool_call("send_message")],
        inner_state={"cognition": "我准备回复 Alice 说我现在过去。"},
    )

    assert executed == ["send_message"]
    assert len(guard.calls) == 1
    assert "门口见就行" in guard.calls[0]["user_content"]
    assert outcome.tool_calls_log[0]["result"] == {"ok": True, "name": "send_message"}

