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

STRUCTURED_DECISION_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点0分</current_time>
<qq>
<conversation type="group" id="42">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
</chat_logs>
</conversation>
</qq>
</world>
"""

STRUCTURED_SELF_ONLY_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<conversation type="group" id="42">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
  <message id="2" timestamp="刚刚">
    <sender id="self"/>
    <content type="text">我现在过去。</content>
  </message>
</chat_logs>
</conversation>
</qq>
</world>
"""

STRUCTURED_USER_AFTER_SELF_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<conversation type="group" id="42">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
  <message id="2" timestamp="刚刚">
    <sender id="self"/>
    <content type="text">我现在过去。</content>
  </message>
  <message id="3" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">不用来了，我已经出门了</content>
  </message>
</chat_logs>
</conversation>
</qq>
</world>
"""

STRUCTURED_SELF_NOTE_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<conversation type="group" id="42">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
  <note timestamp="刚刚">
    <operator id="10000" nickname="Bot"/>
    <content type="recall">Bot 撤回了一条消息</content>
  </note>
</chat_logs>
</conversation>
</qq>
</world>
"""

STRUCTURED_USER_NOTE_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<conversation type="group" id="42">
<self id="10000" name="Bot"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚">
    <sender id="10001" nickname="Alice"/>
    <content type="text">你现在能过来吗？</content>
  </message>
  <note timestamp="刚刚">
    <operator id="10001" nickname="Alice"/>
    <content type="recall">Alice 撤回了一条消息</content>
  </note>
</chat_logs>
</conversation>
</qq>
</world>
"""

PRIVATE_DECISION_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点0分</current_time>
<qq>
<conversation type="private">
<self id="10000" name="Bot"/>
<other id="10001" name="Alice"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚" from="other">
    <content type="text">你现在能过来吗？</content>
  </message>
</chat_logs>
</conversation>
</qq>
</world>
"""

PRIVATE_SELF_ONLY_WORLD = """
<world>
<current_time>现在是2026年的夏天，6月19日上午10点1分</current_time>
<qq>
<conversation type="private">
<self id="10000" name="Bot"/>
<other id="10001" name="Alice"/>
<chat_logs mode="current" has_previous="false">
  <message id="1" timestamp="刚刚" from="other">
    <content type="text">你现在能过来吗？</content>
  </message>
  <message id="2" timestamp="刚刚" from="self">
    <content type="text">我现在过去。</content>
  </message>
</chat_logs>
</conversation>
</qq>
</world>
"""


def test_world_signature_ignores_current_time_only_changes():
    later_time_world = DECISION_WORLD.replace("10点0分", "10点1分")

    assert world_semantically_changed(DECISION_WORLD, later_time_world) is False


def test_world_signature_ignores_self_messages_from_group_chat():
    assert world_semantically_changed(
        STRUCTURED_DECISION_WORLD,
        STRUCTURED_SELF_ONLY_WORLD,
    ) is False


def test_world_signature_ignores_self_messages_from_private_chat():
    assert world_semantically_changed(
        PRIVATE_DECISION_WORLD,
        PRIVATE_SELF_ONLY_WORLD,
    ) is False


def test_world_signature_still_detects_user_messages_after_self_message():
    assert world_semantically_changed(
        STRUCTURED_DECISION_WORLD,
        STRUCTURED_USER_AFTER_SELF_WORLD,
    ) is True


def test_world_signature_ignores_self_operated_notes():
    assert world_semantically_changed(
        STRUCTURED_DECISION_WORLD,
        STRUCTURED_SELF_NOTE_WORLD,
    ) is False


def test_world_signature_still_detects_user_operated_notes():
    assert world_semantically_changed(
        STRUCTURED_DECISION_WORLD,
        STRUCTURED_USER_NOTE_WORLD,
    ) is True


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


def test_external_effect_guard_ignores_prior_self_message_in_same_round():
    executed: list[str] = []
    world_state = {"sent": 0}
    guard = FakeGuardAdapter('{"execute": false, "reason": "should not be called"}')

    def execute(**_kwargs):
        world_state["sent"] += 1
        executed.append("send_message")
        return {"ok": True, "name": "send_message", "index": world_state["sent"]}

    collection = ToolCollection(
        active_specs={
            "send_message": ToolSpec(
                name="send_message",
                declaration=_declaration("send_message"),
                handler=execute,
                module_name="tools.send_message",
                externally_perceptible=True,
            )
        }
    )

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
        decision_world=STRUCTURED_DECISION_WORLD,
        current_world_provider=lambda: (
            STRUCTURED_SELF_ONLY_WORLD if world_state["sent"] else STRUCTURED_DECISION_WORLD
        ),
        tool_execution_guard_adapter=guard,
        tool_execution_guard_cfg={"enabled": True},
    ).execute(
        [_tool_call("send_message"), _tool_call("send_message")],
        inner_state={"cognition": "我准备连续发送两条消息。"},
    )

    assert executed == ["send_message", "send_message"]
    assert guard.calls == []
    assert [item["result"]["index"] for item in outcome.tool_calls_log] == [1, 2]


def test_external_effect_guard_still_checks_new_user_message_in_same_round():
    executed: list[str] = []
    world_state = {"sent": 0}
    guard = FakeGuardAdapter('{"execute": false, "reason": "对方已经取消请求"}')

    def execute(**_kwargs):
        world_state["sent"] += 1
        executed.append("send_message")
        return {"ok": True, "name": "send_message", "index": world_state["sent"]}

    collection = ToolCollection(
        active_specs={
            "send_message": ToolSpec(
                name="send_message",
                declaration=_declaration("send_message"),
                handler=execute,
                module_name="tools.send_message",
                externally_perceptible=True,
            )
        }
    )

    outcome = ToolExecutor(
        provider_name="test",
        tool_collection=collection,
        decision_world=STRUCTURED_DECISION_WORLD,
        current_world_provider=lambda: (
            STRUCTURED_USER_AFTER_SELF_WORLD if world_state["sent"] else STRUCTURED_DECISION_WORLD
        ),
        tool_execution_guard_adapter=guard,
        tool_execution_guard_cfg={"enabled": True},
    ).execute(
        [_tool_call("send_message"), _tool_call("send_message")],
        inner_state={"cognition": "我准备连续发送两条消息。"},
    )

    assert executed == ["send_message"]
    assert len(guard.calls) == 1
    assert "不用来了" in guard.calls[0]["user_content"]
    second_result = outcome.tool_calls_log[1]["result"]
    assert second_result["tool_not_executed"] is True
    assert second_result["blocked_by"] == "tool_execution_guard"

