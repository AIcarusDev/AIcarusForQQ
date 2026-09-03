from agent_events import (
    AgentActionStreamProjector,
    clear_agent_events_for_test,
    emit_agent_tool_hook,
    snapshot_events,
)


def test_agent_aic_action_stream_projector_hides_action_tags():
    clear_agent_events_for_test()
    projector = AgentActionStreamProjector(round_id="r1", provider="test", model="m")

    projector.feed("<cog")
    projector.feed("nition>先看清楚上下文。")
    projector.feed("</cognition><action><tool_call>")
    projector.feed('{"name":"runtime_manage","arguments":{"action":"wait","seconds":1}}')
    projector.feed("</tool_call></action>")
    projector.finish()

    events = snapshot_events()
    text = "\n".join(str(event) for event in events)
    assert "<cognition>" not in text
    assert "<tool_call>" not in text
    assert any(event["type"] == "cognition_delta" and "先看清楚上下文" in event["text"] for event in events)
    assert any(
        event["type"] == "tool_planned"
        and event["tool_name"] == "runtime_manage"
        and event["call_id"] == "call_1"
        for event in events
    )


def test_agent_stream_projects_native_reasoning_as_cognition_only_once():
    clear_agent_events_for_test()
    projector = AgentActionStreamProjector(
        round_id="r-native",
        provider="deepseek",
        model="deepseek-v4-flash",
        project_cognition=False,
    )

    projector.feed_cognition_delta("原生思考。")
    projector.feed(
        "<cognition>正文里的兼容认知。</cognition>"
        '<action><tool_call>{"name":"runtime_manage","arguments":{"action":"wait"}}</tool_call></action>'
    )
    projector.finish()

    events = snapshot_events()
    cognition_text = "".join(
        event.get("text", "")
        for event in events
        if event["type"] == "cognition_delta"
    )
    assert cognition_text == "原生思考。"
    assert [event["type"] for event in events].count("cognition_start") == 1
    assert [event["type"] for event in events].count("cognition_end") == 1
    assert any(event["type"] == "tool_planned" for event in events)


def test_agent_aic_action_stream_projector_assigns_parser_style_call_ids():
    clear_agent_events_for_test()
    projector = AgentActionStreamProjector(round_id="r-call-ids", provider="test", model="m")

    projector.feed("<action>")
    projector.feed('<tool_call>{"name":"runtime_manage","arguments":{"action":"wait","seconds":1}}</tool_call>')
    projector.feed('<tool_call>{"name":"runtime_manage","arguments":{"action":"sleep","minutes":30}}</tool_call>')
    projector.feed("</action>")
    projector.finish()

    planned = [event for event in snapshot_events() if event["type"] == "tool_planned"]
    assert [(event["tool_name"], event["tool_index"], event["call_id"]) for event in planned] == [
        ("runtime_manage", 1, "call_1"),
        ("runtime_manage", 2, "call_2"),
    ]


def test_agent_aic_action_stream_projector_preserves_namespace():
    clear_agent_events_for_test()
    projector = AgentActionStreamProjector(round_id="r-namespace", provider="test", model="m")

    projector.feed("<action>")
    projector.feed('<tool_call>{"namespace":"core","name":"runtime_manage","arguments":{"action":"wait"}}</tool_call>')
    projector.feed(
        '<tool_call>{"function":{"namespace":"qq_social","name":"send_message",'
        '"arguments":{"text":"hi"}}}</tool_call>'
    )
    projector.feed("</action>")
    projector.finish()

    planned = [event for event in snapshot_events() if event["type"] == "tool_planned"]
    assert [(event["namespace"], event["tool_name"], event["call_id"]) for event in planned] == [
        ("core", "core.runtime_manage", "call_1"),
        ("qq_social", "qq_social.send_message", "call_2"),
    ]


def test_agent_tool_hook_maps_finished_event():
    clear_agent_events_for_test()

    emit_agent_tool_hook(
        "finally_call",
        target="calculator",
        args={"expression": "1+1"},
        result={"ok": True, "value": 2},
        context={"round_id": "r2", "call_id": "call_1", "elapsed_ms": 12.5},
    )

    events = snapshot_events()
    assert events == [
        {
            "seq": events[0]["seq"],
            "type": "tool_finished",
            "created_at": events[0]["created_at"],
            "round_id": "r2",
            "call_id": "call_1",
            "tool_name": "calculator",
            "module": "",
            "elapsed_ms": 12.5,
            "result": {"ok": True, "value": 2},
            "result_preview": "ok=True, value=2",
            "ok": True,
        }
    ]
