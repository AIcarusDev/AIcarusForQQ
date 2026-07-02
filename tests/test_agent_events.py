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
    projector.feed('{"name":"wait","arguments":{"seconds":1}}')
    projector.feed("</tool_call></action>")
    projector.finish()

    events = snapshot_events()
    text = "\n".join(str(event) for event in events)
    assert "<cognition>" not in text
    assert "<tool_call>" not in text
    assert any(event["type"] == "cognition_delta" and "先看清楚上下文" in event["text"] for event in events)
    assert any(
        event["type"] == "tool_planned"
        and event["tool_name"] == "wait"
        and event["call_id"] == "call_1"
        for event in events
    )


def test_agent_aic_action_stream_projector_assigns_parser_style_call_ids():
    clear_agent_events_for_test()
    projector = AgentActionStreamProjector(round_id="r-call-ids", provider="test", model="m")

    projector.feed("<action>")
    projector.feed('<tool_call>{"name":"wait","arguments":{"seconds":1}}</tool_call>')
    projector.feed('<tool_call>{"name":"sleep","arguments":{"duration":30}}</tool_call>')
    projector.feed("</action>")
    projector.finish()

    planned = [event for event in snapshot_events() if event["type"] == "tool_planned"]
    assert [(event["tool_name"], event["tool_index"], event["call_id"]) for event in planned] == [
        ("wait", 1, "call_1"),
        ("sleep", 2, "call_2"),
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
