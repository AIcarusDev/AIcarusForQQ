from __future__ import annotations

import json
import re

import consciousness.flow as flow_module
from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.core.tool_calling.aic_action import AIC_ACTION_ERROR_NAME


def _result_payloads(content: str) -> list[dict]:
    return [
        json.loads(match)
        for match in re.findall(r"<result>(.*?)</result>", content)
    ]


def test_to_xml_messages_groups_tool_results_in_action_response():
    flow = ConsciousnessFlow()
    flow.append_round(
        [
            ToolCall(name="runtime_manage", args={"action": "wait", "seconds": 1}, call_id="call_1"),
            ToolCall(name="send_message", args={"segments": []}, call_id="call_2"),
        ],
        [
            ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_1"),
            ToolResponse(
                name="send_message",
                response={"sent_count": 1, "failed_count": 0},
                call_id="call_2",
            ),
        ],
        cognition="需要先等一下。",
    )

    messages = flow.to_xml_messages()

    assert [message["role"] for message in messages] == ["assistant", "user"]
    content = messages[1]["content"]
    assert isinstance(content, str)
    assert content.startswith("<action_response>\n")
    assert content.endswith("\n</action_response>")
    assert "<tool_response>" not in content
    assert "<tool_feedback>" not in content
    assert _result_payloads(content) == [
        {"id": "call_1", "name": "runtime_manage", "result": {"ok": True}},
        {
            "id": "call_2",
            "name": "send_message",
            "result": {"sent_count": 1, "failed_count": 0},
        },
    ]


def test_to_xml_messages_renders_aic_action_error_as_plain_feedback():
    flow = ConsciousnessFlow()
    flow.append_round(
        [],
        [
            ToolResponse(
                name=AIC_ACTION_ERROR_NAME,
                response={"error": "bad <tool_call>& details", "retryable": True},
                call_id="call_1",
            ),
        ],
    )

    messages = flow.to_xml_messages()

    assert [message["role"] for message in messages] == ["user"]
    assert messages[0]["content"] == (
        "<action_response>\n"
        "<feedback>aic_action_error: bad &lt;tool_call&gt;&amp; details</feedback>\n"
        "</action_response>"
    )


def test_to_xml_messages_preserves_tool_call_namespace():
    flow = ConsciousnessFlow()
    flow.append_round(
        [
            ToolCall(
                namespace="qq_social",
                name="send_message",
                args={"text": "hi"},
                call_id="call_1",
            )
        ],
        [
            ToolResponse(
                namespace="qq_social",
                name="send_message",
                response={"ok": True},
                call_id="call_1",
            )
        ],
    )

    messages = flow.to_xml_messages()

    assert [message["role"] for message in messages] == ["assistant", "user"]
    assert '<tool_call>{"id": "call_1", "namespace": "qq_social", "name": "send_message"' in messages[0]["content"]
    assert _result_payloads(messages[1]["content"]) == [
        {
            "id": "call_1",
            "namespace": "qq_social",
            "name": "send_message",
            "result": {"ok": True},
        }
    ]


def test_visible_cognitions_excludes_compressed_rounds():
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait", "seconds": 1}, call_id="call_1")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_1")],
        cognition="old cognition",
    )
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait", "seconds": 1}, call_id="call_2")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_2")],
        cognition="visible cognition",
    )
    assert flow.queue_compression_summary("summary", coverage_end_seq=1)
    assert flow.promote_ready_compression_summary(max_rounds=1)

    assert flow.visible_cognitions(limit=8) == ["visible cognition"]


def test_flow_round_memory_candidates_survive_dump_restore():
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait", "seconds": 1}, call_id="call_1")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_1")],
        cognition="华风身份信息需要记住。",
        memory_candidates=[
            {
                "memory_kind": "summary",
                "summary_id": "local:abc",
                "summary": "华风身份信息事件簇。",
                "source_event_ids": [11, 12],
            }
        ],
    )

    data, timestamps = flow.dump()
    restored = ConsciousnessFlow()
    restored.restore(data, timestamps)
    job = restored.build_compression_job(trigger_rounds=1)

    assert job is not None
    assert job.rounds[0].memory_candidates == [
        {
            "memory_kind": "summary",
            "summary_id": "local:abc",
            "summary": "华风身份信息事件簇。",
            "source_event_ids": [11, 12],
        }
    ]


def test_to_xml_messages_keeps_multimodal_parts_adjacent_to_result(monkeypatch):
    monkeypatch.setattr(
        flow_module,
        "make_data_url",
        lambda b64, mime: f"data:{mime};base64,{b64}",
    )
    flow = ConsciousnessFlow()
    flow.append_round(
        [
            ToolCall(name="view_image_by_ref", args={"image_ref": "abc"}, call_id="call_1"),
            ToolCall(name="runtime_manage", args={"action": "wait", "seconds": 1}, call_id="call_2"),
        ],
        [
            ToolResponse(
                name="view_image_by_ref",
                response={"ok": True, "image_ref": "abc"},
                call_id="call_1",
                multimodal_parts=[
                    {
                        "mime_type": "image/png",
                        "display_name": "abc.png",
                        "data": b"abc",
                    }
                ],
            ),
            ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_2"),
        ],
    )

    messages = flow.to_xml_messages()
    content = messages[1]["content"]

    assert isinstance(content, list)
    assert [part["type"] for part in content] == [
        "text",
        "text",
        "image_url",
        "text",
        "text",
    ]
    assert content[0]["text"] == "<action_response>\n"
    assert content[1]["text"] == (
        '<result>{"id": "call_1", "name": "view_image_by_ref", '
        '"result": {"ok": true, "image_ref": "abc"}}</result>'
    )
    assert content[2]["image_url"]["url"] == "data:image/png;base64,YWJj"
    assert content[3]["text"].startswith("\n<result>")
    assert content[4]["text"] == "\n</action_response>"
