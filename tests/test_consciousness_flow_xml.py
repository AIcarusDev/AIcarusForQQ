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
    assert messages[0]["content"].startswith("<old_cycles>\n")
    assert "<motive>有点记不清了</motive>" in messages[0]["content"]
    assert (
        "<feedback>aic_action_error: bad &lt;tool_call&gt;&amp; details</feedback>"
        in messages[0]["content"]
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
        cognition="需要发送消息。",
        motive="回应对方。",
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
        motive="避免忘记这条信息。",
        request_started_at=100.0,
        timestamp=105.0,
        memory_candidates=[
            {
                "memory_kind": "summary",
                "summary_id": "local:abc",
                "summary": "华风身份信息故事线。",
                "source_event_ids": [11, 12],
            }
        ],
    )

    data, timestamps = flow.dump()
    restored = ConsciousnessFlow()
    restored.restore(data, timestamps)
    job = restored.build_compression_job(trigger_rounds=1)

    assert job is not None
    assert job.rounds[0].motive == "避免忘记这条信息。"
    assert job.rounds[0].request_started_at == 100.0
    assert job.rounds[0].timestamp == 105.0
    assert "<cognition>华风身份信息需要记住。</cognition>" in job.task_xml
    assert "避免忘记这条信息。" not in job.task_xml
    assert job.rounds[0].memory_candidates == [
        {
            "memory_kind": "summary",
            "summary_id": "local:abc",
            "summary": "华风身份信息故事线。",
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
        cognition="需要查看图片。",
        motive="确认图片内容。",
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


def test_to_xml_messages_collapses_six_rounds_after_summary_and_keeps_two_cognitions_raw():
    flow = ConsciousnessFlow()
    for index in range(1, 10):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait", "seconds": index}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True, "index": index}, call_id=f"call_{index}")],
            cognition=f"cognition {index}",
            motive=f"motive {index}",
            request_started_at=800.0 + index,
            timestamp=810.0 + index,
        )
    assert flow.queue_compression_summary("summary through one", coverage_end_seq=1)
    assert flow.promote_ready_compression_summary(max_rounds=8)

    messages = flow.to_xml_messages(reference_time=1000.0)

    assert [message["role"] for message in messages] == [
        "user", "user", "assistant", "user", "assistant", "user",
    ]
    assert messages[0]["content"] == "<summary>\nsummary through one\n</summary>"
    old_cycles = messages[1]["content"]
    assert isinstance(old_cycles, str)
    assert old_cycles.count("<cycle ") == 6
    assert '<cycle start_ago="3m18s" end_ago="3m08s">' in old_cycles
    assert "<cognition>" not in old_cycles
    assert "<motive>motive 2</motive>" in old_cycles
    assert "<motive>motive 7</motive>" in old_cycles
    assert "motive 8" not in old_cycles
    assert messages[2]["content"].startswith("<cognition>cognition 8</cognition>\n<motive>motive 8</motive>")
    assert messages[4]["content"].startswith("<cognition>cognition 9</cognition>\n<motive>motive 9</motive>")
    assert flow.visible_cognitions(limit=8) == ["cognition 8", "cognition 9"]


def test_no_cognition_round_does_not_consume_raw_cognition_slots():
    flow = ConsciousnessFlow()
    for index, cognition in enumerate(("c1", "c2", "", "c3"), start=1):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait", "seconds": index}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True}, call_id=f"call_{index}")],
            cognition=cognition,
            motive=f"m{index}" if cognition else "",
            request_started_at=100.0 + index,
            timestamp=101.0 + index,
        )

    messages = flow.to_xml_messages(reference_time=200.0)

    assert messages[0]["role"] == "user"
    assert "<cognition>c1</cognition>" not in messages[0]["content"]
    assistant_contents = [message["content"] for message in messages if message["role"] == "assistant"]
    assert len(assistant_contents) == 3
    assert assistant_contents[0].startswith("<cognition>c2</cognition>")
    assert assistant_contents[1].startswith("<motive>有点记不清了</motive>")
    assert assistant_contents[2].startswith("<cognition>c3</cognition>")
    assert flow.visible_cognitions(limit=8) == ["c2", "c3"]


def test_old_cycles_preserves_multimodal_action_response(monkeypatch):
    monkeypatch.setattr(
        flow_module,
        "make_data_url",
        lambda b64, mime: f"data:{mime};base64,{b64}",
    )
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(name="view_image_by_ref", args={"image_ref": "abc"}, call_id="call_1")],
        [ToolResponse(
            name="view_image_by_ref",
            response={"ok": True, "image_ref": "abc"},
            call_id="call_1",
            multimodal_parts=[{"mime_type": "image/png", "display_name": "abc.png", "data": b"abc"}],
        )],
        cognition="old image cognition",
        motive="inspect image",
        request_started_at=10.0,
        timestamp=11.0,
    )
    for index in (2, 3):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True}, call_id=f"call_{index}")],
            cognition=f"recent {index}",
            motive=f"recent motive {index}",
            request_started_at=10.0 + index,
            timestamp=11.0 + index,
        )

    messages = flow.to_xml_messages(reference_time=20.0)
    old_content = messages[0]["content"]

    assert isinstance(old_content, list)
    assert [part["type"] for part in old_content] == ["text", "image_url", "text"]
    assert "<old_cycles>" in old_content[0]["text"]
    assert "<action_response>" in old_content[0]["text"]
    assert old_content[1]["image_url"]["url"] == "data:image/png;base64,YWJj"
    assert old_content[2]["text"].endswith("</old_cycles>")


def test_restart_marker_splits_old_cycles_without_reordering():
    flow = ConsciousnessFlow()
    for index in (1, 2):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True}, call_id=f"call_{index}")],
            cognition=f"c{index}",
            motive=f"m{index}",
        )
    flow.append_shutdown_marker()
    flow.complete_startup_marker()
    for index in (3, 4, 5, 6):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True}, call_id=f"call_{index}")],
            cognition=f"c{index}",
            motive=f"m{index}",
        )

    messages = flow.to_xml_messages()

    old_cycle_indexes = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "user"
        and isinstance(message["content"], str)
        and message["content"].startswith("<old_cycles>")
    ]
    restart_indexes = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "user"
        and isinstance(message["content"], str)
        and message["content"].startswith("[系统通知]")
    ]
    assert len(old_cycle_indexes) == 2
    assert len(restart_indexes) == 2
    assert old_cycle_indexes[0] < restart_indexes[0] < restart_indexes[1] < old_cycle_indexes[1]


def test_old_cycle_legacy_time_and_missing_motive_fallback_are_deterministic():
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id="call_1")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_1")],
        cognition="legacy cognition",
        motive="",
        request_started_at=None,
        timestamp=970.0,
    )
    for index in (2, 3):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True}, call_id=f"call_{index}")],
            cognition=f"recent {index}",
            motive=f"motive {index}",
            request_started_at=980.0 + index,
            timestamp=981.0 + index,
        )

    messages = flow.to_xml_messages(reference_time=1000.0)
    old_cycles = messages[0]["content"]

    assert '<cycle start_ago="30s" end_ago="30s">' in old_cycles
    assert "<motive>有点记不清了</motive>" in old_cycles
    assert flow_module._format_compact_duration(45) == "45s"
    assert flow_module._format_compact_duration(90) == "1m30s"
    assert flow_module._format_compact_duration(2 * 3600 + 5 * 60) == "2h05m"
    assert flow_module._format_compact_duration(3 * 86400 + 4 * 3600) == "3d04h"


def test_old_cycle_escapes_motive_without_changing_persisted_value():
    flow = ConsciousnessFlow()
    motives = ["because <now> & later", "recent two", "recent three"]
    for index, motive in enumerate(motives, start=1):
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id=f"call_{index}")],
            [ToolResponse(name="runtime_manage", response={"ok": True}, call_id=f"call_{index}")],
            cognition=f"c{index}",
            motive=motive,
        )

    old_cycles = flow.to_xml_messages()[0]["content"]

    assert "<motive>because &lt;now&gt; &amp; later</motive>" in old_cycles
    assert flow.recent_rounds(3)[0].motive == "because <now> & later"
