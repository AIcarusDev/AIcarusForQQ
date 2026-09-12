from __future__ import annotations

from types import SimpleNamespace

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.core.round_runner import LLMRoundRunner
from llm.prompt.composer import merge_prompt_contents
from llm.prompt.sections import PromptPrelude, UserPromptSections


class _ToolCollection:
    def active_names(self):
        return ["runtime_manage"]

    def latent_names(self):
        return ["qq_social"]

    def namespace_prompt_blocks(self):
        return [{"name": "core", "active": True, "declarations": []}]

    def has_active_tools(self):
        return True


def test_merge_prompt_contents_preserves_multimodal_source_order():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    merged = merge_prompt_contents((
        "<memory/>",
        [
            {"type": "text", "text": "<world>"},
            image,
            {"type": "text", "text": "</world>"},
        ],
        "<container/>",
    ))

    assert isinstance(merged, list)
    assert merged[1] == image
    assert merged[0]["text"].endswith("<world>")
    assert merged[2]["text"].startswith("</world>\n<container/>")


def test_round_runner_composes_real_message_roles_and_source_order(monkeypatch):
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id="old")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="old")],
        cognition="covered cognition",
        motive="covered motive",
    )
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait"}, call_id="visible")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="visible")],
        cognition="visible cognition",
        motive="visible motive",
    )
    assert flow.queue_compression_summary("compressed history", coverage_end_seq=1)
    assert flow.promote_ready_compression_summary(max_rounds=1)

    runner = object.__new__(LLMRoundRunner)
    runner.provider = "test"
    runner.model = "test-model"
    runner._vision_enabled = True
    runner._assistant_prefill_supported = True
    runner._prompt_snapshot_cfg = {"enabled": False}
    runner._discarded_response_log_cfg = {"enabled": False}
    runner._last_main_stable_prompt_prefix = None
    monkeypatch.setattr(runner, "_normalize_generation_for_transport", lambda gen: dict(gen or {}))
    monkeypatch.setattr("llm.core.round_runner._record_usage_event", lambda **_kwargs: None)

    captured: list[dict] = []

    def fake_completion(**kwargs):
        captured.extend(kwargs["all_messages"])
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="current cognition</cognition><motive>wait</motive><action></action>",
                    reasoning_content="",
                ),
            )],
        )

    monkeypatch.setattr(runner, "_create_chat_completion", fake_completion)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    sections = UserPromptSections(
        memory="<memory><des>memory</des></memory>",
        skills="<skills><des>skills</des></skills>",
        world=[
            {"type": "text", "text": "<world><des>world</des>"},
            image,
            {"type": "text", "text": "</world>"},
        ],
        container="<container/>",
    )
    prelude = PromptPrelude(
        system_prompt="system",
        instruction="<instruction>custom</instruction>",
        guardian_card="<guardian_card>guardian</guardian_card>",
    )

    result = runner.call_one_round(
        lambda *_args, **_kwargs: prelude,
        sections.world,
        {"llm_contents_max_rounds": 1},
        _ToolCollection(),
        flow,
        assistant_prefill="<cognition>",
        prompt_sections=sections,
    )

    assert [message["role"] for message in captured] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
        "user",
        "assistant",
    ]
    front = captured[1]["content"]
    assert front.index("<instruction>") < front.index("<guardian_card>") < front.index("<tools>")
    assert captured[2]["content"].startswith("<summary>\n<des>")
    assert "visible cognition" in captured[3]["content"]

    tail = captured[-2]["content"]
    assert isinstance(tail, list)
    assert image in tail
    tail_text = "".join(part.get("text", "") for part in tail if part.get("type") == "text")
    assert tail_text.index("<memory>") < tail_text.index("<skills>") < tail_text.index("<world>")
    assert tail_text.index("</world>") < tail_text.index("<container/>") < tail_text.index("<output_schema>")
    assert tail_text.rstrip().endswith("</output_schema>")
    assert captured[-1] == {"role": "assistant", "content": "<cognition>"}
    assert "<instruction>" in runner._last_main_stable_prompt_prefix
    assert "<summary>" not in runner._last_main_stable_prompt_prefix
    assert "<world>" in result.world_xml
    assert "<memory>" not in result.world_xml
    assert "<output_schema>" not in result.world_xml
