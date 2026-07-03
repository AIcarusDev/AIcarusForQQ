from __future__ import annotations

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.core.round_runner import LLMRoundRunner


class _ToolCollection:
    def active_names(self):
        return []

    def latent_names(self):
        return []

    def namespace_prompt_blocks(self):
        return []

    def has_active_tools(self):
        return True


def test_round_runner_discards_repeated_cognition_before_action(monkeypatch):
    repeated = (
        "米塔说这个功能还有别的 bug 要修。我不需要追问，只要简单表示收到，"
        "然后等待她继续。这个判断已经在上一轮出现过，现在如果再次输出就是复读。"
    )
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(name="runtime_manage", args={"action": "wait", "seconds": 1}, call_id="call_1")],
        [ToolResponse(name="runtime_manage", response={"ok": True}, call_id="call_1")],
        cognition=repeated,
        raw_response=f"<cognition>{repeated}</cognition>",
    )

    runner = object.__new__(LLMRoundRunner)
    runner.provider = "local-qwen"
    runner.model = "Qwen3.6-27B"
    runner._vision_enabled = True
    runner._prompt_snapshot_cfg = {"enabled": False}
    runner._discarded_response_log_cfg = {"enabled": True}
    runner._last_main_stable_prompt_prefix = None
    monkeypatch.setattr(runner, "_normalize_generation_for_transport", lambda gen: dict(gen or {}))
    discard_log_call = {}

    def fake_save_cognition_prefill_discard(cfg, **kwargs):
        discard_log_call["cfg"] = cfg
        discard_log_call["kwargs"] = kwargs
        return "discard-log-1"

    def fake_create_chat_completion(*, all_messages, create_kwargs, on_text_delta=None, on_chunk=None):
        assert on_text_delta is not None
        on_text_delta(f"<cognition>{repeated}</cognition>")
        raise AssertionError("stream should abort before action text is generated")

    monkeypatch.setattr(runner, "_create_chat_completion", fake_create_chat_completion)
    monkeypatch.setattr(
        "llm.core.round_runner.save_cognition_prefill_discard",
        fake_save_cognition_prefill_discard,
    )

    result = runner.call_one_round(
        lambda activated_names=None, latent_names=None: "system",
        "<world>current</world>",
        {"duplicate_model_response_guard": {"prefill_guidance": {"enabled": True, "min_chars": 20}}},
        _ToolCollection(),
        flow,
    )

    assert result.cognition_prefill_retry is True
    assert result.cognition_prefill.startswith("<cognition>\n")
    assert result.cognition_prefill_retry_error["discard_log_id"] == "discard-log-1"
    assert result.had_tool_call is False
    assert flow.round_count == 1
    assert discard_log_call["cfg"] == {"enabled": True}
    saved = discard_log_call["kwargs"]
    assert saved["discarded_cognition"] == repeated
    assert saved["matched_cognition"] == repeated
    assert saved["matched_index"] == 0
    assert saved["retry_attempt"] == 1
    assert saved["visible_cognitions_count"] == 1
