from __future__ import annotations

from types import SimpleNamespace

from consciousness.flow import ConsciousnessFlow, ToolCall, ToolResponse
from llm.core.round_runner import LLMRoundRunner
from llm.core.tool_executor import ToolExecutionOutcome


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
        lambda activated_names=None, latent_names=None, **_kwargs: "system",
        "<world>current</world>",
        {"duplicate_model_response_guard": {"prefill_guidance": {"enabled": True, "min_chars": 20}}},
        _ToolCollection(),
        flow,
    )

    assert result.cognition_prefill_retry is True
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


def test_round_runner_persists_motive_and_cycle_boundaries(monkeypatch):
    flow = ConsciousnessFlow()
    runner = object.__new__(LLMRoundRunner)
    runner.provider = "test"
    runner.model = "test-model"
    runner._vision_enabled = True
    runner._prompt_snapshot_cfg = {"enabled": False}
    runner._discarded_response_log_cfg = {"enabled": False}
    runner._last_main_stable_prompt_prefix = None
    monkeypatch.setattr(runner, "_normalize_generation_for_transport", lambda gen: dict(gen or {}))
    monkeypatch.setattr("llm.core.round_runner._record_usage_event", lambda **_kwargs: None)

    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=(
                "<cognition>需要等待。</cognition>"
                "<motive>给外部事件一点时间。</motive>"
                '<action><tool_call>{"namespace":"core","name":"runtime_manage",'
                '"arguments":{"action":"wait","seconds":1}}</tool_call></action>'
            )),
        )],
    )
    monkeypatch.setattr(
        runner,
        "_create_chat_completion",
        lambda **_kwargs: response,
    )

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def execute(self, _tool_calls, *, inner_state):
            assert inner_state == {"cognition": "需要等待。", "think": "需要等待。"}
            return ToolExecutionOutcome(
                tool_calls_log=[{
                    "namespace": "core",
                    "function": "runtime_manage",
                    "arguments": {"action": "wait", "seconds": 1},
                    "result": {"ok": True},
                }],
                round_calls=[ToolCall(
                    namespace="core",
                    name="runtime_manage",
                    args={"action": "wait", "seconds": 1},
                    call_id="call_1",
                )],
                round_responses=[ToolResponse(
                    namespace="core",
                    name="runtime_manage",
                    response={"ok": True},
                    call_id="call_1",
                )],
            )

    monkeypatch.setattr("llm.core.round_runner.ToolExecutor", FakeExecutor)

    result = runner.call_one_round(
        lambda activated_names=None, latent_names=None, **_kwargs: "system",
        "<world>current</world>",
        {},
        _ToolCollection(),
        flow,
    )

    assert result.motive == "给外部事件一点时间。"
    assert result.request_started_at is not None
    assert result.action_finished_at is not None
    assert result.request_started_at <= result.action_finished_at
    saved_round = flow.recent_rounds(1)[0]
    assert saved_round.motive == result.motive
    assert saved_round.request_started_at == result.request_started_at
    assert saved_round.timestamp == result.action_finished_at


def test_round_runner_routes_native_reasoning_through_cognition_consumers(monkeypatch):
    flow = ConsciousnessFlow()
    flow.append_round(
        [ToolCall(
            name="runtime_manage",
            args={"action": "wait", "seconds": 1},
            call_id="call_previous",
        )],
        [ToolResponse(
            name="runtime_manage",
            response={"ok": True},
            call_id="call_previous",
        )],
        cognition="上一轮内部思考。",
        motive="继续观察。",
    )
    runner = object.__new__(LLMRoundRunner)
    runner.provider = "deepseek"
    runner.model = "deepseek-v4-flash"
    runner._vision_enabled = True
    runner._prompt_snapshot_cfg = {"enabled": False}
    runner._discarded_response_log_cfg = {"enabled": False}
    runner._last_main_stable_prompt_prefix = None
    normalized_generation = {}

    def normalize_generation(gen):
        normalized_generation.update(gen)
        return dict(gen)

    monkeypatch.setattr(runner, "_normalize_generation_for_transport", normalize_generation)
    monkeypatch.setattr("llm.core.round_runner._record_usage_event", lambda **_kwargs: None)

    response = SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(
                reasoning_content="API 返回的原生思考。",
                content=(
                    "<cognition>旧 prompt 仍要求输出的认知。</cognition>"
                    "<motive>等待外部事件。</motive>"
                    '<action><tool_call>{"namespace":"core","name":"runtime_manage",'
                    '"arguments":{"action":"wait","seconds":1}}</tool_call></action>'
                ),
            ),
        )],
    )
    request_messages = []

    def fake_completion(**kwargs):
        request_messages.extend(kwargs["all_messages"])
        return response

    monkeypatch.setattr(runner, "_create_chat_completion", fake_completion)

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

        def execute(self, _tool_calls, *, inner_state):
            assert inner_state == {
                "cognition": "API 返回的原生思考。",
                "think": "API 返回的原生思考。",
            }
            return ToolExecutionOutcome(
                round_calls=[ToolCall(
                    namespace="core",
                    name="runtime_manage",
                    args={"action": "wait", "seconds": 1},
                    call_id="call_1",
                )],
                round_responses=[ToolResponse(
                    namespace="core",
                    name="runtime_manage",
                    response={"ok": True},
                    call_id="call_1",
                )],
            )

    monkeypatch.setattr("llm.core.round_runner.ToolExecutor", FakeExecutor)

    prompt_route = {}

    def build_system_prompt(
        activated_names=None,
        latent_names=None,
        *,
        native_reasoning_as_cognition=False,
    ):
        prompt_route["native_reasoning_as_cognition"] = (
            native_reasoning_as_cognition
        )
        return "system"

    result = runner.call_one_round(
        build_system_prompt,
        "<world>current</world>",
        {
            "enable_thinking": False,
            "native_reasoning_as_cognition": True,
        },
        _ToolCollection(),
        flow,
    )

    assert normalized_generation["enable_thinking"] is True
    assert prompt_route["native_reasoning_as_cognition"] is True
    previous_assistant = next(
        message
        for message in request_messages
        if message["role"] == "assistant"
    )
    assert "<cognition>" not in previous_assistant["content"]
    assert previous_assistant["content"].startswith(
        "<thinking>上一轮内部思考。</thinking>"
    )
    assert result.cognition == "API 返回的原生思考。"
    assert flow.recent_rounds(1)[0].cognition == "API 返回的原生思考。"

    legacy_history = flow.to_xml_messages(native_reasoning_as_cognition=False)
    assert any(
        message["role"] == "assistant"
        and "<cognition>API 返回的原生思考。</cognition>" in message["content"]
        for message in legacy_history
    )
