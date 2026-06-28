from __future__ import annotations

import pytest

from llm.core.duplicate_response_guard import (
    COGNITION_PREFILL_POOL,
    CognitionPrefillRetrySignal,
    CognitionRepeatStreamGuard,
    choose_cognition_prefill,
    cognition_prefill_provider_supported,
    find_repeated_visible_cognition,
    is_passive_duplicate_tool_set,
    normalize_duplicate_model_response_guard_config,
)


def test_passive_duplicate_tool_set_allows_wait_and_sleep_only():
    assert is_passive_duplicate_tool_set(("wait",))
    assert is_passive_duplicate_tool_set(("wait", "sleep"))
    assert not is_passive_duplicate_tool_set(())
    assert not is_passive_duplicate_tool_set(("wait", "send_message"))


def test_duplicate_guard_config_includes_prefill_guidance_defaults():
    cfg = normalize_duplicate_model_response_guard_config({})

    assert cfg["enabled"] is False
    assert cfg["prefill_guidance"] == {
        "enabled": True,
        "lookback_rounds": 8,
        "similarity_threshold": 0.9,
        "min_chars": 80,
        "max_retries": 2,
    }


def test_cognition_repeat_detection_uses_visible_cognition_similarity():
    visible = [
        "我先看一下当前消息，然后决定是否回复。",
        "米塔说这个功能还有别的 bug 要修。我不需要追问，只要简单表示收到，然后等待她继续。",
    ]
    current = "米塔说这个功能还有别的 bug 要修。我不需要追问，只要简单表示收到，然后等待她继续。"

    repeated = find_repeated_visible_cognition(
        current,
        visible,
        similarity_threshold=0.9,
        min_chars=20,
    )

    assert repeated is not None
    assert repeated["matched_index"] == 1
    assert repeated["similarity"] == 1.0


def test_stream_guard_raises_when_cognition_closes_before_action():
    guard = CognitionRepeatStreamGuard(
        visible_cognitions=[
            "米塔说这个功能还有别的 bug 要修。我不需要追问，只要简单表示收到，然后等待她继续。"
        ],
        similarity_threshold=0.9,
        min_chars=20,
    )

    with pytest.raises(CognitionPrefillRetrySignal) as exc_info:
        guard.feed("<cognition>米塔说这个功能还有别的 bug 要修。")
        guard.feed("我不需要追问，只要简单表示收到，然后等待她继续。</cognition>")

    assert exc_info.value.similarity == 1.0
    assert exc_info.value.matched_index == 0


def test_choose_cognition_prefill_avoids_visible_and_used_exact_matches():
    visible = [COGNITION_PREFILL_POOL[0]]
    used = [COGNITION_PREFILL_POOL[1]]

    selected = choose_cognition_prefill(
        visible,
        used_prefills=used,
        seed_text="stable",
    )

    assert selected in COGNITION_PREFILL_POOL
    assert selected not in visible
    assert selected not in used


def test_cognition_prefill_marks_gemini_incompatible():
    assert not cognition_prefill_provider_supported("gemini", "gemini-2.5-flash")
    assert not cognition_prefill_provider_supported("custom", "gemini-2.5-flash")
    assert cognition_prefill_provider_supported("local-qwen", "Qwen3.6-27B")
