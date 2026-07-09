import asyncio

import app_state
import platforms.qq.handler as qq_handler
from llm.session import ConversationSession, sessions
from memory.recall.activation import RecallActivationDecision, RecallActivationTracker, recall_strength


def _events(*scores: float) -> list[dict]:
    return [{"recall_score": score} for score in scores]


def test_recall_strength_sums_positive_scores_only():
    assert recall_strength(_events(0.6, -1.0, 0.4)) == 1.0


def test_recall_activation_uses_existing_p80_before_recording_current_sample():
    tracker = RecallActivationTracker(random_fn=lambda: 1.0)

    cold = tracker.evaluate(_events(0.9))

    assert cold.activated is False
    assert cold.reason == "cold_start"

    hot = tracker.evaluate(_events(1.2))

    assert hot.activated is True
    assert hot.reason == "p80"
    assert hot.threshold == 0.9


def test_recall_activation_keeps_probability_fallback_below_threshold():
    tracker = RecallActivationTracker(random_fn=lambda: 0.0)
    tracker.observe(_events(1.0))

    decision = tracker.evaluate(_events(0.1))

    assert decision.activated is True
    assert decision.reason == "fallback_probability"


def test_memory_activation_wake_only_runs_when_focus_is_idling(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        focus = ConversationSession()
        focus.set_conversation_meta("group", "focus", "Focus")
        incoming = ConversationSession()
        incoming.set_conversation_meta("group", "incoming", "Incoming")
        sessions["qq:group:focus"] = focus
        monkeypatch.setattr(app_state, "current_focus", focus.focus)

        calls = 0

        async def fake_prepare(*, evaluate_activation: bool = False):
            nonlocal calls
            calls += 1
            return RecallActivationDecision(
                strength=1.0,
                threshold=0.5,
                activated=True,
                reason="p80",
                sample_count=8,
            )

        monkeypatch.setattr(incoming, "prepare_memory_recall", fake_prepare)

        assert asyncio.run(qq_handler._maybe_memory_activation_wake(incoming)) is None
        assert calls == 0

        focus.sleep_arming = True
        focus.sleep_wake_action = "idle"
        decision = asyncio.run(qq_handler._maybe_memory_activation_wake(incoming))

        assert decision is not None
        assert decision.activated is True
        assert calls == 1
    finally:
        sessions.clear()
        sessions.update(original_sessions)


def test_memory_activation_wake_does_not_run_when_focus_is_sleeping(monkeypatch):
    original_sessions = dict(sessions)
    sessions.clear()
    try:
        focus = ConversationSession()
        focus.set_conversation_meta("group", "focus", "Focus")
        incoming = ConversationSession()
        incoming.set_conversation_meta("group", "incoming", "Incoming")
        sessions["qq:group:focus"] = focus
        monkeypatch.setattr(app_state, "current_focus", focus.focus)

        calls = 0

        async def fake_prepare(*, evaluate_activation: bool = False):
            nonlocal calls
            calls += 1
            return RecallActivationDecision(
                strength=1.0,
                threshold=0.5,
                activated=True,
                reason="p80",
                sample_count=8,
            )

        monkeypatch.setattr(incoming, "prepare_memory_recall", fake_prepare)

        focus.sleep_arming = True
        focus.sleep_wake_action = "sleep"

        assert asyncio.run(qq_handler._maybe_memory_activation_wake(incoming)) is None
        assert calls == 0
    finally:
        sessions.clear()
        sessions.update(original_sessions)
