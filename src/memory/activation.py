"""Recall-strength based runtime activation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RecallActivationDecision:
    strength: float
    threshold: float | None
    activated: bool
    reason: str
    sample_count: int


@dataclass
class RecallActivationTracker:
    percentile: float = 0.80
    fallback_probability: float = 0.20
    history_limit: int = 128
    decay: float = 0.98
    random_fn: Callable[[], float] = random.random
    _samples: deque[tuple[float, float]] = field(default_factory=deque)

    def observe(self, events: list[dict[str, Any]] | None) -> float:
        strength = recall_strength(events)
        self.record(strength)
        return strength

    def evaluate(self, events: list[dict[str, Any]] | None) -> RecallActivationDecision:
        strength = recall_strength(events)
        threshold = self.threshold()
        sample_count = len(self._samples)
        threshold_hit = threshold is not None and strength > 0.0 and strength >= threshold
        fallback_hit = bool(strength > 0.0 and not threshold_hit and self.random_fn() < self.fallback_probability)

        if threshold_hit:
            activated = True
            reason = "p80"
        elif fallback_hit:
            activated = True
            reason = "fallback_probability"
        else:
            activated = False
            reason = "below_threshold" if threshold is not None else "cold_start"

        self.record(strength)
        return RecallActivationDecision(
            strength=round(strength, 6),
            threshold=round(threshold, 6) if threshold is not None else None,
            activated=activated,
            reason=reason,
            sample_count=sample_count,
        )

    def record(self, strength: float) -> None:
        strength = max(0.0, float(strength or 0.0))
        decayed = deque(
            (value, weight * self.decay)
            for value, weight in self._samples
            if weight * self.decay > 0.001
        )
        decayed.append((strength, 1.0))
        while len(decayed) > self.history_limit:
            decayed.popleft()
        self._samples = decayed

    def threshold(self) -> float | None:
        if not self._samples:
            return None
        return _weighted_percentile(list(self._samples), self.percentile)


_GLOBAL_RECALL_ACTIVATION = RecallActivationTracker()


def get_global_recall_activation_tracker() -> RecallActivationTracker:
    return _GLOBAL_RECALL_ACTIVATION


def recall_strength(events: list[dict[str, Any]] | None) -> float:
    total = 0.0
    for event in events or []:
        try:
            score = float(event.get("recall_score", 0.0))
        except (TypeError, ValueError, AttributeError):
            score = 0.0
        total += max(0.0, score)
    return total


def _weighted_percentile(samples: list[tuple[float, float]], percentile: float) -> float:
    ordered = sorted((max(0.0, value), max(0.0, weight)) for value, weight in samples)
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0.0:
        return 0.0

    target = max(0.0, min(1.0, float(percentile))) * total_weight
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= target:
            return value
    return ordered[-1][0]
