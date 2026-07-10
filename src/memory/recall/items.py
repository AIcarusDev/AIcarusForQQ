"""Typed recall item helpers used before external dict projection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class RecallItem:
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RecallItem":
        return cls(dict(value))

    @property
    def event_id(self) -> int:
        try:
            return int(self.data.get("event_id", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def item_key(self) -> str:
        return str(self.data.get("event_id") or self.data.get("summary_id") or "")

    @property
    def summary_id(self) -> str:
        return str(self.data.get("summary_id") or "").strip()

    @property
    def memory_kind(self) -> str:
        return str(self.data.get("memory_kind") or "event")

    @property
    def recall_score(self) -> float:
        return _float_value(self.data.get("recall_score"), 0.0)

    @property
    def occurred_at(self) -> int:
        return _int_value(self.data.get("occurred_at") or self.data.get("created_at"), 0)

    @property
    def source_event_ids(self) -> set[int]:
        ids: set[int] = set()
        for value in self.data.get("source_event_ids") or ():
            event_id = _int_value(value, 0)
            if event_id > 0:
                ids.add(event_id)
        return ids

    @property
    def recall_reasons(self) -> set[str]:
        return {str(item) for item in self.data.get("recall_reasons", []) or [] if str(item)}

    def with_updates(
        self,
        *,
        recall_score: float | None = None,
        recall_reasons: Iterable[str] | None = None,
        contributing_event_ids: Iterable[int] | None = None,
    ) -> "RecallItem":
        data = dict(self.data)
        if recall_score is not None:
            data["recall_score"] = round(float(recall_score), 6)
        if recall_reasons is not None:
            data["recall_reasons"] = sorted({str(item) for item in recall_reasons if str(item)})
        if contributing_event_ids is not None:
            data["contributing_event_ids"] = sorted({int(item) for item in contributing_event_ids if int(item) > 0})
        return RecallItem(data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = ["RecallItem"]
