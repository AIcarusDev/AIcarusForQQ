"""Compatibility exports for the Memory V2 event repository."""

from __future__ import annotations

from .events_v2 import (
    ensure_schema,
    load_events_for_recall,
    merge_event_occurrence,
    prefetch_candidates_for_archiver,
    soft_delete_event,
    write_event,
    write_prompt_event,
)

__all__ = [
    "ensure_schema",
    "load_events_for_recall",
    "merge_event_occurrence",
    "prefetch_candidates_for_archiver",
    "soft_delete_event",
    "write_event",
    "write_prompt_event",
]

