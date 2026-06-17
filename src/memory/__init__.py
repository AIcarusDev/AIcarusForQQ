"""Top-level memory domain package (events-only)."""

from .render import build_memory_xml
from .repo.events import (
    ensure_schema,
    load_events_for_recall,
    merge_event_occurrence,
    prefetch_candidates_for_archiver,
    rebuild_embeddings,
    run_embedding_backfill,
    write_event,
    write_prompt_event,
)

__all__ = [
    "build_memory_xml",
    "ensure_schema",
    "load_events_for_recall",
    "merge_event_occurrence",
    "prefetch_candidates_for_archiver",
    "rebuild_embeddings",
    "run_embedding_backfill",
    "write_event",
    "write_prompt_event",
]
