"""Top-level memory domain package."""

from .recall import recall_memories
from .render import build_memory_xml
from .service import add_memory, remove_memory
from .runtime import (
    configure,
    get_active_count,
    get_all,
    get_deletable_ids,
    get_max_active,
    get_max_passive,
    get_passive_count,
    restore,
)

__all__ = [
    "add_memory",
    "build_memory_xml",
    "configure",
    "get_active_count",
    "get_all",
    "get_deletable_ids",
    "get_max_active",
    "get_max_passive",
    "get_passive_count",
    "recall_memories",
    "remove_memory",
    "restore",
]