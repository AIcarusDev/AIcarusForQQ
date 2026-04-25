"""Runtime state for the memory domain."""

import asyncio
import time

_active_memories: list[dict] = []
_passive_memories: list[dict] = []
_self_memories: list[dict] = []
_max_active: int = 8
_max_passive: int = 15
_max_self: int = 50
_last_recalled_ids: set[int] = set()
_mem_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    """Create the runtime lock lazily inside the current event loop."""
    global _mem_lock
    if _mem_lock is None:
        _mem_lock = asyncio.Lock()
    return _mem_lock


def _ms() -> int:
    return int(time.time() * 1000)


def configure(max_active: int = 8, max_passive: int = 15, max_self: int = 50) -> None:
    global _max_active, _max_passive, _max_self
    _max_active = max_active
    _max_passive = max_passive
    _max_self = max_self


def restore(rows: list[dict]) -> None:
    """Restore runtime caches from MemoryTriples rows."""
    global _active_memories, _passive_memories, _self_memories
    valid = [row for row in rows if "id" in row]
    raw_self = [row for row in valid if row.get("subject", "") == "Bot:self"]
    _self_memories = sorted(
        raw_self,
        key=lambda row: row.get("confidence", 0.5),
        reverse=True,
    )[:_max_self]
    non_self = [row for row in valid if row.get("subject", "") != "Bot:self"]
    _active_memories = [
        row for row in non_self if row.get("origin", "passive") == "active"
    ][-_max_active:]
    _passive_memories = [
        row for row in non_self if row.get("origin", "passive") == "passive"
    ][-_max_passive:]


def get_all() -> list[dict]:
    return list(_active_memories) + list(_passive_memories) + list(_self_memories)


def get_active_count() -> int:
    return len(_active_memories)


def get_max_active() -> int:
    return _max_active


def get_passive_count() -> int:
    return len(_passive_memories)


def get_max_passive() -> int:
    return _max_passive


def get_deletable_ids() -> list[str]:
    ids: set[str] = {
        str(memory["id"])
        for memory in _active_memories + _passive_memories + _self_memories
        if "id" in memory
    }
    ids.update(str(memory_id) for memory_id in _last_recalled_ids)
    return sorted(ids, key=lambda value: int(value))


def set_last_recalled_ids(memory_ids: set[int]) -> None:
    global _last_recalled_ids
    _last_recalled_ids = set(memory_ids)


def get_all_cached_entries() -> list[dict]:
    return list(_active_memories) + list(_passive_memories) + list(_self_memories)


def get_total_pool_size() -> int:
    return len(_active_memories) + len(_passive_memories)


def choose_target_pool(subject: str, origin: str) -> tuple[list[dict], int]:
    if subject == "Bot:self":
        return _self_memories, -1
    if origin == "active":
        return _active_memories, _max_active
    return _passive_memories, _max_passive


def remove_cached_memory(triple_id: int) -> None:
    global _active_memories, _passive_memories, _self_memories
    _active_memories = [m for m in _active_memories if m.get("id") != triple_id]
    _passive_memories = [m for m in _passive_memories if m.get("id") != triple_id]
    _self_memories = [m for m in _self_memories if m.get("id") != triple_id]