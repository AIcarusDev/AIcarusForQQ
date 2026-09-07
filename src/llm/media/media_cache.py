"""In-memory LRU cache for recent images (L1 Cache)."""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

DEFAULT_MEDIA_CACHE_CAPACITY = 128


class RecentMediaCache:
    """Thread-safe LRU cache for recently seen images across all sessions."""

    def __init__(self, capacity: int = DEFAULT_MEDIA_CACHE_CAPACITY) -> None:
        self.capacity = max(1, capacity)
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, tuple[dict[str, Any], str]] = OrderedDict()

    def put(self, image_ref: str, image: dict[str, Any], source: str = "cache") -> None:
        ref = str(image_ref or "").strip()
        if not ref or not isinstance(image, dict):
            return

        with self._lock:
            # Status snapshots go stale when download replaces the entry.
            if any(image.get(key) for key in ("pending", "failed", "expired", "unavailable_status")):
                self._cache.pop(ref, None)
                return
            if ref in self._cache:
                self._cache.move_to_end(ref)
            self._cache[ref] = (dict(image), str(source or "cache"))
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def get(self, image_ref: str) -> tuple[dict[str, Any], str] | None:
        ref = str(image_ref or "").strip()
        if not ref:
            return None

        with self._lock:
            if ref not in self._cache:
                return None
            self._cache.move_to_end(ref)
            image, source = self._cache[ref]
            return dict(image), source

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


_GLOBAL_CACHE = RecentMediaCache()


def cache_recent_image(image_ref: str, image: dict[str, Any], source: str = "cache") -> None:
    _GLOBAL_CACHE.put(image_ref, image, source=source)


def get_recent_image(image_ref: str) -> tuple[dict[str, Any], str] | None:
    return _GLOBAL_CACHE.get(image_ref)


def clear_recent_media_cache() -> None:
    _GLOBAL_CACHE.clear()


__all__ = [
    "DEFAULT_MEDIA_CACHE_CAPACITY",
    "RecentMediaCache",
    "cache_recent_image",
    "clear_recent_media_cache",
    "get_recent_image",
]
