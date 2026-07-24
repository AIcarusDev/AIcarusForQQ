"""Core runtime event queue used by passive waits and long-running jobs."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any


class RuntimeEventHub:
    def __init__(self, *, capacity: int = 512) -> None:
        self.capacity = max(32, int(capacity))
        self._condition = asyncio.Condition()
        self._events: list[dict[str, Any]] = []

    async def publish(
        self,
        event: Mapping[str, Any],
        *,
        target: str = "",
    ) -> str:
        payload = dict(event)
        event_id = str(payload.get("event_id") or uuid.uuid4().hex)
        payload["event_id"] = event_id
        payload.setdefault("occurred_at", time.time())
        payload["target"] = str(target or payload.get("target") or "")
        payload["_consumed"] = False
        async with self._condition:
            self._events.append(payload)
            if len(self._events) > self.capacity:
                consumed = [item for item in self._events if item.get("_consumed")]
                live = [item for item in self._events if not item.get("_consumed")]
                self._events = (consumed + live)[-self.capacity:]
            self._condition.notify_all()
        return event_id

    def publish_threadsafe(
        self,
        loop: asyncio.AbstractEventLoop,
        event: Mapping[str, Any],
        *,
        target: str = "",
    ) -> None:
        if loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.publish(event, target=target), loop)

    async def wait(
        self,
        *,
        timeout: float,
        target: str = "",
        event_types: Iterable[str] | None = None,
        consume: bool = True,
    ) -> list[dict[str, Any]]:
        allowed = {str(value) for value in event_types or [] if str(value)}

        def collect() -> list[dict[str, Any]]:
            matches: list[dict[str, Any]] = []
            for item in self._events:
                if item.get("_consumed"):
                    continue
                item_target = str(item.get("target") or "")
                if item_target and item_target != target:
                    continue
                if allowed and str(item.get("type") or "") not in allowed:
                    continue
                if consume:
                    item["_consumed"] = True
                matches.append(self._public(item))
            return matches

        async with self._condition:
            ready = collect()
            if ready or timeout <= 0:
                return ready
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(
                        lambda: any(
                            not item.get("_consumed")
                            and (not item.get("target") or str(item.get("target")) == target)
                            and (not allowed or str(item.get("type") or "") in allowed)
                            for item in self._events
                        )
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return []
            return collect()

    async def acknowledge(self, *, event_type: str, key: str, value: str) -> int:
        count = 0
        async with self._condition:
            for item in self._events:
                if item.get("_consumed"):
                    continue
                if str(item.get("type") or "") != event_type:
                    continue
                if str(item.get(key) or "") != value:
                    continue
                item["_consumed"] = True
                count += 1
        return count

    @staticmethod
    def _public(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in item.items()
            if key not in {"_consumed", "target", "occurred_at", "event_id"}
        }


__all__ = ["RuntimeEventHub"]
