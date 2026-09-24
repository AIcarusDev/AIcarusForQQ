"""One persistent FIFO worker. A failed head batch waits and retries in place."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm.core.daemon_thread import run_in_daemon_thread
from . import store
from .prompt import CONSOLIDATION_PROMPT

logger = logging.getLogger("AICQ.memory.active")
RELATED_LIMIT = 8


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    action: Literal["upsert", "retract"]
    id: str | None = None
    content: str = ""

    @model_validator(mode="after")
    def valid_change(self):
        if self.action == "upsert" and not self.content.strip():
            raise ValueError("upsert requires content")
        if self.action == "retract" and self.id is None:
            raise ValueError("retract requires an existing memory id")
        if self.id is not None:
            store.parse_id(self.id, "M")
        return self


class Consolidation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    changes: list[Change]
    note: str = Field(min_length=1)


def enabled() -> bool:
    import app_state
    return (getattr(app_state, "config", {}) or {}).get("memory", {}).get("active", {}).get("enabled", True) is not False


def _adapter():
    import app_state
    return getattr(app_state, "memory_processing_adapter", None)


def _generate(payload: dict) -> str:
    import app_state
    adapter = _adapter()
    if adapter is None:
        raise RuntimeError("memory_processing_model_unavailable")
    cfg = getattr(app_state, "memory_processing_cfg", {}) or {}
    gen = dict(cfg.get("generation") or {})
    gen.setdefault("temperature", 0.2)
    gen.setdefault("max_output_tokens", 4000)
    return adapter.call_simple_text(
        CONSOLIDATION_PROMPT, json.dumps(payload, ensure_ascii=False), gen,
        log_tag="memory/active_consolidation",
    )


class ActiveMemoryService:
    def __init__(self, path: str, generate=None):
        self.path = path
        self.generate = generate or _generate
        self.task: asyncio.Task | None = None
        self.wake = asyncio.Event()
        self.accepting = True

    def start(self) -> None:
        if self.task is not None and not self.task.done():
            return
        store.initialize(self.path)
        store.recover(self.path)
        self.accepting = True
        self.task = asyncio.create_task(self.run(), name="active-memory-fifo")

    async def stop(self) -> None:
        self.accepting = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    def submit(self, cognition: str, memories: list[str]) -> dict:
        if not self.accepting:
            return {"saved": False, "error": "active_memory_stopping"}
        if not enabled():
            return {"saved": False, "error": "active_memory_disabled"}
        self.start()
        result = store.enqueue(self.path, cognition, memories)
        self.wake.set()
        result["processor_ready"] = _adapter() is not None
        return result

    async def process_one(self) -> bool:
        batch = store.head(self.path)
        if batch is None or batch["retry_at"] > time.time():
            return False
        if not store.begin(self.path, batch["id"]):
            return False
        try:
            memories = json.loads(batch["memories_json"])
            # Search each proposal; cognition remains intact as context.
            related_by_id = {}
            for memory in memories:
                for item in await asyncio.to_thread(store.search, self.path, memory, limit=RELATED_LIMIT):
                    related_by_id.setdefault(item["id"], item)
            related = sorted(related_by_id.values(), key=lambda item: (-item["matches"], item["id"]))[:RELATED_LIMIT]
            payload = {
                "cognition": batch["cognition"],
                "memories": memories,
                "related": [{"id": item["id"], "content": item["content"]} for item in related],
            }
            raw = await run_in_daemon_thread(self.generate, payload, thread_name="active-memory-model")
            result = Consolidation.model_validate_json(raw)
            store.complete(self.path, batch["id"], [item.model_dump() for item in result.changes], result.note, related)
            logger.info("[active_memory] batch B%06d completed changes=%d", batch["id"], len(result.changes))
        except asyncio.CancelledError:
            # No model thread can write to the DB. Startup reclaims this running row.
            raise
        except Exception as exc:
            delay = min(300, 5 * 2 ** min(batch["attempts"], 6))
            code = "model_unavailable" if self.generate is _generate and _adapter() is None else type(exc).__name__
            store.retry(self.path, batch["id"], code, delay)
            logger.warning("[active_memory] batch B%06d retry in %ss (%s)", batch["id"], delay, code)
        return True

    async def run(self) -> None:
        repair_views = True
        recover_queue = False
        while True:
            self.wake.clear()
            publication_failed = False
            try:
                # A failed mirror write is retried without rerunning consolidation.
                publication = asyncio.create_task(asyncio.to_thread(store.publish, self.path, force=repair_views))
                try:
                    await asyncio.shield(publication)
                except asyncio.CancelledError:
                    # Finish filesystem/DB publication before maintenance can delete.
                    try:
                        await publication
                    finally:
                        raise asyncio.CancelledError
                repair_views = False
            except Exception:
                publication_failed = True
                logger.warning("[active_memory] Markdown publication pending", exc_info=True)
            try:
                if recover_queue:
                    store.recover(self.path)
                    recover_queue = False
                if enabled() and await self.process_one():
                    continue
                pending = store.head(self.path)
            except asyncio.CancelledError:
                raise
            except Exception:
                recover_queue = True
                pending = True
                logger.warning("[active_memory] queue unavailable; will retry", exc_info=True)
            try:
                if pending or publication_failed:
                    await asyncio.wait_for(self.wake.wait(), timeout=5)
                else:
                    await self.wake.wait()
            except asyncio.TimeoutError:
                pass


_service: ActiveMemoryService | None = None


def service() -> ActiveMemoryService:
    global _service
    path = store.default_path()
    if _service is None or _service.path != path:
        if _service is not None and _service.task is not None and not _service.task.done():
            raise RuntimeError("stop the active memory worker before changing databases")
        _service = ActiveMemoryService(path)
    return _service


async def start() -> None:
    service().start()


async def stop() -> None:
    await service().stop()


async def submit(cognition: str, memories: list[str]) -> dict:
    return service().submit(cognition, memories)


async def recall(query: str, limit: int) -> list[dict]:
    if not enabled() or not query.strip():
        return []
    items = await asyncio.to_thread(store.search, store.default_path(), query, limit=limit)
    return [
        {"memory_kind": "active", "memory_id": item["id"], "summary_id": item["id"],
         "summary": item["content"], "revision": item["revision"], "occurred_at": item["updated_at"],
         "recall_score": 1.0, "recall_reasons": ["active_keyword"]}
        for item in items
    ]
