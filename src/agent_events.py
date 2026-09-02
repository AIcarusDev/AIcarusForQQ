"""Realtime Agent View event stream.

This module is intentionally UI-facing. It observes the existing round,
transport, and tool lifecycle without changing the model-facing AIC Action.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re
import threading
import time
import uuid
from collections import deque
from typing import Any

import app_state

_MAX_EVENTS = 1200
_MAX_TEXT = 12000
_MAX_PREVIEW = 520

_events: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
_queues: set[asyncio.Queue] = set()
_seq = itertools.count(1)
_stream_id = uuid.uuid4().hex
_lock = threading.RLock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _truncate(text: str, limit: int = _MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _safe_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, bytes):
        return f"[bytes {len(value)}]"
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:40]:
            result[str(key)] = _safe_json_value(item, depth=depth + 1)
        if len(value) > 40:
            result["…"] = f"{len(value) - 40} more keys"
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        result_items = [_safe_json_value(item, depth=depth + 1) for item in items[:40]]
        if len(items) > 40:
            result_items.append(f"… {len(items) - 40} more items")
        return result_items
    return _truncate(str(value), _MAX_PREVIEW)


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: _safe_json_value(value) for key, value in payload.items()}


def _put_event_to_queues(event: dict[str, Any]) -> None:
    with _lock:
        queues = list(_queues)
    for queue in queues:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def emit_agent_event(event_type: str, **payload: Any) -> dict[str, Any]:
    """Emit one UI-facing agent event from any thread."""
    with _lock:
        event = {
            "seq": next(_seq),
            "type": str(event_type),
            "created_at": _now_ms(),
            **_safe_payload(payload),
        }
        _events.append(event)

    loop = getattr(app_state, "main_loop", None)
    if loop and loop.is_running():
        loop.call_soon_threadsafe(_put_event_to_queues, event)
    return event


def snapshot_events(*, since: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    with _lock:
        events = [event for event in _events if int(event.get("seq") or 0) > since]
    if limit is not None and limit >= 0:
        return events[-limit:]
    return events


def subscribe_agent_events(maxsize: int = 1024) -> asyncio.Queue:
    queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
    with _lock:
        _queues.add(queue)
    return queue


def unsubscribe_agent_events(queue: asyncio.Queue) -> None:
    with _lock:
        _queues.discard(queue)


def agent_event_stats() -> dict[str, Any]:
    with _lock:
        latest = _events[-1] if _events else None
        return {
            "buffer_size": len(_events),
            "latest_seq": int(latest.get("seq") or 0) if latest else 0,
            "stream_id": _stream_id,
            "subscribers": len(_queues),
        }


def clear_agent_events_for_test() -> None:
    with _lock:
        _events.clear()


def summarize_tool_payload(value: Any) -> str:
    safe = _safe_json_value(value)
    if isinstance(safe, dict):
        for key in ("error", "reason", "message", "content", "to"):
            item = safe.get(key)
            if item:
                return _truncate(str(item), 180)
        bits: list[str] = []
        for key, item in list(safe.items())[:4]:
            if key.startswith("_"):
                continue
            bits.append(f"{key}={_short_scalar(item)}")
        return _truncate(", ".join(bits) or "{}", 220)
    if isinstance(safe, list):
        return _truncate(f"{len(safe)} items", 80)
    return _truncate(str(safe), 220)


def _short_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return _truncate(json.dumps(value, ensure_ascii=False, separators=(",", ":")), 80)
        except Exception:
            return _truncate(str(value), 80)
    return _truncate(str(value), 80)


def emit_agent_tool_hook(
    point: str,
    *,
    target: str,
    args: dict[str, Any] | None = None,
    result: Any = None,
    error: BaseException | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    context = dict(context or {})
    round_id = str(context.get("round_id") or "")
    if not round_id:
        return
    call_id = str(context.get("call_id") or "")
    base = {
        "round_id": round_id,
        "call_id": call_id,
        "tool_name": target,
        "module": context.get("module", ""),
    }
    if point == "before_call":
        emit_agent_event(
            "tool_started",
            **base,
            args=args or {},
            args_preview=summarize_tool_payload(args or {}),
        )
    elif point == "progress":
        emit_agent_event(
            "tool_progress",
            **base,
            message=str(context.get("message") or ""),
            args=args or {},
        )
    elif point == "guard_allowed":
        emit_agent_event(
            "tool_guard",
            **base,
            status="allowed",
            result=result,
            result_preview=summarize_tool_payload(result),
        )
    elif point == "guard_blocked":
        emit_agent_event(
            "tool_blocked",
            **base,
            result=result,
            result_preview=summarize_tool_payload(result),
        )
    elif point == "skipped":
        emit_agent_event(
            "tool_skipped",
            **base,
            result=result,
            result_preview=summarize_tool_payload(result),
        )
    elif point == "on_error":
        emit_agent_event(
            "tool_error",
            **base,
            error=str(error or ""),
            result=result,
            result_preview=summarize_tool_payload(result),
        )
    elif point == "finally_call":
        emit_agent_event(
            "tool_finished",
            **base,
            elapsed_ms=context.get("elapsed_ms"),
            result=result,
            result_preview=summarize_tool_payload(result),
            ok=_result_ok(result, error),
        )


def _result_ok(result: Any, error: BaseException | None) -> bool:
    if error is not None:
        return False
    if isinstance(result, dict):
        if result.get("tool_not_executed") or result.get("interrupted"):
            return False
        if "ok" in result:
            return bool(result.get("ok"))
        if result.get("error"):
            return False
    return True


class AgentActionStreamProjector:
    """Project streamed AIC Action text into human-facing deltas."""

    _TAG_RE = re.compile(r"^</?\s*([a-zA-Z_][\w:-]*)")

    def __init__(
        self,
        *,
        round_id: str,
        provider: str = "",
        model: str = "",
    ) -> None:
        self.round_id = round_id
        self.provider = provider
        self.model = model
        self._mode = "outside"
        self._tag_buf = ""
        self._text_buf: list[str] = []
        self._tool_buf: list[str] = []
        self._tool_index = 0

    def feed(self, text: str) -> None:
        for ch in text or "":
            if self._tag_buf:
                self._tag_buf += ch
                if ch == ">":
                    self._handle_tag(self._tag_buf)
                    self._tag_buf = ""
                continue

            if ch == "<":
                self._flush_text()
                self._tag_buf = "<"
                continue

            if self._mode == "cognition":
                self._text_buf.append(ch)
            elif self._mode == "tool_call":
                self._tool_buf.append(ch)
        self._flush_text()

    def finish(self) -> None:
        self._flush_text()
        if self._tag_buf:
            self._tag_buf = ""

    def _handle_tag(self, raw_tag: str) -> None:
        normalized = raw_tag.strip().lower()
        match = self._TAG_RE.match(normalized)
        name = match.group(1) if match else ""
        is_close = normalized.startswith("</")

        if name == "cognition":
            if is_close:
                self._mode = "outside"
                emit_agent_event("cognition_end", round_id=self.round_id)
            else:
                self._mode = "cognition"
                emit_agent_event("cognition_start", round_id=self.round_id)
            return

        if name == "action":
            if is_close:
                self._mode = "outside"
                emit_agent_event("action_end", round_id=self.round_id)
            else:
                self._mode = "action"
                emit_agent_event("action_start", round_id=self.round_id)
            return

        if name == "tool_call":
            if is_close:
                self._emit_best_effort_tool_plan()
                self._tool_buf = []
                self._mode = "action"
            else:
                self._tool_buf = []
                self._mode = "tool_call"

    def _flush_text(self) -> None:
        if not self._text_buf:
            return
        text = "".join(self._text_buf)
        self._text_buf = []
        if text:
            emit_agent_event(
                "cognition_delta",
                round_id=self.round_id,
                text=text,
                provider=self.provider,
                model=self.model,
            )

    def _emit_best_effort_tool_plan(self) -> None:
        body = "".join(self._tool_buf).strip()
        if not body:
            return
        try:
            value = json.loads(body)
        except Exception:
            return
        namespace = ""
        if isinstance(value, dict) and "function" in value and isinstance(value["function"], dict):
            name = str(value["function"].get("name") or "")
            namespace = str(value.get("namespace") or value["function"].get("namespace") or "")
            args = value["function"].get("arguments") or value.get("arguments") or {}
        elif isinstance(value, dict):
            name = str(value.get("name") or value.get("tool") or "")
            namespace = str(value.get("namespace") or "")
            args = value.get("arguments") or value.get("args") or {}
        else:
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}
        if not name:
            return
        namespace = namespace.strip()
        tool_name = f"{namespace}.{name}" if namespace else name
        self._tool_index += 1
        call_id = f"call_{self._tool_index}"
        payload = {
            "round_id": self.round_id,
            "call_id": call_id,
            "tool_index": self._tool_index,
            "tool_name": tool_name,
            "args": args,
            "args_preview": summarize_tool_payload(args),
        }
        if namespace:
            payload["namespace"] = namespace
        emit_agent_event("tool_planned", **payload)
