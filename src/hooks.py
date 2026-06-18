"""Lightweight runtime hook contract.

Hooks are intentionally observation-only in this first version: subscribers can
see events, but mutating an event is not a supported way to change execution.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

logger = logging.getLogger("AICQ.hooks")


HookHandler = Callable[["HookEvent"], None]


@dataclass(frozen=True)
class HookEvent:
    """A single runtime hook notification."""

    name: str
    point: str
    target: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: BaseException | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HookRegistry:
    """Thread-safe subscriber list for runtime hooks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: list[HookHandler] = []

    def subscribe(self, handler: HookHandler) -> Callable[[], None]:
        """Register a hook handler and return an unsubscribe callback."""
        with self._lock:
            if handler not in self._subscribers:
                self._subscribers.append(handler)

        def _unsubscribe() -> None:
            self.unsubscribe(handler)

        return _unsubscribe

    def unsubscribe(self, handler: HookHandler) -> None:
        with self._lock:
            self._subscribers = [
                subscriber for subscriber in self._subscribers
                if subscriber is not handler
            ]

    def clear(self) -> None:
        with self._lock:
            self._subscribers.clear()

    def emit(self, event: HookEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                logger.warning("[hooks] hook subscriber failed: %s", event.name, exc_info=True)


runtime_hooks = HookRegistry()
_current_scope = threading.local()


def _event_name(namespace: str, point: str) -> str:
    return f"{namespace}.{point}"


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def emit_hook(
    *,
    namespace: str,
    point: str,
    target: str,
    args: dict[str, Any] | None = None,
    result: Any = None,
    error: BaseException | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Emit a hook event to all current subscribers."""
    runtime_hooks.emit(
        HookEvent(
            name=_event_name(namespace, point),
            point=point,
            target=target,
            args=_copy_mapping(args),
            result=dict(result) if isinstance(result, dict) else result,
            error=error,
            context=_copy_mapping(context),
        )
    )


@contextmanager
def hook_scope(
    *,
    namespace: str,
    target: str,
    context: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Make progress hooks emitted inside this block inherit a target/context."""
    previous = getattr(_current_scope, "value", None)
    _current_scope.value = {
        "namespace": namespace,
        "target": target,
        "context": _copy_mapping(context),
    }
    try:
        yield
    finally:
        _current_scope.value = previous


def emit_progress(
    message: str = "",
    *,
    target: str | None = None,
    args: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Emit a progress hook from inside a function/tool execution."""
    scope = getattr(_current_scope, "value", None) or {}
    namespace = str(scope.get("namespace") or "function")
    resolved_target = target or str(scope.get("target") or "")
    merged_context = _copy_mapping(scope.get("context"))
    merged_context.update(_copy_mapping(context))
    if message:
        merged_context["message"] = message
    emit_hook(
        namespace=namespace,
        point="progress",
        target=resolved_target,
        args=args,
        context=merged_context,
    )


@contextmanager
def hook_subscription(handler: HookHandler) -> Iterator[None]:
    """Temporarily subscribe a handler, useful for tests and diagnostics."""
    unsubscribe = runtime_hooks.subscribe(handler)
    try:
        yield
    finally:
        unsubscribe()


__all__ = [
    "HookEvent",
    "HookHandler",
    "HookRegistry",
    "emit_hook",
    "emit_progress",
    "hook_scope",
    "hook_subscription",
    "runtime_hooks",
]
