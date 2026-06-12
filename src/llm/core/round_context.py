"""Per-round LLM context visible to tool handlers."""

from __future__ import annotations

from contextvars import ContextVar, Token


_current_inner_state: ContextVar[dict] = ContextVar(
    "current_inner_state",
    default={},
)


def set_current_inner_state(inner_state: dict | None) -> Token:
    return _current_inner_state.set(dict(inner_state or {}))


def reset_current_inner_state(token: Token) -> None:
    _current_inner_state.reset(token)


def get_current_inner_state() -> dict:
    return dict(_current_inner_state.get() or {})
