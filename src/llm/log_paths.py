"""Shared path helpers for local LLM diagnostic logs."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_log_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
