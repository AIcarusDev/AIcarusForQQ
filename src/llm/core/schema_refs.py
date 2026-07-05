"""Small JSON Schema reference helpers shared by prompt and validation code."""

from __future__ import annotations

from typing import Any


def resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, dict) else None
