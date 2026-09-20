"""Global, persistent task checklist and its preset container contract."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime
import html
import time

from .container import ContainerContract


_snapshot: dict = {"plan": [], "explanation": None, "updated_at": 0}
_update_lock = asyncio.Lock()


def restore(snapshot: dict | None) -> None:
    """Restore at startup, before any tool updates are running."""
    global _snapshot, _update_lock
    _snapshot = deepcopy(snapshot) if snapshot is not None else {
        "plan": [], "explanation": None, "updated_at": 0,
    }
    _update_lock = asyncio.Lock()


def get_snapshot() -> dict:
    return deepcopy(_snapshot)


async def update_plan(plan: list[dict[str, str]], explanation: str | None = None) -> None:
    """Serialize writes on the main loop and publish only committed snapshots."""
    global _snapshot
    from database import save_todo_snapshot

    candidate = {"plan": deepcopy(plan), "explanation": explanation, "updated_at": 0}
    async with _update_lock:
        candidate["updated_at"] = int(time.time() * 1000)
        await save_todo_snapshot(candidate)
        _snapshot = candidate


def _render_container_todo(now: datetime) -> str:
    snapshot = _snapshot
    if not snapshot["plan"]:
        return ""
    lines = []
    if snapshot["explanation"] is not None:
        lines.append(f'<explanation>{html.escape(snapshot["explanation"])}</explanation>')
    for item in snapshot["plan"]:
        status = html.escape(item["status"], quote=True)
        lines.append(f'<item status="{status}">{html.escape(item["step"])}</item>')
    return "\n".join(lines)


CONTAINER_CONTRACT = ContainerContract(
    tag="todo",
    section="preset",
    description=(
        "This is your current task checklist. Keep it up to date with core.update_plan, "
        "submitting the complete ordered list each time. Mark finished steps completed; "
        "at most one step should be in_progress. Replace the list when plans change "
        "or submit an empty list to clear it."
    ),
    render=_render_container_todo,
)
