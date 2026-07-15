"""Recover active command monitoring from persisted consciousness results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def running_command_ids_from_flow_dump(entries: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return command ids whose latest persisted workspace result is running."""

    latest: dict[str, str] = {}
    for entry in entries:
        responses = entry.get("responses")
        if not isinstance(responses, list):
            continue
        for response in responses:
            if not isinstance(response, Mapping):
                continue
            if response.get("namespace") != "workspace" or response.get("name") != "command":
                continue
            result = response.get("response")
            if not isinstance(result, Mapping):
                continue
            command_id = str(result.get("command_id") or "").strip()
            status = str(result.get("status") or "").strip()
            if command_id and status:
                latest[command_id] = status
    return tuple(command_id for command_id, status in latest.items() if status == "running")


__all__ = ["running_command_ids_from_flow_dump"]
