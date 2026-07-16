"""Typed internal results for workspace operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import DEFAULT_AGENT_HOME


@dataclass(frozen=True, slots=True)
class CommandResult:
    command_id: str
    workspace_id: str
    status: str
    cwd: str
    exit_code: int | None
    started_at: str
    finished_at: str | None
    timed_out: bool
    cursor: int = 0
    has_more: bool = False
    truncated: bool = False
    content: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in {"completed", "timed_out", "stopped", "interrupted"}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CommandResult":
        raw_exit = payload.get("exit_code")
        return cls(
            command_id=str(payload.get("command_id", "")),
            workspace_id=str(payload.get("workspace_id", "default")),
            status=str(payload.get("status", "unknown")),
            cwd=str(payload.get("cwd", DEFAULT_AGENT_HOME)),
            exit_code=None if raw_exit is None else int(raw_exit),
            started_at=str(payload.get("started_at", "")),
            finished_at=(None if payload.get("finished_at") is None else str(payload.get("finished_at"))),
            timed_out=bool(payload.get("timed_out", False)),
            cursor=max(0, int(payload.get("cursor", 0) or 0)),
            has_more=bool(payload.get("has_more", False)),
            truncated=bool(payload.get("truncated", False)),
            content=str(payload.get("content", "")),
        )


@dataclass(frozen=True, slots=True)
class FileReadResult:
    path: str
    content: str
    revision: str
    start_line: int
    end_line: int
    total_lines: int
    has_more: bool
    next_line: int | None
    truncated_lines: tuple[int, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FileReadResult":
        raw_next = payload.get("next_line")
        return cls(
            path=str(payload.get("path", "")),
            content=str(payload.get("content", "")),
            revision=str(payload.get("revision", "")),
            start_line=max(1, int(payload.get("start_line", 1) or 1)),
            end_line=max(0, int(payload.get("end_line", 0) or 0)),
            total_lines=max(0, int(payload.get("total_lines", 0) or 0)),
            has_more=bool(payload.get("has_more", False)),
            next_line=None if raw_next is None else int(raw_next),
            truncated_lines=tuple(int(value) for value in payload.get("truncated_lines") or []),
        )


@dataclass(frozen=True, slots=True)
class TextListResult:
    content: str
    count: int
    offset: int
    next_offset: int | None
    has_more: bool
    truncated: bool
    path: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TextListResult":
        raw_next = payload.get("next_offset")
        return cls(
            content=str(payload.get("content", "")),
            count=max(0, int(payload.get("count", 0) or 0)),
            offset=max(0, int(payload.get("offset", 0) or 0)),
            next_offset=None if raw_next is None else int(raw_next),
            has_more=bool(payload.get("has_more", False)),
            truncated=bool(payload.get("truncated", False)),
            path=str(payload.get("path", DEFAULT_AGENT_HOME)),
        )


@dataclass(frozen=True, slots=True)
class HealthResult:
    protocol_version: int
    broker_version: str
    distro: str
    container_exists: bool
    container_running: bool
    image_digest: str
    firewall_active: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HealthResult":
        return cls(
            protocol_version=int(payload.get("protocol_version", 0) or 0),
            broker_version=str(payload.get("broker_version", "")),
            distro=str(payload.get("distro", "")),
            container_exists=bool(payload.get("container_exists", False)),
            container_running=bool(payload.get("container_running", False)),
            image_digest=str(payload.get("image_digest", "")),
            firewall_active=bool(payload.get("firewall_active", False)),
        )


@dataclass(frozen=True, slots=True)
class EnsureResult:
    workspace_id: str
    container_name: str
    created: bool
    started: bool
    image_digest: str
    limits: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EnsureResult":
        return cls(
            workspace_id=str(payload.get("workspace_id", "default")),
            container_name=str(payload.get("container_name", "")),
            created=bool(payload.get("created", False)),
            started=bool(payload.get("started", False)),
            image_digest=str(payload.get("image_digest", "")),
            limits=dict(payload.get("limits") or {}),
        )


__all__ = [
    "CommandResult",
    "EnsureResult",
    "FileReadResult",
    "HealthResult",
    "TextListResult",
]
