"""Typed internal results for workspace operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class StreamResult:
    text: str
    total_bytes: int
    truncated: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "StreamResult":
        data = payload or {}
        return cls(
            text=str(data.get("text", "")),
            total_bytes=max(0, int(data.get("total_bytes", 0) or 0)),
            truncated=bool(data.get("truncated", False)),
        )


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
    stdout: StreamResult
    stderr: StreamResult

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CommandResult":
        raw_exit = payload.get("exit_code")
        return cls(
            command_id=str(payload.get("command_id", "")),
            workspace_id=str(payload.get("workspace_id", "default")),
            status=str(payload.get("status", "unknown")),
            cwd=str(payload.get("cwd", "/workspace")),
            exit_code=None if raw_exit is None else int(raw_exit),
            started_at=str(payload.get("started_at", "")),
            finished_at=(
                None if payload.get("finished_at") is None else str(payload.get("finished_at"))
            ),
            timed_out=bool(payload.get("timed_out", False)),
            stdout=StreamResult.from_payload(payload.get("stdout")),
            stderr=StreamResult.from_payload(payload.get("stderr")),
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
