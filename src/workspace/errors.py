"""Structured workspace errors shared by service and transport layers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class WorkspaceErrorCode(str, Enum):
    DISTRO_UNAVAILABLE = "distro_unavailable"
    BROKER_UNAVAILABLE = "broker_unavailable"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    CONTAINER_START_FAILED = "container_start_failed"
    INVALID_ARGUMENT = "invalid_argument"
    COMMAND_TIMEOUT = "command_timeout"
    PATH_ERROR = "path_error"
    INTERNAL_ERROR = "internal_error"


class WorkspaceError(RuntimeError):
    """A stable, machine-readable failure returned by the workspace stack."""

    def __init__(
        self,
        code: WorkspaceErrorCode | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        if isinstance(code, WorkspaceErrorCode):
            self.code = code
        else:
            try:
                self.code = WorkspaceErrorCode(str(code))
            except ValueError:
                self.code = WorkspaceErrorCode.INTERNAL_ERROR
        self.message = str(message)
        self.details = dict(details or {})
        self.request_id = request_id
        super().__init__(f"{self.code.value}: {self.message}")
