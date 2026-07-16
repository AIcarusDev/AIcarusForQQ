"""Structured workspace errors shared by service and transport layers."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class WorkspaceErrorCode(str, Enum):
    WORKSPACE_NOT_BUILT = "workspace_not_built"
    WORKSPACE_NEEDS_UPGRADE = "workspace_needs_upgrade"
    WORKSPACE_BUSY = "workspace_busy"
    DISTRO_UNAVAILABLE = "distro_unavailable"
    BROKER_UNAVAILABLE = "broker_unavailable"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    CONTAINER_START_FAILED = "container_start_failed"
    PREVIEW_UNAVAILABLE = "preview_unavailable"
    INVALID_ARGUMENT = "invalid_argument"
    COMMAND_NOT_FOUND = "command_not_found"
    PATH_ERROR = "path_error"
    BINARY_FILE = "binary_file"
    FILE_NOT_READ = "file_not_read"
    STALE_FILE = "stale_file"
    AMBIGUOUS_EDIT = "ambiguous_edit"
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
