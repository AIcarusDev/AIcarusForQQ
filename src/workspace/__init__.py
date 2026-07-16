"""Internal service for the Agent's model-facing Linux computer."""

from .backend import WorkspaceBackend, WslWorkspaceBackend
from .config import PROTOCOL_VERSION, WorkspaceConfig, WorkspaceProvisionConfig
from .errors import WorkspaceError, WorkspaceErrorCode
from .models import CommandResult, EnsureResult, FileReadResult, HealthResult, TextListResult
from .service import WorkspaceService

__all__ = [
    "CommandResult",
    "EnsureResult",
    "FileReadResult",
    "HealthResult",
    "PROTOCOL_VERSION",
    "TextListResult",
    "WorkspaceBackend",
    "WorkspaceConfig",
    "WorkspaceProvisionConfig",
    "WorkspaceError",
    "WorkspaceErrorCode",
    "WorkspaceService",
    "WslWorkspaceBackend",
]
