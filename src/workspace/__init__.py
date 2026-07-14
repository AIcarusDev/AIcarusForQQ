"""Internal Linux workspace foundation (not a model-facing tool surface)."""

from .backend import WorkspaceBackend, WslWorkspaceBackend
from .config import PROTOCOL_VERSION, WorkspaceConfig, WorkspaceProvisionConfig
from .errors import WorkspaceError, WorkspaceErrorCode
from .models import CommandResult, EnsureResult, HealthResult, StreamResult
from .service import WorkspaceService

__all__ = [
    "CommandResult",
    "EnsureResult",
    "HealthResult",
    "PROTOCOL_VERSION",
    "StreamResult",
    "WorkspaceBackend",
    "WorkspaceConfig",
    "WorkspaceProvisionConfig",
    "WorkspaceError",
    "WorkspaceErrorCode",
    "WorkspaceService",
    "WslWorkspaceBackend",
]
