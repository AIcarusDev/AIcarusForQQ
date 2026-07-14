"""Internal configuration for the isolated Linux workspace."""

from __future__ import annotations

import ntpath
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


PROTOCOL_VERSION = 2
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_DISTRO_NAME = "AICQ-Workspace"
DEFAULT_APPLIANCE_USER = "aicqws"
DEFAULT_BRIDGE_PATH = "/usr/local/bin/aicq-workspace-bridge"

MAX_COMMAND_TIMEOUT_SECONDS = 900.0
COMMAND_OBSERVATION_SECONDS = 15.0
MAX_COMMAND_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_COMMAND_PAGE_BYTES = 64 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_STDIN_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 1024 * 1024
MAX_READ_PAGE_BYTES = 256 * 1024
MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000


@dataclass(frozen=True, slots=True)
class WorkspaceConfig:
    """Core-side transport settings.

    Constructing this value or a backend is deliberately side-effect free.
    WSL is only started by an explicit service method call.
    """

    distro_name: str = DEFAULT_DISTRO_NAME
    appliance_user: str = DEFAULT_APPLIANCE_USER
    bridge_path: str = DEFAULT_BRIDGE_PATH
    wsl_executable: str = "wsl.exe"
    bridge_grace_seconds: float = 30.0


_WINDOWS_ENV_RE = re.compile(r"%([^%]+)%")


def _expand_windows_env(value: str, environ: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        for candidate, replacement in environ.items():
            if candidate.casefold() == key.casefold():
                return replacement
        return match.group(0)

    return _WINDOWS_ENV_RE.sub(replace, value)


@dataclass(frozen=True, slots=True)
class WorkspaceProvisionConfig:
    """Machine-local provisioning settings loaded from the user config.

    Provisioning remains an explicit developer action. Reading this value does
    not inspect WSL, create directories, or start the workspace appliance.
    """

    install_root: str

    @classmethod
    def from_root_config(
        cls,
        config: Mapping[str, Any],
        *,
        environ: Mapping[str, str],
    ) -> "WorkspaceProvisionConfig":
        workspace = config.get("workspace", {})
        if workspace is None:
            workspace = {}
        if not isinstance(workspace, Mapping):
            raise ValueError("workspace config must be a mapping")

        provisioning = workspace.get("provisioning", {})
        if provisioning is None:
            provisioning = {}
        if not isinstance(provisioning, Mapping):
            raise ValueError("workspace.provisioning config must be a mapping")

        configured = str(provisioning.get("install_root", "") or "").strip()
        if configured:
            install_root = _expand_windows_env(configured, environ)
        else:
            local_app_data = str(environ.get("LOCALAPPDATA", "") or "").strip()
            if not local_app_data:
                user_profile = str(environ.get("USERPROFILE", "") or "").strip()
                if not user_profile:
                    raise ValueError(
                        "workspace install_root is empty and no Windows user profile is available"
                    )
                local_app_data = ntpath.join(user_profile, "AppData", "Local")
            install_root = ntpath.join(local_app_data, "AICQ", "Workspace")

        install_root = ntpath.normpath(install_root)
        drive, tail = ntpath.splitdrive(install_root)
        if len(drive) != 2 or drive[1] != ":" or not tail.startswith(("\\", "/")):
            raise ValueError("workspace install_root must be an absolute local Windows drive path")
        return cls(install_root=install_root)
