"""Internal configuration for the Agent's isolated Linux computer."""

from __future__ import annotations

import ntpath
import os
import re
import ctypes
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil


PROTOCOL_VERSION = 4
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_AGENT_HOME = "/home/agent"
DEFAULT_DISTRO_NAME = "AICQ-Workspace"
DEFAULT_APPLIANCE_USER = "aicqws"
DEFAULT_BRIDGE_PATH = "/usr/local/bin/aicq-workspace-bridge"
DEFAULT_CONTAINER_NAME = "aicq-workspace-default"
DEFAULT_INSTALL_ROOT = "data/computer"
DEFAULT_CPUS = 4
DEFAULT_MEMORY_GIB = 8
DEFAULT_DISK_GIB = 64
MIN_MEMORY_GIB = 2
MAX_MEMORY_GIB = 64
MIN_DISK_GIB = 32
MAX_DISK_GIB = 512
FIXED_PIDS_LIMIT = 1024

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
    enabled: bool = False
    cpus: int = DEFAULT_CPUS
    memory_gib: int = DEFAULT_MEMORY_GIB
    disk_gib: int = DEFAULT_DISK_GIB

    @classmethod
    def from_root_config(
        cls,
        config: Mapping[str, Any],
        *,
        environ: Mapping[str, str],
        project_root: str | os.PathLike[str] | None = None,
    ) -> "WorkspaceProvisionConfig":
        workspace = config.get("workspace", {})
        if workspace is None:
            workspace = {}
        if not isinstance(workspace, Mapping):
            raise ValueError("Agent computer config must be a mapping")

        legacy = workspace.get("provisioning", {})
        if legacy is None:
            legacy = {}
        if not isinstance(legacy, Mapping):
            raise ValueError("legacy Agent computer provisioning config must be a mapping")
        resources = workspace.get("resources", {})
        if resources is None:
            resources = {}
        if not isinstance(resources, Mapping):
            raise ValueError("Agent computer resources config must be a mapping")

        configured = str(
            workspace.get("install_root", legacy.get("install_root", DEFAULT_INSTALL_ROOT))
            or DEFAULT_INSTALL_ROOT
        ).strip()
        configured = _expand_windows_env(configured, environ)
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        if ntpath.isabs(configured):
            install_root = ntpath.normpath(configured)
        else:
            install_root = ntpath.normpath(str(root / Path(configured)))

        drive, tail = ntpath.splitdrive(install_root)
        if len(drive) != 2 or drive[1] != ":" or not tail.startswith(("\\", "/")):
            raise ValueError("Agent computer install_root must be an absolute local Windows drive path")
        if install_root.startswith(("\\\\", "//")):
            raise ValueError("Agent computer install_root cannot be a UNC path")

        normalized_root = ntpath.normcase(ntpath.normpath(str(root)))
        normalized_install = ntpath.normcase(install_root)
        drive_root = ntpath.normcase(ntpath.normpath(f"{drive}\\"))
        if normalized_install == drive_root:
            raise ValueError("Agent computer install_root cannot be a drive root")
        if normalized_install == normalized_root or normalized_root.startswith(normalized_install + "\\"):
            raise ValueError("Agent computer install_root cannot be the project root or one of its parents")

        protected = [
            environ.get("WINDIR", ""),
            environ.get("ProgramFiles", ""),
            environ.get("ProgramFiles(x86)", ""),
            environ.get("ProgramData", ""),
            environ.get("USERPROFILE", ""),
        ]
        for raw_protected in protected:
            if not raw_protected:
                continue
            protected_path = ntpath.normcase(ntpath.normpath(str(raw_protected)))
            if normalized_install == protected_path or protected_path.startswith(normalized_install + "\\"):
                raise ValueError("Agent computer install_root cannot be a protected Windows directory")
        if os.name == "nt":
            # DRIVE_FIXED=3. Reject removable, optical and network-backed roots.
            drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\"))
            if drive_type != 3:
                raise ValueError("Agent computer install_root must be on a fixed local drive")

        max_cpus = max(1, min(32, int(os.cpu_count() or 1)))
        cpus = _bounded_int(resources.get("cpus", DEFAULT_CPUS), "Agent computer CPU cores", 1, max_cpus)
        physical_memory_gib = max(1, int(psutil.virtual_memory().total // (1024**3)))
        max_memory = max(MIN_MEMORY_GIB, min(MAX_MEMORY_GIB, physical_memory_gib))
        memory_gib = _bounded_int(
            resources.get("memory_gib", DEFAULT_MEMORY_GIB),
            "Agent computer memory_gib",
            MIN_MEMORY_GIB,
            max_memory,
        )
        disk_gib = _bounded_int(
            resources.get("disk_gib", DEFAULT_DISK_GIB),
            "Agent computer disk_gib",
            MIN_DISK_GIB,
            MAX_DISK_GIB,
        )
        return cls(
            install_root=install_root,
            enabled=workspace.get("enabled") is True,
            cpus=cpus,
            memory_gib=memory_gib,
            disk_gib=disk_gib,
        )

    def to_config_dict(self, *, project_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
        root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        default_path = ntpath.normcase(ntpath.normpath(str(root / DEFAULT_INSTALL_ROOT)))
        install_root = (
            DEFAULT_INSTALL_ROOT
            if ntpath.normcase(ntpath.normpath(self.install_root)) == default_path
            else self.install_root
        )
        return {
            "enabled": self.enabled,
            "install_root": install_root,
            "resources": {
                "cpus": self.cpus,
                "memory_gib": self.memory_gib,
                "disk_gib": self.disk_gib,
            },
        }

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "install_root": self.install_root,
            "resources": {
                "cpus": self.cpus,
                "memory_gib": self.memory_gib,
                "disk_gib": self.disk_gib,
            },
            "limits": {
                "max_cpus": max(1, min(32, int(os.cpu_count() or 1))),
                "max_memory_gib": max(
                    MIN_MEMORY_GIB,
                    min(MAX_MEMORY_GIB, int(psutil.virtual_memory().total // (1024**3))),
                ),
                "min_disk_gib": MIN_DISK_GIB,
                "max_disk_gib": MAX_DISK_GIB,
            },
        }


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def normalize_workspace_config_inplace(
    config: dict[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> WorkspaceProvisionConfig:
    """Move legacy provisioning config into the single canonical workspace shape."""

    normalized = WorkspaceProvisionConfig.from_root_config(
        config,
        environ=environ or os.environ,
        project_root=project_root,
    )
    config["workspace"] = normalized.to_config_dict(project_root=project_root)
    return normalized


def workspace_enabled(config: Mapping[str, Any]) -> bool:
    workspace = config.get("workspace")
    return bool(isinstance(workspace, Mapping) and workspace.get("enabled") is True)
