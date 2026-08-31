"""User-owned workspace control plane and detached job runner."""

from __future__ import annotations

import json
import ntpath
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping

import psutil

from .config import (
    DEFAULT_CONTAINER_NAME,
    DEFAULT_DISTRO_NAME,
    PROTOCOL_VERSION,
    WorkspaceProvisionConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_ROOT = REPO_ROOT / "data" / "workspace-control"
JOBS_ROOT = CONTROL_ROOT / "jobs"
LOCK_PATH = CONTROL_ROOT / "active-job.json"
DIRECTORY_SELECTIONS_ROOT = CONTROL_ROOT / "directory-selections"
WORKER_SCRIPT = REPO_ROOT / "scripts" / "workspace" / "workspace_worker.py"
PROVISION_SCRIPT = REPO_ROOT / "scripts" / "workspace" / "provision-workspace.ps1"
APPLY_RESOURCES_SCRIPT = REPO_ROOT / "scripts" / "workspace" / "apply-workspace-resources.ps1"
MAINTENANCE_SCRIPT = REPO_ROOT / "scripts" / "workspace" / "workspace-maintenance.ps1"
SOURCE_MANIFEST_PATH = REPO_ROOT / "scripts" / "workspace" / "appliance" / "opt" / "aicq-workspace" / "protocol-manifest.json"
MANAGED_MARKER = ".aicq-workspace-managed.json"
PROVISIONING_MARKER = ".aicq-workspace-provisioning.json"

JOB_ACTIONS = {"build", "apply", "upgrade", "rebuild", "restart", "clear", "uninstall"}
TERMINAL_JOB_STATES = {"ready", "failed", "waiting_reboot"}
ACTION_STATES = {
    "build": "building",
    "apply": "applying",
    "upgrade": "upgrading",
    "rebuild": "rebuilding",
    "restart": "restarting",
    "clear": "clearing",
    "uninstall": "uninstalling",
}
ACTION_CONFIRMATIONS = {
    "rebuild": "REBUILD COMPUTER",
    "restart": "RESTART COMPUTER",
    "clear": "ERASE AGENT HOME",
    "uninstall": "UNINSTALL COMPUTER",
}


class WorkspaceControlError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _directory_selection_path(selection_id: str, *, control_root: Path = CONTROL_ROOT) -> Path:
    if not selection_id or not all(ch.isalnum() or ch in "-_" for ch in selection_id):
        raise WorkspaceControlError("无效的 Agent 电脑目录选择 ID")
    return control_root / "directory-selections" / f"{selection_id}.json"


def publish_workspace_directory_selection(
    selection_id: str,
    *,
    status: str,
    path: str = "",
    error: str = "",
    control_root: Path = CONTROL_ROOT,
) -> None:
    if status not in {"selected", "canceled", "failed"}:
        raise WorkspaceControlError("无效的 Agent 电脑目录选择状态")
    _atomic_json(
        _directory_selection_path(selection_id, control_root=control_root),
        {
            "selection_id": selection_id,
            "status": status,
            "path": str(path or ""),
            "error": str(error or ""),
            "created_at": _utc_now(),
        },
    )


def consume_workspace_directory_selection(
    selection_id: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any] | None:
    selection_path = _directory_selection_path(selection_id, control_root=control_root)
    selection = _read_json(selection_path)
    if selection is None:
        return None
    try:
        selection_path.unlink()
    except FileNotFoundError:
        pass
    return selection


def _pid_alive(pid: Any) -> bool:
    try:
        return int(pid) > 0 and psutil.pid_exists(int(pid))
    except (TypeError, ValueError):
        return False


def _decode_output(raw: bytes) -> str:
    if not raw:
        return ""
    # Console-less wsl.exe writes UTF-16LE while PowerShell and Python write
    # UTF-8 to the same redirected stream. UTF-16LE ASCII is also valid UTF-8,
    # so trying UTF-8 first silently produces NUL-filled mojibake.
    nul_ratio = raw.count(b"\x00") / len(raw)
    encodings = ("utf-16-le", "utf-8", "gb18030") if nul_ratio >= 0.05 else ("utf-8", "gb18030", "utf-16-le")
    for encoding in encodings:
        try:
            return raw.decode(encoding).replace("\x00", "")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").replace("\x00", "")


def _iter_decoded_output_lines(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode a mixed UTF-8/UTF-16LE redirected Windows process stream."""

    pending = b""
    for chunk in chunks:
        pending += chunk
        while True:
            newline = pending.find(b"\n")
            if newline < 0 or newline == len(pending) - 1:
                break
            utf16_line = pending[newline + 1] == 0
            end = newline + (2 if utf16_line else 1)
            raw_line, pending = pending[:end], pending[end:]
            text = _decode_output(raw_line).replace("\r\n", "\n").replace("\r", "\n")
            yield text if text.endswith("\n") else text + "\n"
    if pending:
        text = _decode_output(pending).replace("\r\n", "\n").replace("\r", "\n")
        if text:
            yield text if text.endswith("\n") else text + "\n"


def _read_process_chunks(stream: BinaryIO, *, size: int = 4096) -> Iterator[bytes]:
    read = getattr(stream, "read1", stream.read)
    while True:
        chunk = read(size)
        if not chunk:
            return
        yield chunk


def _run_capture(argv: list[str], *, timeout: float = 20.0) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return_code = int(completed.returncode)
    stdout = _decode_output(completed.stdout or b"").strip()
    if return_code == 0:
        # WSL can emit host diagnostics (for example proxy mirroring warnings)
        # on stderr even when the requested command succeeded. Machine-readable
        # probes must parse stdout alone.
        return return_code, stdout
    stderr = _decode_output(completed.stderr or b"").strip()
    return return_code, "\n".join(part for part in (stdout, stderr) if part)


@dataclass(frozen=True, slots=True)
class WorkspaceObservedState:
    state: str
    built: bool
    path_locked: bool
    distro_exists: bool
    image_exists: bool
    container_exists: bool
    container_running: bool
    managed: bool
    partial_install: bool
    partial_repair_mode: str
    actual_install_location: str
    install_location_matches: bool
    protocol_version: int | None
    broker_version: str
    installed_resources: dict[str, int] | None
    pending_changes: list[str]
    isolated_network_ready: bool = False
    egress_firewall_ready: bool = False
    browser_tunnel_ready: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorkspaceControlPlane:
    """Inspect installation state and coordinate one detached control job."""

    def __init__(self, *, control_root: Path = CONTROL_ROOT) -> None:
        self.control_root = control_root
        self.jobs_root = control_root / "jobs"
        self.lock_path = control_root / "active-job.json"

    def _job_path(self, job_id: str) -> Path:
        if not job_id or not all(ch.isalnum() or ch in "-_" for ch in job_id):
            raise WorkspaceControlError("无效的 Agent 电脑任务 ID")
        return self.jobs_root / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self.jobs_root / f"{job_id}.log"

    def _load_lock(self) -> dict[str, Any] | None:
        lock = _read_json(self.lock_path)
        if not lock:
            return None
        job = _read_json(self._job_path(str(lock.get("job_id") or "")))
        if job and job.get("status") not in TERMINAL_JOB_STATES and _pid_alive(lock.get("pid")):
            return lock
        if job and job.get("status") not in TERMINAL_JOB_STATES and not _pid_alive(lock.get("pid")):
            job.update(
                status="failed",
                stage="worker_exited",
                error="Agent 电脑控制进程意外退出",
                finished_at=_utc_now(),
            )
            _atomic_json(self._job_path(str(job["job_id"])), job)
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        return None

    def is_busy(self) -> bool:
        return self._load_lock() is not None

    def current_job(self) -> dict[str, Any] | None:
        lock = self._load_lock()
        if lock:
            return _read_json(self._job_path(str(lock.get("job_id") or "")))
        jobs = sorted(self.jobs_root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True) if self.jobs_root.exists() else []
        return _read_json(jobs[0]) if jobs else None

    def get_job(self, job_id: str, *, cursor: int = 0) -> dict[str, Any]:
        job = _read_json(self._job_path(job_id))
        if not job:
            raise WorkspaceControlError("Agent 电脑任务不存在", status_code=404)
        log_path = self._log_path(job_id)
        raw = b""
        next_cursor = max(0, int(cursor or 0))
        try:
            with log_path.open("rb") as handle:
                handle.seek(next_cursor)
                raw = handle.read(256 * 1024)
                next_cursor = handle.tell()
        except FileNotFoundError:
            pass
        return {
            **job,
            "log": _decode_output(raw),
            "log_cursor": next_cursor,
            "log_has_more": bool(log_path.exists() and log_path.stat().st_size > next_cursor),
        }

    def _wsl(self, *args: str, timeout: float = 20.0) -> tuple[int, str]:
        return _run_capture(["wsl.exe", *args], timeout=timeout)

    def _managed_marker_exists(self, config: WorkspaceProvisionConfig) -> bool:
        expected_location = Path(config.install_root) / DEFAULT_DISTRO_NAME
        marker = _read_json(expected_location / MANAGED_MARKER)
        if not marker or marker.get("distro_name") != DEFAULT_DISTRO_NAME:
            return False
        try:
            marked_location = ntpath.normcase(ntpath.normpath(str(marker.get("install_location") or "")))
            return marked_location == ntpath.normcase(ntpath.normpath(str(expected_location)))
        except (TypeError, ValueError):
            return False

    def _provisioning_marker_exists(self, config: WorkspaceProvisionConfig) -> bool:
        expected_location = Path(config.install_root) / DEFAULT_DISTRO_NAME
        marker = _read_json(Path(config.install_root) / PROVISIONING_MARKER)
        if not marker or marker.get("distro_name") != DEFAULT_DISTRO_NAME:
            return False
        try:
            marked_location = ntpath.normcase(ntpath.normpath(str(marker.get("install_location") or "")))
            return marked_location == ntpath.normcase(ntpath.normpath(str(expected_location)))
        except (TypeError, ValueError):
            return False

    def _distro_install_location(self) -> str:
        if os.name != "nt":
            return ""
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Lxss",
            ) as root:
                index = 0
                while True:
                    try:
                        child_name = winreg.EnumKey(root, index)
                    except OSError:
                        break
                    index += 1
                    try:
                        with winreg.OpenKey(root, child_name) as child:
                            distro_name = str(winreg.QueryValueEx(child, "DistributionName")[0])
                            if distro_name.casefold() != DEFAULT_DISTRO_NAME.casefold():
                                continue
                            return os.path.expandvars(str(winreg.QueryValueEx(child, "BasePath")[0]))
                    except OSError:
                        continue
        except OSError:
            pass
        return ""

    def _distro_names(self) -> tuple[list[str], str]:
        code, output = self._wsl("--list", "--quiet")
        if code != 0:
            return [], output
        return [line.strip() for line in output.splitlines() if line.strip()], ""

    def probe(self, config: WorkspaceProvisionConfig) -> WorkspaceObservedState:
        names, list_error = self._distro_names()
        if DEFAULT_DISTRO_NAME.casefold() not in {name.casefold() for name in names}:
            return WorkspaceObservedState(
                state="not_built",
                built=False,
                path_locked=False,
                distro_exists=False,
                image_exists=False,
                container_exists=False,
                container_running=False,
                managed=False,
                partial_install=False,
                partial_repair_mode="",
                actual_install_location="",
                install_location_matches=False,
                protocol_version=None,
                broker_version="",
                installed_resources=None,
                pending_changes=[],
                error=list_error,
            )

        actual_install_location = self._distro_install_location()
        expected_install_location = str(Path(config.install_root) / DEFAULT_DISTRO_NAME)
        install_location_matches = bool(
            actual_install_location
            and ntpath.normcase(ntpath.normpath(actual_install_location))
            == ntpath.normcase(ntpath.normpath(expected_install_location))
        )

        manifest_code, manifest_text = self._wsl(
            "--distribution", DEFAULT_DISTRO_NAME, "--user", "root", "--exec",
            "/bin/cat", "/opt/aicq-workspace/protocol-manifest.json",
        )
        manifest: dict[str, Any] = {}
        if manifest_code == 0:
            try:
                loaded = json.loads(manifest_text.lstrip("\ufeff"))
                manifest = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                pass

        resource_code, resource_text = self._wsl(
            "--distribution", DEFAULT_DISTRO_NAME, "--user", "root", "--exec",
            "/bin/cat", "/etc/aicq-workspace-config.json",
        )
        installed_resources: dict[str, int] | None = None
        if resource_code == 0:
            try:
                raw_resources = json.loads(resource_text.lstrip("\ufeff"))
                installed_resources = {
                    "cpus": int(raw_resources["cpus"]),
                    "memory_gib": int(raw_resources["memory_gib"]),
                    "disk_gib": int(raw_resources["disk_gib"]),
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                installed_resources = None
        elif isinstance(manifest.get("limits"), dict):
            limits = manifest["limits"]
            installed_resources = {
                "cpus": int(limits.get("cpus", 0) or 0),
                "memory_gib": int(int(limits.get("memory_bytes", 0) or 0) // (1024**3)),
                "disk_gib": 64,
            }

        podman_prefix = [
            "--distribution", DEFAULT_DISTRO_NAME, "--user", "aicqws", "--exec",
            "/usr/bin/env", "XDG_RUNTIME_DIR=/run/user/1000",
            "DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus", "/usr/bin/podman",
        ]
        expected_manifest = _read_json(SOURCE_MANIFEST_PATH) or {}
        isolation_config = expected_manifest.get("network_isolation")
        expected_isolated_network = (
            str(isolation_config.get("network") or "") if isinstance(isolation_config, dict) else ""
        )
        image_code, _ = self._wsl(
            *podman_prefix,
            "image",
            "exists",
            str(manifest.get("image_name") or "localhost/aicq-workspace-dev:5"),
        )
        container_code, _ = self._wsl(*podman_prefix, "container", "exists", DEFAULT_CONTAINER_NAME)
        running = False
        isolated_network_ready = False
        egress_firewall_ready = False
        tunnel_code, _ = self._wsl(
            "--distribution",
            DEFAULT_DISTRO_NAME,
            "--user",
            "aicqws",
            "--exec",
            "/usr/bin/test",
            "-x",
            "/usr/local/bin/aicq-workspace-browser-connect",
        )
        browser_tunnel_ready = tunnel_code == 0
        if container_code == 0:
            inspect_code, inspect_text = self._wsl(
                *podman_prefix,
                "inspect",
                "--format",
                "{{.State.Running}}",
                DEFAULT_CONTAINER_NAME,
            )
            running = inspect_code == 0 and inspect_text.strip().lower() == "true"
            create_code, create_text = self._wsl(
                *podman_prefix,
                "inspect",
                "--format",
                "{{json .Config.CreateCommand}}",
                DEFAULT_CONTAINER_NAME,
            )
            if create_code == 0 and expected_isolated_network:
                try:
                    create_command = json.loads(create_text.lstrip("\ufeff"))
                except json.JSONDecodeError:
                    create_command = []
                if isinstance(create_command, list):
                    args = [str(item) for item in create_command]
                    isolated_network_ready = bool(
                        not any(item in {"--publish", "-p"} for item in args)
                        and any(
                            item == "--network"
                            and index + 1 < len(args)
                            and args[index + 1] == expected_isolated_network
                            for index, item in enumerate(args)
                        )
                    )
            firewall_code, firewall_text = self._wsl(
                "--distribution",
                DEFAULT_DISTRO_NAME,
                "--user",
                "root",
                "--exec",
                "/usr/sbin/nft",
                "list",
                "table",
                "inet",
                "aicq_workspace",
            )
            egress_firewall_ready = bool(
                firewall_code == 0
                and len(re.findall(
                    r"ip daddr @blocked_ipv4 .*comment \"aicq-block-private-v4\"",
                    firewall_text,
                )) >= 2
                and len(re.findall(
                    r"ip6 daddr @blocked_ipv6 .*comment \"aicq-block-private-v6\"",
                    firewall_text,
                )) >= 2
                and re.search(
                    r"iifname != \"lo\" meta l4proto tcp ct state new .*comment \"aicq-block-nonloopback-inbound\"",
                    firewall_text,
                )
                and "aicq-web-projection-return" not in firewall_text
            )

        protocol = manifest.get("protocol_version")
        try:
            protocol_version = int(protocol)
        except (TypeError, ValueError):
            protocol_version = None
        broker_version = str(manifest.get("broker_version") or "")
        image_exists = image_code == 0
        container_exists = container_code == 0
        managed = self._managed_marker_exists(config)
        built = bool(protocol_version and image_exists and container_exists)
        provisioning_owned = self._provisioning_marker_exists(config)
        pristine_code = 1
        if not built and protocol_version is None and not managed and install_location_matches:
            pristine_code, _ = self._wsl(
                "--distribution", DEFAULT_DISTRO_NAME, "--user", "root", "--exec",
                "/bin/sh", "-c",
                "[ ! -e /opt/aicq-workspace ] && [ ! -e /var/lib/aicq-workspace ] && ! id aicqws >/dev/null 2>&1",
            )
        partial_install = bool(
            not built
            and not managed
            and install_location_matches
            and (provisioning_owned or (protocol_version is None and pristine_code == 0))
        )
        resumable_code = 1
        if partial_install and provisioning_owned and protocol_version is not None and installed_resources:
            resumable_code, _ = self._wsl(
                "--distribution", DEFAULT_DISTRO_NAME, "--user", "root", "--exec",
                "/bin/sh", "-c",
                "test -x /opt/aicq-workspace/provision-container.sh "
                "&& test -f /etc/aicq-workspace-config.json "
                "&& id aicqws >/dev/null 2>&1",
            )
        partial_repair_mode = (
            "resume" if partial_install and resumable_code == 0
            else "recreate" if partial_install
            else ""
        )
        pending: list[str] = []
        if installed_resources:
            requested = {
                "cpus": config.cpus,
                "memory_gib": config.memory_gib,
                "disk_gib": config.disk_gib,
            }
            pending = [name for name, value in requested.items() if installed_resources.get(name) != value]
        if built and not managed:
            pending.append("ownership_marker")
        if built and not isolated_network_ready:
            pending.append("isolated_network")
        if built and not egress_firewall_ready:
            pending.append("egress_firewall")
        if built and not browser_tunnel_ready:
            pending.append("browser_tunnel")
        version_matches = bool(
            protocol_version == PROTOCOL_VERSION
            and broker_version == str(expected_manifest.get("broker_version") or "")
            and str(manifest.get("base_image_digest") or "")
            == str(expected_manifest.get("base_image_digest") or "")
            and str(manifest.get("image_name") or "")
            == str(expected_manifest.get("image_name") or "")
        )
        if not install_location_matches:
            state = "failed"
        elif partial_install:
            state = "not_built"
        elif not version_matches or (built and (not isolated_network_ready or not browser_tunnel_ready)):
            state = "needs_upgrade"
        elif not built:
            state = "not_built"
        elif pending:
            state = "needs_apply"
        else:
            state = "ready"
        return WorkspaceObservedState(
            state=state,
            built=built,
            path_locked=True,
            distro_exists=True,
            image_exists=image_exists,
            container_exists=container_exists,
            container_running=running,
            managed=managed,
            partial_install=partial_install,
            partial_repair_mode=partial_repair_mode,
            actual_install_location=actual_install_location,
            install_location_matches=install_location_matches,
            protocol_version=protocol_version,
            broker_version=broker_version,
            installed_resources=installed_resources,
            pending_changes=pending,
            isolated_network_ready=isolated_network_ready,
            egress_firewall_ready=egress_firewall_ready,
            browser_tunnel_ready=browser_tunnel_ready,
            error=(
                (
                    "检测到上次构建留下的受管半成品；再次构建会复用已完成的 appliance 并从失败阶段继续。"
                    if partial_repair_mode == "resume"
                    else "检测到上次首次构建留下的可安全恢复安装；再次构建会安全清理该半成品并继续。"
                )
                if partial_install
                else (
                    f"同名 WSL 发行版位于 {actual_install_location or '未知位置'}，与当前配置不一致；已拒绝自动处理。"
                    if not install_location_matches
                    else ("" if manifest_code == 0 else manifest_text)
                )
            ),
        )

    def status_payload(
        self,
        config: WorkspaceProvisionConfig,
        *,
        observed: WorkspaceObservedState | None = None,
        job: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        observed = observed or self.probe(config)
        job = job if job is not None else self.current_job()
        if (
            job
            and job.get("status") == "waiting_reboot"
            and not job.get("resumed_after_reboot")
            and float(job.get("boot_time") or 0) + 30 < psutil.boot_time()
        ):
            self._resume_after_reboot(job)
            job = self.current_job()
        state = observed.state
        if job:
            job_state = str(job.get("status") or "")
            if job_state not in {"ready", ""}:
                state = job_state
        return {
            "ok": True,
            "config": config.to_public_dict(),
            "observed": observed.to_dict(),
            "state": state,
            "job": job,
        }

    def describe_actions(
        self,
        config: WorkspaceProvisionConfig,
        *,
        observed: WorkspaceObservedState | None = None,
        current_job: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Describe workspace jobs using the same guards enforced by start_job."""
        observed = observed or self.probe(config)
        current_job = current_job if current_job is not None else self.current_job()
        busy = bool(
            current_job
            and str(current_job.get("status") or "") not in TERMINAL_JOB_STATES | {""}
        )
        install_target = str(Path(config.install_root) / DEFAULT_DISTRO_NAME)

        metadata = {
            "build": {
                "label": "构建工作区",
                "danger": "medium",
                "target": install_target,
                "summary": "创建受管 WSL 发行版、容器镜像与默认隔离工作区。",
                "effects": ["下载并安装工作区 appliance", "创建专用 WSL 发行版与默认容器", "应用当前 CPU、内存与磁盘配置"],
                "preserves": ["AIcarus 主数据库", "宿主机其他 WSL 发行版", "现有配置文件"],
            },
            "apply": {
                "label": "应用工作区配置",
                "danger": "medium",
                "target": install_target,
                "summary": "升级受管组件并应用资源配置；磁盘容量只允许扩容。",
                "effects": ["更新工作区 appliance 与协议组件", "应用 CPU、内存和磁盘扩容", "重启并验证工作区服务"],
                "preserves": ["工作区 /workspace 数据", "AIcarus 主数据库", "宿主机其他 WSL 发行版"],
            },
            "restart": {
                "label": "重启工作区",
                "danger": "medium",
                "target": DEFAULT_DISTRO_NAME,
                "summary": "终止并重新启动专用 WSL 发行版，然后执行健康检查。",
                "effects": ["中断当前工作区任务", "重启专用 WSL 发行版", "重新验证 broker 与隔离运行环境"],
                "preserves": ["工作区文件", "容器镜像", "资源配置"],
            },
            "clear": {
                "label": "清空工作区数据",
                "danger": "high",
                "target": "/var/lib/aicq-workspace/workspace",
                "summary": "停止默认容器并删除隔离工作目录中的全部用户文件。",
                "effects": ["停止默认工作区容器", "删除 /workspace 中的全部文件", "保留并重新验证工作区基础环境"],
                "preserves": ["WSL 发行版", "容器镜像", "工作区配置", "AIcarus 主数据库"],
            },
            "uninstall": {
                "label": "完全卸载工作区",
                "danger": "critical",
                "target": install_target,
                "summary": "注销受管 WSL 发行版并删除其专用安装目录。",
                "effects": ["终止并注销 AICQ-Workspace 发行版", "删除受管安装目录及其中的工作区数据", "解除安装路径锁定"],
                "preserves": ["工作区父目录", "AIcarus 配置", "AIcarus 主数据库", "宿主机其他 WSL 发行版"],
            },
        }

        def unavailable_reason(action: str) -> str:
            if busy:
                return "另一个工作区任务正在执行"
            if action == "build":
                if observed.distro_exists and not observed.install_location_matches:
                    return "同名 WSL 发行版的安装位置与当前配置不一致"
                if observed.distro_exists and not observed.partial_install:
                    return "工作区已经存在；请改用应用配置"
                return ""
            if not observed.distro_exists:
                return "工作区尚未构建"
            if action == "apply" and observed.partial_install:
                return "首次构建尚未完成；请继续构建"
            if action == "uninstall" and not observed.managed:
                return "工作区缺少受管所有权标记"
            if action == "apply" and observed.installed_resources:
                if config.disk_gib < observed.installed_resources.get("disk_gib", 0):
                    return "磁盘配置小于已安装容量；工作区只支持扩容"
            return ""

        actions: list[dict[str, Any]] = []
        for action in ("build", "apply", "restart", "clear", "uninstall"):
            details = metadata[action]
            disabled_reason = unavailable_reason(action)
            confirmation = ACTION_CONFIRMATIONS.get(action, "")
            actions.append({
                "id": action,
                "domain": "workspace",
                **details,
                "available": not disabled_reason,
                "disabled_reason": disabled_reason,
                "confirmation": confirmation,
                "expected_confirmation": confirmation,
                "confirmation_required": bool(confirmation),
                "keeps": "保留" + "、".join(details["preserves"]) + "。",
                "backup": {
                    "created": False,
                    "kind": "none",
                    "description": "工作区任务不会自动创建数据备份；清空或卸载前请自行导出需要的文件。",
                },
            })
        return actions

    def _spawn_worker(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job["job_id"])
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            process = subprocess.Popen(
                [sys.executable, str(WORKER_SCRIPT), "--job-id", job_id, "--control-root", str(self.control_root)],
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=creationflags,
            )
        except Exception:
            self.lock_path.unlink(missing_ok=True)
            raise
        job["pid"] = process.pid
        _atomic_json(self._job_path(job_id), job)
        _atomic_json(self.lock_path, {"job_id": job_id, "pid": process.pid, "created_at": job["created_at"]})
        return job

    def _resume_after_reboot(self, job: dict[str, Any]) -> None:
        if self._load_lock():
            return
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            return
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"job_id": job["job_id"], "pid": os.getpid(), "created_at": job["created_at"]}, handle)
        job.update(
            status=ACTION_STATES[str(job["action"])],
            stage="resuming_after_reboot",
            started_at=None,
            finished_at=None,
            pid=None,
            exit_code=None,
            error="",
            boot_time=psutil.boot_time(),
            resumed_after_reboot=True,
        )
        _atomic_json(self._job_path(str(job["job_id"])), job)
        self._spawn_worker(job)

    def start_job(
        self,
        action: str,
        config: WorkspaceProvisionConfig,
        *,
        confirmation: str = "",
    ) -> dict[str, Any]:
        action = str(action or "").strip().lower()
        if action not in JOB_ACTIONS:
            raise WorkspaceControlError("未知的 Agent 电脑任务")
        expected = ACTION_CONFIRMATIONS.get(action)
        if expected and confirmation != expected:
            raise WorkspaceControlError("确认字符串不匹配")
        if self._load_lock():
            raise WorkspaceControlError("另一个 Agent 电脑任务正在执行", status_code=409)

        observed = self.probe(config)
        if action == "build" and observed.distro_exists and not observed.install_location_matches:
            raise WorkspaceControlError("同名 WSL 发行版的安装位置与配置不一致，拒绝自动清理")
        if action == "build" and observed.distro_exists and not observed.partial_install:
            raise WorkspaceControlError("Agent 电脑已经存在且不是可安全恢复的首次构建半成品，请使用更新系统")
        if action in {"apply", "upgrade", "rebuild", "restart", "clear", "uninstall"} and not observed.distro_exists:
            raise WorkspaceControlError("Agent 电脑不存在或尚未构建")
        if action == "apply" and observed.partial_install:
            raise WorkspaceControlError("首次构建尚未完成，请使用修复并继续构建")
        if action == "apply" and observed.state == "needs_upgrade":
            raise WorkspaceControlError("Agent 电脑系统需要先更新；资源应用不会重建或升级系统")
        if action == "uninstall" and not observed.managed:
            raise WorkspaceControlError("Agent 电脑缺少受管所有权标记；请先更新系统")
        if action == "apply" and observed.installed_resources:
            installed_disk = observed.installed_resources.get("disk_gib", 0)
            if config.disk_gib < installed_disk:
                raise WorkspaceControlError("Agent 电脑磁盘只支持扩容；缩容需要完全卸载后重建")

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "action": action,
            "status": ACTION_STATES[action],
            "stage": "queued",
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "exit_code": None,
            "error": "",
            "boot_time": psutil.boot_time(),
            "resumed_after_reboot": False,
            "repair_partial_install": bool(action == "build" and observed.partial_install),
            "resume_partial_install": bool(
                action == "build" and observed.partial_repair_mode == "resume"
            ),
            "config": config.to_public_dict(),
        }
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(self._job_path(job_id), job)
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as exc:
            raise WorkspaceControlError("另一个 Agent 电脑任务正在执行", status_code=409) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            # Keep the lock owned while the detached child is being spawned.
            json.dump({"job_id": job_id, "pid": os.getpid(), "created_at": _utc_now()}, handle)

        return self._spawn_worker(job)


def workspace_control_busy() -> bool:
    return WorkspaceControlPlane().is_busy()


def detect_workspace_presence() -> str:
    """Return ``present``, ``absent`` or ``unknown`` without starting WSL."""

    if os.name != "nt":
        return "unknown"
    try:
        import winreg

        try:
            root = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Lxss",
            )
        except FileNotFoundError:
            return "absent"
        except OSError:
            return "unknown"
        uncertain = False
        with root:
            index = 0
            while True:
                try:
                    child_name = winreg.EnumKey(root, index)
                except OSError as exc:
                    # ERROR_NO_MORE_ITEMS is the normal end of enumeration.
                    if getattr(exc, "winerror", None) not in {None, 259}:
                        uncertain = True
                    break
                index += 1
                try:
                    with winreg.OpenKey(root, child_name) as child:
                        name = str(winreg.QueryValueEx(child, "DistributionName")[0])
                except OSError:
                    uncertain = True
                    continue
                if name.casefold() == DEFAULT_DISTRO_NAME.casefold():
                    return "present"
        return "unknown" if uncertain else "absent"
    except Exception:
        return "unknown"


def require_workspace_runtime_ready() -> None:
    """Reject runtime use until a user-owned provisioning job installed this build."""

    import app_state

    from .errors import WorkspaceError, WorkspaceErrorCode

    root_config = app_state.config
    if not isinstance(root_config.get("workspace") if isinstance(root_config, dict) else None, Mapping):
        from config_loader import load_config

        root_config, _prompt_docs = load_config()
    try:
        config = WorkspaceProvisionConfig.from_root_config(root_config, environ=os.environ)
    except ValueError as exc:
        raise WorkspaceError(WorkspaceErrorCode.WORKSPACE_NOT_BUILT, str(exc)) from exc
    marker_path = Path(config.install_root) / DEFAULT_DISTRO_NAME / MANAGED_MARKER
    marker = _read_json(marker_path)
    expected = _read_json(SOURCE_MANIFEST_PATH) or {}
    if marker:
        expected_location = ntpath.normcase(
            ntpath.normpath(str(Path(config.install_root) / DEFAULT_DISTRO_NAME))
        )
        marked_location = ntpath.normcase(ntpath.normpath(str(marker.get("install_location") or "")))
        try:
            protocol_matches = int(marker.get("protocol_version") or 0) == int(
                expected.get("protocol_version") or PROTOCOL_VERSION
            )
        except (TypeError, ValueError):
            protocol_matches = False
        if (
            marker.get("distro_name") == DEFAULT_DISTRO_NAME
            and marked_location == expected_location
            and protocol_matches
            and str(marker.get("broker_version") or "") == str(expected.get("broker_version") or "")
        ):
            return
        raise WorkspaceError(
            WorkspaceErrorCode.WORKSPACE_NEEDS_UPGRADE,
            "Agent 电脑系统与当前程序不兼容，请前往 Web 配置中的“Agent 电脑”页面更新系统。",
        )

    names, _error = WorkspaceControlPlane()._distro_names()
    if DEFAULT_DISTRO_NAME.casefold() in {name.casefold() for name in names}:
        observed = WorkspaceControlPlane().probe(config)
        if observed.partial_install or not observed.install_location_matches:
            raise WorkspaceError(
                WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
                "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
            )
        raise WorkspaceError(
            WorkspaceErrorCode.WORKSPACE_NEEDS_UPGRADE,
            "现有 Agent 电脑尚未同步到受管版本，请前往 Web 配置中的“Agent 电脑”页面更新系统。",
        )
    raise WorkspaceError(
        WorkspaceErrorCode.WORKSPACE_NOT_BUILT,
        "Agent 电脑不存在或尚未安装，请前往 Web 配置中的“Agent 电脑”页面完成安装。",
    )


def execute_job(job_id: str, *, control_root: Path = CONTROL_ROOT) -> int:
    """Worker entry point. It owns only the user-started provisioning/maintenance path."""

    control = WorkspaceControlPlane(control_root=control_root)
    job_path = control._job_path(job_id)
    job = _read_json(job_path)
    if not job:
        return 2
    log_path = control._log_path(job_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    job.update(pid=os.getpid(), started_at=_utc_now(), stage="starting")
    _atomic_json(job_path, job)
    _atomic_json(control.lock_path, {"job_id": job_id, "pid": os.getpid(), "created_at": job["created_at"]})

    action = str(job.get("action") or "")
    cfg = job.get("config") if isinstance(job.get("config"), dict) else {}
    resources = cfg.get("resources") if isinstance(cfg.get("resources"), dict) else {}
    install_root = str(cfg.get("install_root") or "")
    if action in {"build", "upgrade", "rebuild"}:
        argv = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PROVISION_SCRIPT),
            "-InstallRoot", install_root,
            "-Cpus", str(resources.get("cpus", 4)),
            "-MemoryGiB", str(resources.get("memory_gib", 8)),
            "-DiskGiB", str(resources.get("disk_gib", 64)),
        ]
        if action == "build" and job.get("resume_partial_install"):
            argv.append("-Resume")
        elif action == "build":
            argv.append("-Recreate")
        elif action == "rebuild":
            argv.append("-RebuildSystem")
    elif action == "apply":
        argv = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(APPLY_RESOURCES_SCRIPT),
            "-InstallRoot", install_root,
            "-Cpus", str(resources.get("cpus", 4)),
            "-MemoryGiB", str(resources.get("memory_gib", 8)),
            "-DiskGiB", str(resources.get("disk_gib", 64)),
        ]
    else:
        argv = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(MAINTENANCE_SCRIPT),
            "-Action", action.capitalize(), "-InstallRoot", install_root,
        ]

    return_code = 1
    try:
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"[{_utc_now()}] computer job {action} started\n")
            log.flush()
            job["stage"] = "running"
            _atomic_json(job_path, job)
            process = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            assert process.stdout is not None
            for line in _iter_decoded_output_lines(_read_process_chunks(process.stdout)):
                log.write(line)
                log.flush()
                if line.startswith("[computer][stage] "):
                    stage = line.removeprefix("[computer][stage] ").strip()
                    if stage:
                        job["stage"] = stage
                        _atomic_json(job_path, job)
            return_code = int(process.wait())
            log.write(f"[{_utc_now()}] computer job exited with code {return_code}\n")
        if return_code == 3010:
            job.update(status="waiting_reboot", stage="waiting_reboot", exit_code=return_code, finished_at=_utc_now())
        elif return_code == 0:
            job.update(status="ready", stage="completed", exit_code=0, finished_at=_utc_now())
        else:
            job.update(
                status="failed", stage="failed", exit_code=return_code,
                error=f"Agent 电脑任务失败，退出码 {return_code}", finished_at=_utc_now(),
            )
    except Exception as exc:
        job.update(status="failed", stage="failed", error=str(exc), exit_code=return_code, finished_at=_utc_now())
    finally:
        _atomic_json(job_path, job)
        lock = _read_json(control.lock_path)
        if lock and lock.get("job_id") == job_id:
            control.lock_path.unlink(missing_ok=True)
    return 0 if job.get("status") in {"ready", "waiting_reboot"} else 1


__all__ = [
    "ACTION_CONFIRMATIONS",
    "WorkspaceControlError",
    "WorkspaceControlPlane",
    "WorkspaceObservedState",
    "execute_job",
    "require_workspace_runtime_ready",
    "detect_workspace_presence",
    "workspace_control_busy",
]
