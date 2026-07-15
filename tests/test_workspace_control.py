from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from workspace.config import WorkspaceProvisionConfig, normalize_workspace_config_inplace
from workspace.control import WorkspaceControlError, WorkspaceControlPlane
from config_loader import save_config, save_workspace_config


def workspace_config(**overrides) -> WorkspaceProvisionConfig:
    values = {
        "install_root": "E:\\Aic_forQ\\workspace-data",
        "enabled": True,
        "cpus": 4,
        "memory_gib": 8,
        "disk_gib": 64,
    }
    values.update(overrides)
    return WorkspaceProvisionConfig(**values)


class ProbeControl(WorkspaceControlPlane):
    def __init__(self, root: Path, *, distro: bool = True, protocol: int = 2, resources=None):
        super().__init__(control_root=root)
        self.distro = distro
        self.protocol = protocol
        self.resources = resources or {"cpus": 4, "memory_gib": 8, "disk_gib": 64}

    def _distro_names(self):
        return (["AICQ-Workspace"] if self.distro else []), ""

    def _managed_marker_exists(self, config):
        return True

    def _wsl(self, *args: str, timeout: float = 20.0):
        joined = " ".join(args)
        if "protocol-manifest.json" in joined:
            return 0, json.dumps({
                "protocol_version": self.protocol,
                "broker_version": "0.3.0" if self.protocol == 2 else "0.1.0",
                "image_name": f"localhost/aicq-workspace-dev:{self.protocol}",
                "base_image_digest": "sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90",
            })
        if "aicq-workspace-config.json" in joined:
            return 0, json.dumps(self.resources)
        if "image exists" in joined or "container exists" in joined:
            return 0, ""
        if "inspect --format" in joined:
            return 0, "true"
        return 1, "unexpected fake command"


def test_workspace_config_migrates_legacy_shape_to_one_canonical_block() -> None:
    root = {"workspace": {"provisioning": {"install_root": "E:\\Aic_forQ\\wsl"}}}
    normalized = normalize_workspace_config_inplace(root, environ={})
    assert normalized.install_root == "E:\\Aic_forQ\\wsl"
    assert root["workspace"] == {
        "enabled": False,
        "install_root": "E:\\Aic_forQ\\wsl",
        "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
    }
    assert "provisioning" not in root["workspace"]


def test_workspace_and_general_config_saves_do_not_overwrite_each_other(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    original = {
        "model": "old-model",
        "workspace": {
            "enabled": False,
            "install_root": "E:\\Aic_forQ\\workspace-data",
            "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 64},
        },
    }
    save_config(original, str(path), preserve_latest_workspace=False)
    workspace = {
        "enabled": True,
        "install_root": "E:\\Aic_forQ\\workspace-data",
        "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 96},
    }
    merged = save_workspace_config(workspace, base_config=original, config_path=str(path))
    assert merged["workspace"]["enabled"] is True

    stale_general = {**original, "model": "new-model"}
    save_config(stale_general, str(path))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert loaded["model"] == "new-model"
    assert loaded["workspace"] == workspace


def test_probe_distinguishes_upgrade_pending_resources_and_ready(tmp_path: Path) -> None:
    old = ProbeControl(tmp_path / "old", protocol=1)
    assert old.probe(workspace_config()).state == "needs_upgrade"

    pending = ProbeControl(
        tmp_path / "pending",
        resources={"cpus": 2, "memory_gib": 4, "disk_gib": 64},
    ).probe(workspace_config())
    assert pending.state == "needs_apply"
    assert pending.pending_changes == ["cpus", "memory_gib"]

    ready = ProbeControl(tmp_path / "ready").probe(workspace_config())
    assert ready.state == "ready"
    assert ready.built is True
    assert ready.container_running is True


def test_probe_reports_not_built_without_starting_provisioning(tmp_path: Path) -> None:
    control = ProbeControl(tmp_path, distro=False)
    observed = control.probe(workspace_config())
    assert observed.state == "not_built"
    assert observed.path_locked is False
    assert observed.distro_exists is False


def test_control_job_requires_confirmation_and_is_process_persistent(tmp_path: Path, monkeypatch) -> None:
    control = ProbeControl(tmp_path, distro=True)

    class Process:
        pid = 424242

    monkeypatch.setattr("workspace.control.subprocess.Popen", lambda *args, **kwargs: Process())
    with pytest.raises(WorkspaceControlError, match="确认字符串"):
        control.start_job("clear", workspace_config(), confirmation="wrong")

    job = control.start_job("clear", workspace_config(), confirmation="CLEAR WORKSPACE")
    assert job["status"] == "clearing"
    assert control.lock_path.is_file()
    stored = json.loads((control.jobs_root / f"{job['job_id']}.json").read_text(encoding="utf-8"))
    assert stored["pid"] == 424242


def test_stale_worker_lock_is_reconciled_to_failed(tmp_path: Path) -> None:
    control = ProbeControl(tmp_path, distro=False)
    control.jobs_root.mkdir(parents=True)
    job_id = "stale-job"
    (control.jobs_root / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id,
        "action": "build",
        "status": "building",
        "stage": "running",
    }), encoding="utf-8")
    control.lock_path.write_text(json.dumps({"job_id": job_id, "pid": 2**30}), encoding="utf-8")
    job = control.current_job()
    assert job["status"] == "failed"
    assert job["stage"] == "worker_exited"
    assert not control.lock_path.exists()


def test_waiting_reboot_job_resumes_after_boot_changes(tmp_path: Path, monkeypatch) -> None:
    control = ProbeControl(tmp_path, distro=False)
    control.jobs_root.mkdir(parents=True)
    job_id = "reboot-job"
    (control.jobs_root / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id,
        "action": "build",
        "status": "waiting_reboot",
        "stage": "waiting_reboot",
        "created_at": "2026-01-01T00:00:00+00:00",
        "boot_time": 100.0,
        "resumed_after_reboot": False,
    }), encoding="utf-8")
    monkeypatch.setattr("workspace.control.psutil.boot_time", lambda: 1000.0)
    spawned = []
    monkeypatch.setattr(control, "_spawn_worker", lambda job: spawned.append(dict(job)) or job)
    payload = control.status_payload(workspace_config())
    assert spawned and spawned[0]["stage"] == "resuming_after_reboot"
    assert payload["job"]["resumed_after_reboot"] is True
    assert payload["state"] == "building"


def test_apply_rejects_disk_shrink(tmp_path: Path) -> None:
    control = ProbeControl(
        tmp_path,
        distro=True,
        resources={"cpus": 4, "memory_gib": 8, "disk_gib": 128},
    )
    with pytest.raises(WorkspaceControlError, match="只支持扩容"):
        control.start_job("apply", workspace_config(disk_gib=64))
