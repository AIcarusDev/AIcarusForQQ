from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from workspace.config import WorkspaceProvisionConfig, normalize_workspace_config_inplace
from workspace import control as control_module
from workspace.control import (
    WorkspaceControlError,
    WorkspaceControlPlane,
    consume_workspace_directory_selection,
    execute_job,
    publish_workspace_directory_selection,
)
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


def test_workspace_directory_selection_handoff_is_one_shot(tmp_path: Path) -> None:
    publish_workspace_directory_selection(
        "selection-1",
        status="selected",
        path="E:\\Aic_forQ\\workspace-new",
        control_root=tmp_path,
    )

    selected = consume_workspace_directory_selection("selection-1", control_root=tmp_path)
    assert selected is not None
    assert selected["status"] == "selected"
    assert selected["path"] == "E:\\Aic_forQ\\workspace-new"
    assert consume_workspace_directory_selection("selection-1", control_root=tmp_path) is None

    with pytest.raises(WorkspaceControlError, match="目录选择 ID"):
        consume_workspace_directory_selection("../outside", control_root=tmp_path)


class ProbeControl(WorkspaceControlPlane):
    def __init__(
        self,
        root: Path,
        *,
        distro: bool = True,
        protocol: int = 5,
        resources=None,
        managed: bool = True,
        partial: bool = False,
        resumable: bool = False,
        location_matches: bool = True,
        isolated_network: bool = True,
        egress_firewall: bool = True,
        browser_tunnel: bool = True,
    ):
        super().__init__(control_root=root)
        self.distro = distro
        self.protocol = protocol
        self.resources = resources or {"cpus": 4, "memory_gib": 8, "disk_gib": 64}
        self.managed = managed
        self.partial = partial
        self.resumable = resumable
        self.location_matches = location_matches
        self.isolated_network = isolated_network
        self.egress_firewall = egress_firewall
        self.browser_tunnel = browser_tunnel

    def _distro_names(self):
        return (["AICQ-Workspace"] if self.distro else []), ""

    def _managed_marker_exists(self, config):
        return self.managed

    def _provisioning_marker_exists(self, config):
        return self.partial

    def _distro_install_location(self):
        if self.location_matches:
            return "E:\\Aic_forQ\\workspace-data\\AICQ-Workspace"
        return "D:\\AnotherRoot\\AICQ-Workspace"

    def _wsl(self, *args: str, timeout: float = 20.0):
        joined = " ".join(args)
        if self.partial and not self.resumable:
            if "/bin/sh -c" in joined:
                return 0, ""
            return 1, "not installed"
        if "protocol-manifest.json" in joined:
            return 0, json.dumps({
                "protocol_version": self.protocol,
                "broker_version": (
                    "0.6.0" if self.protocol == 5 else "0.5.3" if self.protocol == 4 else "0.4.0"
                ),
                "image_name": f"localhost/aicq-workspace-dev:{self.protocol}",
                "base_image_digest": "sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90",
            })
        if "aicq-workspace-config.json" in joined:
            return 0, json.dumps(self.resources)
        if self.resumable and ("image exists" in joined or "container exists" in joined):
            return 1, "not built yet"
        if self.resumable and "test -x /opt/aicq-workspace/provision-container.sh" in joined:
            return 0, ""
        if "image exists" in joined or "container exists" in joined:
            return 0, ""
        if "/usr/local/bin/aicq-workspace-browser-connect" in joined:
            return (0, "") if self.browser_tunnel else (1, "missing")
        if "{{json .Config.CreateCommand}}" in joined:
            command = ["/usr/bin/podman", "create", "--network"]
            if self.isolated_network:
                command.append("slirp4netns:allow_host_loopback=false")
            else:
                command.extend(["pasta", "--publish", "127.0.0.1::6080"])
            return 0, json.dumps(command)
        if "inspect --format" in joined:
            return 0, "true"
        if "nft list table inet aicq_workspace" in joined:
            if not self.egress_firewall:
                return 0, 'meta skuid 1000 ip daddr @blocked_ipv4 counter reject comment "aicq-block-private-v4"'
            return 0, (
                'meta skuid 1000 ip daddr @blocked_ipv4 counter packets 0 bytes 0 '
                'reject comment "aicq-block-private-v4"\n'
                'meta skuid 100999 ip daddr @blocked_ipv4 counter packets 0 bytes 0 '
                'reject comment "aicq-block-private-v4"\n'
                'meta skuid 1000 ip6 daddr @blocked_ipv6 counter packets 0 bytes 0 '
                'reject comment "aicq-block-private-v6"\n'
                'meta skuid 100999 ip6 daddr @blocked_ipv6 counter packets 0 bytes 0 '
                'reject comment "aicq-block-private-v6"\n'
                'iifname != "lo" meta l4proto tcp ct state new counter packets 0 bytes 0 '
                'reject comment "aicq-block-nonloopback-inbound"'
            )
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
    old = ProbeControl(tmp_path / "old", protocol=2)
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
    assert ready.isolated_network_ready is True
    assert ready.egress_firewall_ready is True
    assert ready.browser_tunnel_ready is True


def test_probe_requires_the_isolated_agent_network(tmp_path: Path) -> None:
    observed = ProbeControl(tmp_path, isolated_network=False).probe(workspace_config())

    assert observed.state == "needs_upgrade"
    assert observed.pending_changes == ["isolated_network"]
    assert observed.isolated_network_ready is False


def test_probe_requires_the_egress_firewall_rule(tmp_path: Path) -> None:
    observed = ProbeControl(tmp_path, egress_firewall=False).probe(workspace_config())

    assert observed.state == "needs_apply"
    assert observed.pending_changes == ["egress_firewall"]
    assert observed.egress_firewall_ready is False


def test_probe_requires_the_browser_tunnel_helper(tmp_path: Path) -> None:
    observed = ProbeControl(tmp_path, browser_tunnel=False).probe(workspace_config())

    assert observed.state == "needs_upgrade"
    assert observed.pending_changes == ["browser_tunnel"]
    assert observed.browser_tunnel_ready is False


def test_probe_reports_not_built_without_starting_provisioning(tmp_path: Path) -> None:
    control = ProbeControl(tmp_path, distro=False)
    observed = control.probe(workspace_config())
    assert observed.state == "not_built"
    assert observed.path_locked is False
    assert observed.distro_exists is False


def test_probe_recognizes_repairable_partial_first_build(tmp_path: Path) -> None:
    observed = ProbeControl(
        tmp_path,
        managed=False,
        partial=True,
    ).probe(workspace_config())

    assert observed.state == "not_built"
    assert observed.partial_install is True
    assert observed.install_location_matches is True
    assert observed.built is False
    assert "可安全恢复" in observed.error


def test_partial_build_retries_as_build_but_location_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Process:
        pid = 424242

    monkeypatch.setattr("workspace.control.subprocess.Popen", lambda *args, **kwargs: Process())
    partial = ProbeControl(tmp_path / "partial", managed=False, partial=True)
    job = partial.start_job("build", workspace_config())
    assert job["repair_partial_install"] is True
    assert job["status"] == "building"

    mismatch = ProbeControl(
        tmp_path / "mismatch",
        managed=False,
        partial=True,
        location_matches=False,
    )
    with pytest.raises(WorkspaceControlError, match="安装位置与配置不一致"):
        mismatch.start_job("build", workspace_config())


def test_owned_advanced_partial_build_resumes_without_recreating_distro(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Process:
        pid = 424243

    monkeypatch.setattr("workspace.control.subprocess.Popen", lambda *args, **kwargs: Process())
    control = ProbeControl(
        tmp_path,
        managed=False,
        partial=True,
        resumable=True,
    )

    observed = control.probe(workspace_config())
    assert observed.partial_install is True
    assert observed.partial_repair_mode == "resume"
    assert observed.state == "not_built"

    job = control.start_job("build", workspace_config())
    assert job["repair_partial_install"] is True
    assert job["resume_partial_install"] is True


def test_successful_machine_readable_probe_ignores_wsl_stderr(monkeypatch) -> None:
    completed = SimpleNamespace(
        returncode=0,
        stdout=b'{"protocol_version":2}\n',
        stderr="WSL proxy warning".encode("utf-16-le"),
    )
    monkeypatch.setattr(control_module.subprocess, "run", lambda *args, **kwargs: completed)

    code, output = control_module._run_capture(["wsl.exe", "--distribution", "AICQ-Workspace"])

    assert code == 0
    assert output == '{"protocol_version":2}'


def test_mixed_utf8_and_utf16_wsl_job_output_is_decoded_line_by_line() -> None:
    raw = (
        b"[computer][stage] installing_distro\r\n"
        + "已成功安装 Ubuntu 24.04 LTS\r\n".encode("utf-16-le")
        + b"[computer] continuing\r\n"
    )

    decoded = "".join(control_module._iter_decoded_output_lines([raw[:47], raw[47:83], raw[83:]]))

    assert decoded == (
        "[computer][stage] installing_distro\n"
        "已成功安装 Ubuntu 24.04 LTS\n"
        "[computer] continuing\n"
    )


def test_worker_repair_build_passes_recreate_and_persists_stage_log(tmp_path: Path, monkeypatch) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True)
    job_id = "repair-job"
    (jobs / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id,
        "action": "build",
        "status": "building",
        "stage": "queued",
        "created_at": "2026-07-15T00:00:00+00:00",
        "config": workspace_config().to_public_dict(),
    }), encoding="utf-8")
    seen: list[list[str]] = []

    class Process:
        stdout = io.BytesIO(
            b"[computer][stage] recovering_partial_install\r\n"
            + "正在恢复\r\n".encode("utf-16-le")
            + b"[computer][stage] completed\r\n"
        )

        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        seen.append(list(argv))
        return Process()

    monkeypatch.setattr(control_module.subprocess, "Popen", fake_popen)

    assert execute_job(job_id, control_root=tmp_path) == 0
    assert "-Recreate" in seen[0]
    log = (jobs / f"{job_id}.log").read_text(encoding="utf-8")
    assert "正在恢复" in log
    assert "[computer][stage] completed" in log


def test_worker_resumable_build_passes_resume_not_recreate(tmp_path: Path, monkeypatch) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True)
    job_id = "resume-job"
    (jobs / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id,
        "action": "build",
        "status": "building",
        "stage": "queued",
        "created_at": "2026-07-15T00:00:00+00:00",
        "repair_partial_install": True,
        "resume_partial_install": True,
        "config": workspace_config().to_public_dict(),
    }), encoding="utf-8")
    seen: list[list[str]] = []

    class Process:
        stdout = io.BytesIO(b"[computer][stage] completed\r\n")

        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        seen.append(list(argv))
        return Process()

    monkeypatch.setattr(control_module.subprocess, "Popen", fake_popen)

    assert execute_job(job_id, control_root=tmp_path) == 0
    assert "-Resume" in seen[0]
    assert "-Recreate" not in seen[0]


def test_probe_accepts_utf8_bom_from_legacy_resource_config(tmp_path: Path) -> None:
    class BomProbeControl(ProbeControl):
        def _wsl(self, *args: str, timeout: float = 20.0):
            code, output = super()._wsl(*args, timeout=timeout)
            if "aicq-workspace-config.json" in " ".join(args):
                output = "\ufeff" + output
            return code, output

    observed = BomProbeControl(tmp_path).probe(workspace_config())

    assert observed.state == "ready"
    assert observed.installed_resources == {"cpus": 4, "memory_gib": 8, "disk_gib": 64}


def test_control_job_requires_confirmation_and_is_process_persistent(tmp_path: Path, monkeypatch) -> None:
    control = ProbeControl(tmp_path, distro=True)

    class Process:
        pid = 424242

    monkeypatch.setattr("workspace.control.subprocess.Popen", lambda *args, **kwargs: Process())
    with pytest.raises(WorkspaceControlError, match="确认字符串"):
        control.start_job("clear", workspace_config(), confirmation="wrong")

    job = control.start_job("clear", workspace_config(), confirmation="ERASE AGENT HOME")
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


@pytest.mark.parametrize(
    ("action", "script_name", "required_flag", "forbidden_flag"),
    [
        ("apply", "apply-workspace-resources.ps1", None, "-RebuildSystem"),
        ("upgrade", "provision-workspace.ps1", None, "-RebuildSystem"),
        ("rebuild", "provision-workspace.ps1", "-RebuildSystem", None),
    ],
)
def test_worker_dispatches_resource_apply_and_system_rebuild_separately(
    tmp_path: Path,
    monkeypatch,
    action: str,
    script_name: str,
    required_flag: str | None,
    forbidden_flag: str | None,
) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir(parents=True)
    job_id = f"{action}-job"
    (jobs / f"{job_id}.json").write_text(json.dumps({
        "job_id": job_id,
        "action": action,
        "status": "queued",
        "stage": "queued",
        "created_at": "2026-07-17T00:00:00+00:00",
        "config": workspace_config().to_public_dict(),
    }), encoding="utf-8")
    seen: list[list[str]] = []

    class Process:
        stdout = io.BytesIO(b"[computer][stage] completed\r\n")

        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        seen.append(list(argv))
        return Process()

    monkeypatch.setattr(control_module.subprocess, "Popen", fake_popen)

    assert execute_job(job_id, control_root=tmp_path) == 0
    assert Path(seen[0][5]).name == script_name
    if required_flag:
        assert required_flag in seen[0]
    if forbidden_flag:
        assert forbidden_flag not in seen[0]


@pytest.mark.parametrize(
    ("control_kwargs", "action", "config_overrides", "message"),
    [
        ({"distro": False}, "restart", {}, "尚未构建"),
        ({"distro": True}, "build", {}, "已经存在"),
        ({"distro": True, "managed": False}, "uninstall", {}, "所有权标记"),
        (
            {
                "distro": True,
                "resources": {"cpus": 4, "memory_gib": 8, "disk_gib": 128},
            },
            "apply",
            {"disk_gib": 64},
            "只支持扩容",
        ),
    ],
)
def test_workspace_action_descriptions_match_execution_guards(
    control_kwargs: dict,
    action: str,
    config_overrides: dict,
    message: str,
    tmp_path: Path,
) -> None:
    control = ProbeControl(tmp_path, **control_kwargs)
    config = workspace_config(**config_overrides)
    descriptions = {
        item["id"]: item
        for item in control.describe_actions(config)
    }

    assert descriptions[action]["available"] is False
    assert message in descriptions[action]["disabled_reason"]
    with pytest.raises(WorkspaceControlError, match=message):
        control.start_job(
            action,
            config,
            confirmation=descriptions[action]["expected_confirmation"],
        )
