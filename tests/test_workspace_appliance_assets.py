from pathlib import Path


APPLIANCE_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "workspace" / "appliance"
LINUX_TEXT_SUFFIXES = {".conf", ".json", ".py", ".service", ".sh"}


def _linux_text_assets() -> list[Path]:
    return sorted(
        path
        for path in APPLIANCE_ROOT.rglob("*")
        if path.is_file()
        and (path.name == "Containerfile" or path.suffix in LINUX_TEXT_SUFFIXES)
    )


def test_linux_appliance_assets_use_lf_line_endings():
    invalid = [
        str(path.relative_to(APPLIANCE_ROOT))
        for path in _linux_text_assets()
        if b"\r" in path.read_bytes()
    ]

    assert invalid == []


def test_resource_limit_repair_timer_is_installed():
    bootstrap = (APPLIANCE_ROOT / "bootstrap.sh").read_text(encoding="utf-8")
    apply_limits = (
        APPLIANCE_ROOT / "usr/local/lib/aicq-workspace/apply-resource-limits.sh"
    ).read_text(encoding="utf-8")
    service = (
        APPLIANCE_ROOT
        / "etc/systemd/system/aicq-workspace-resource-limits.service"
    ).read_text(encoding="utf-8")
    timer = (
        APPLIANCE_ROOT
        / "etc/systemd/system/aicq-workspace-resource-limits.timer"
    ).read_text(encoding="utf-8")

    assert 'mode=${1:---apply}' in apply_limits
    assert 'if [[ "$mode" == --ensure ]] && limits_match; then' in apply_limits
    assert "Resource limits do not match after applying them" in apply_limits
    assert "apply-resource-limits.sh --ensure" in service
    assert "OnBootSec=30s" in timer
    assert "OnUnitActiveSec=60s" in timer
    assert "WantedBy=timers.target" in timer
    assert "systemctl enable aicq-workspace-resource-limits.timer" in bootstrap
