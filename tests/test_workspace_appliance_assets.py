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


def test_appliance_tree_contains_no_python_cache_artifacts():
    cache_artifacts = [
        str(path.relative_to(APPLIANCE_ROOT))
        for path in APPLIANCE_ROOT.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]

    assert cache_artifacts == []
