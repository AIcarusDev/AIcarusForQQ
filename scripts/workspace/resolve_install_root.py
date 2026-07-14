"""Resolve the machine-local workspace install root from AICQ configuration."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workspace.config import WorkspaceProvisionConfig  # noqa: E402


def _default_config_path() -> Path:
    user_config = REPO_ROOT / "config" / "config_user.yaml"
    if user_config.is_file():
        return user_config
    return REPO_ROOT / "templates" / "config.yaml.template"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config_path = (args.config or _default_config_path()).resolve()
    if not config_path.is_file():
        parser.error(f"configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        root_config = yaml.safe_load(handle) or {}
    if not isinstance(root_config, dict):
        parser.error("configuration root must be a mapping")

    try:
        provision = WorkspaceProvisionConfig.from_root_config(
            root_config,
            environ=os.environ,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(provision.install_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
