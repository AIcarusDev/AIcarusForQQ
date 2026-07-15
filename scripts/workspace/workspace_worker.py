"""Detached entry point for one user-authorized workspace control job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from workspace.control import execute_job  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args()
    return execute_job(args.job_id, control_root=args.control_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
