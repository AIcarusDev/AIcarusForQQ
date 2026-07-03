"""Small foreground supervisor for run.py.

It keeps the same console attached to the child process.  A normal exit stops
the supervisor; RESTART_EXIT_CODE means run.py requested a graceful relaunch.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from runtime.core_restart import RESTART_EXIT_CODE
except Exception:
    RESTART_EXIT_CODE = 75

RESTART_DELAY_SECONDS = float(os.environ.get("AICQ_CORE_RESTART_DELAY_SECONDS", "1"))
MAX_RESTARTS_PER_HOUR = int(os.environ.get("AICQ_CORE_MAX_RESTARTS_PER_HOUR", "6"))
CTRL_C_GRACE_SECONDS = float(os.environ.get("AICQ_CORE_CTRL_C_GRACE_SECONDS", "30"))


def _prune_recent_restart_times(restart_times: list[float], now: float) -> None:
    cutoff = now - 3600
    restart_times[:] = [ts for ts in restart_times if ts >= cutoff]


def _stop_child(proc: subprocess.Popen, *, graceful_timeout: float = 0.0) -> int:
    if proc.poll() is not None:
        return int(proc.returncode or 0)
    if graceful_timeout > 0:
        try:
            return int(proc.wait(timeout=graceful_timeout) or 0)
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            print(
                "[core-supervisor] second Ctrl+C received; terminating child...",
                flush=True,
            )
    proc.terminate()
    try:
        return int(proc.wait(timeout=10) or 0)
    except subprocess.TimeoutExpired:
        proc.kill()
        return int(proc.wait() or 1)


def main() -> int:
    run_py = BASE_DIR / "run.py"
    child_argv = [sys.executable, str(run_py)]
    child_env = os.environ.copy()
    child_env["AICQ_CORE_SUPERVISED"] = "1"
    restart_times: list[float] = []

    while True:
        print(f"[core-supervisor] launching: {' '.join(child_argv)}", flush=True)
        proc = subprocess.Popen(child_argv, cwd=str(BASE_DIR), env=child_env)
        try:
            return_code = int(proc.wait() or 0)
        except KeyboardInterrupt:
            print(
                "[core-supervisor] Ctrl+C received; waiting for child shutdown...",
                flush=True,
            )
            return _stop_child(proc, graceful_timeout=CTRL_C_GRACE_SECONDS)

        if return_code != RESTART_EXIT_CODE:
            print(f"[core-supervisor] child exited with code {return_code}", flush=True)
            return return_code

        now = time.time()
        _prune_recent_restart_times(restart_times, now)
        if len(restart_times) >= MAX_RESTARTS_PER_HOUR:
            print(
                "[core-supervisor] restart fuse opened: "
                f"{len(restart_times)} restarts in the last hour",
                flush=True,
            )
            return return_code
        restart_times.append(now)
        print(
            "[core-supervisor] restart requested; relaunching "
            f"in {RESTART_DELAY_SECONDS:.1f}s...",
            flush=True,
        )
        time.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
