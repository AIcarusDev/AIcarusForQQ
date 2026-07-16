#!/usr/bin/python3
"""Run one broker-created command file in its own process group."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path("/run/aicq-workspace/commands")


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"[0-9a-f]{32}", sys.argv[1]):
        return 64
    command_dir = ROOT / sys.argv[1]
    command_path = command_dir / "command.sh"
    stdin_path = command_dir / "stdin.bin"
    meta_path = command_dir / "meta.json"
    pid_path = command_dir / "pid"
    if not command_path.is_file() or not stdin_path.is_file() or not meta_path.is_file():
        return 66
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cwd = str(meta["cwd"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 65
    if not cwd.startswith("/") or "\x00" in cwd or "\\" in cwd:
        return 65

    child = os.fork()
    if child:
        _, status = os.waitpid(child, 0)
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        if os.WIFSIGNALED(status):
            return 128 + os.WTERMSIG(status)
        return 1

    os.setsid()
    pid_path.write_text(str(os.getpid()), encoding="ascii")
    try:
        os.chdir(cwd)
    except OSError as exc:
        os.write(2, f"computer cwd unavailable: {exc}\n".encode("utf-8", errors="replace"))
        os._exit(72)
    stdin_fd = os.open(stdin_path, os.O_RDONLY)
    os.dup2(stdin_fd, 0)
    os.close(stdin_fd)
    os.execv("/bin/bash", ["/bin/bash", str(command_path)])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
