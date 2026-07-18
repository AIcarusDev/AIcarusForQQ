#!/usr/bin/python3
"""Open one validated TCP stream in the Agent container over stdio."""

from __future__ import annotations

import ipaddress
import os
import subprocess
import sys


PODMAN = "/usr/bin/podman"
CONTAINER_NAME = "aicq-workspace-default"
CONTAINER_CONNECTOR = "/usr/local/bin/aicq-browser-connect"


def _target(argv: list[str]) -> tuple[str, int]:
    if len(argv) != 3:
        raise ValueError("workspace browser bridge requires a loopback host and port")
    host = argv[1].strip().rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain", "ip6-localhost", "loopback"} or host.endswith(
        ".localhost"
    ):
        host = "127.0.0.1"
    else:
        literal = ipaddress.ip_address(host)
        if not literal.is_loopback:
            raise ValueError("workspace browser bridge target must be loopback")
        host = str(literal)
    port = int(argv[2])
    if port < 1 or port > 65535:
        raise ValueError("workspace browser bridge port must be within 1..65535")
    return host, port


def _podman(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [PODMAN, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def main() -> int:
    try:
        host, port = _target(sys.argv)
    except (ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 64

    exists = _podman("container", "exists", CONTAINER_NAME)
    if exists.returncode != 0:
        print("Agent workspace is not built", file=sys.stderr, flush=True)
        return 69
    running = _podman("inspect", "--format", "{{.State.Running}}", CONTAINER_NAME)
    if running.returncode != 0:
        print("Agent workspace state could not be inspected", file=sys.stderr, flush=True)
        return 69
    if running.stdout.strip().lower() != b"true":
        started = _podman("start", CONTAINER_NAME)
        if started.returncode != 0:
            diagnostic = started.stderr.decode("utf-8", errors="replace").strip()
            print(diagnostic or "Agent workspace could not be started", file=sys.stderr, flush=True)
            return 69

    argv = [
        PODMAN,
        "exec",
        "--interactive",
        "--user",
        "agent",
        "--workdir",
        "/home/agent",
        CONTAINER_NAME,
        CONTAINER_CONNECTOR,
        host,
        str(port),
    ]
    os.execv(PODMAN, argv)
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
