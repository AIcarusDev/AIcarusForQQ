#!/usr/bin/python3
"""Open one validated TCP stream in the Agent container over stdio."""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import subprocess
import sys
import uuid


PODMAN = "/usr/bin/podman"
CONTAINER_NAME = "aicq-workspace-default"
CONTAINER_CONNECTOR = "/usr/local/bin/aicq-browser-connect"
BROKER_SOCKET = "/run/aicq-workspace/broker.sock"
PROTOCOL_VERSION = 5
PODMAN_ENV = {
    **os.environ,
    "HOME": "/home/aicqws",
    "XDG_RUNTIME_DIR": "/run/aicq-workspace/user",
}


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
        env=PODMAN_ENV,
        check=False,
        timeout=30,
    )


def _ensure_through_broker() -> tuple[bool, str]:
    request = {
        "version": PROTOCOL_VERSION,
        "request_id": uuid.uuid4().hex,
        "method": "ensure_default",
        "params": {"workspace_id": "default"},
    }
    encoded = (json.dumps(request, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(60.0)
            client.connect(BROKER_SOCKET)
            client.sendall(encoded)
            response = client.makefile("rb").readline(1024 * 1024)
        decoded = json.loads(response.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"Agent workspace broker unavailable: {exc}"
    if not isinstance(decoded, dict) or not decoded.get("ok"):
        error = decoded.get("error", {}) if isinstance(decoded, dict) else {}
        message = error.get("message") if isinstance(error, dict) else ""
        return False, str(message or "Agent workspace could not be started")
    return True, ""


def main() -> int:
    try:
        host, port = _target(sys.argv)
    except (ValueError, TypeError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 64

    ensured, diagnostic = _ensure_through_broker()
    if not ensured:
        print(diagnostic, file=sys.stderr, flush=True)
        return 69

    exists = _podman("container", "exists", CONTAINER_NAME)
    if exists.returncode != 0:
        print("Agent workspace is not built", file=sys.stderr, flush=True)
        return 69
    running = _podman("inspect", "--format", "{{.State.Running}}", CONTAINER_NAME)
    if running.returncode != 0:
        print("Agent workspace state could not be inspected", file=sys.stderr, flush=True)
        return 69
    if running.stdout.strip().lower() != b"true":
        print("Agent workspace broker did not keep the container running", file=sys.stderr, flush=True)
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
    os.execve(PODMAN, argv, PODMAN_ENV)
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
