#!/usr/bin/python3
"""Bridge stdio to one loopback TCP service inside the Agent container."""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
import threading


HANDSHAKE = b"AICQ-WORKSPACE-TUNNEL/1\n"


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise BrokenPipeError("browser tunnel output closed")
        view = view[written:]


def _target(argv: list[str]) -> tuple[str, int]:
    if len(argv) != 3:
        raise ValueError("browser connector requires a loopback host and port")
    host = argv[1].strip().rstrip(".").casefold()
    if host in {"localhost", "localhost.localdomain", "ip6-localhost", "loopback"} or host.endswith(
        ".localhost"
    ):
        host = "127.0.0.1"
    else:
        literal = ipaddress.ip_address(host)
        if not literal.is_loopback:
            raise ValueError("browser connector target must be loopback")
        host = str(literal)
    port = int(argv[2])
    if port < 1 or port > 65535:
        raise ValueError("browser connector port must be within 1..65535")
    return host, port


def main() -> int:
    try:
        host, port = _target(sys.argv)
        connection = socket.create_connection((host, port), timeout=10.0)
    except (OSError, ValueError) as exc:
        print(f"Agent localhost connection failed: {exc}", file=sys.stderr, flush=True)
        return 69

    connection.settimeout(None)
    _write_all(sys.stdout.fileno(), HANDSHAKE)
    input_done = threading.Event()

    def forward_input() -> None:
        try:
            while True:
                chunk = os.read(sys.stdin.fileno(), 65536)
                if not chunk:
                    break
                connection.sendall(chunk)
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            input_done.set()
            try:
                connection.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    worker = threading.Thread(target=forward_input, name="browser-tunnel-input", daemon=True)
    worker.start()
    try:
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            _write_all(sys.stdout.fileno(), chunk)
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        connection.close()
        worker.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
