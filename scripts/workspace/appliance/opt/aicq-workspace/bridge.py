#!/usr/bin/python3
"""Single-request stdio-to-Unix-socket bridge used by Core."""

from __future__ import annotations

import socket
import sys


SOCKET_PATH = "/run/aicq-workspace/broker.sock"
# A 1 MiB UTF-8 text value can expand to roughly 6 MiB when JSON escapes
# control characters.  Keep the transport bounded while honoring that limit.
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def main() -> int:
    request = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if not request or len(request) > MAX_REQUEST_BYTES or not request.endswith(b"\n"):
        print("bridge requires one bounded NDJSON request", file=sys.stderr)
        return 64
    if sys.stdin.buffer.read(1):
        print("bridge accepts exactly one request", file=sys.stderr)
        return 64

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(935.0)
            client.connect(SOCKET_PATH)
            client.sendall(request)
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    print("broker response exceeded bridge limit", file=sys.stderr)
                    return 70
                chunks.append(chunk)
                if b"\n" in chunk:
                    break
    except OSError as exc:
        print(f"workspace broker unavailable: {exc}", file=sys.stderr)
        return 69

    response = b"".join(chunks)
    if response.count(b"\n") != 1 or not response.endswith(b"\n"):
        print("broker returned an invalid NDJSON response", file=sys.stderr)
        return 70
    sys.stdout.buffer.write(response)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
