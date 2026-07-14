#!/usr/bin/python3
"""Rootless Podman broker for the internal AICQ Linux workspace."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import tempfile
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path("/opt/aicq-workspace")
MANIFEST_PATH = ROOT / "protocol-manifest.json"
IMAGE_CONTEXT = ROOT / "image"
STATE_ROOT = Path("/var/lib/aicq-workspace")
WORKSPACE_ROOT = STATE_ROOT / "workspace"
COMMAND_ROOT = STATE_ROOT / "commands"
SOCKET_PATH = Path("/run/aicq-workspace/broker.sock")
FIREWALL_MARKER = Path("/run/aicq-workspace/firewall.ready")
PODMAN = "/usr/bin/podman"

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_STDIN_BYTES = 1024 * 1024
MAX_TEXT_BYTES = 1024 * 1024
MAX_STORED_STREAM_BYTES = 16 * 1024 * 1024
MAX_INLINE_STREAM_BYTES = 64 * 1024
MAX_TIMEOUT_SECONDS = 900.0
DEFAULT_TIMEOUT_SECONDS = 120.0

ALLOWED_METHODS = {
    "health",
    "ensure_default",
    "exec",
    "get_command",
    "read_text",
    "write_text",
}


class RpcFailure(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")


class StreamCapture:
    def __init__(self, limit: int = MAX_STORED_STREAM_BYTES) -> None:
        self.limit = limit
        self.head_limit = limit // 2
        self.tail_limit = limit - self.head_limit
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self.tail_bytes = 0
        self.total_bytes = 0

    def add(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        remaining = chunk
        if len(self.head) < self.head_limit:
            take = min(self.head_limit - len(self.head), len(remaining))
            self.head.extend(remaining[:take])
            remaining = remaining[take:]
        if remaining:
            self.tail.append(remaining)
            self.tail_bytes += len(remaining)
            while self.tail_bytes > self.tail_limit and self.tail:
                overflow = self.tail_bytes - self.tail_limit
                first = self.tail[0]
                if len(first) <= overflow:
                    self.tail.popleft()
                    self.tail_bytes -= len(first)
                else:
                    self.tail[0] = first[overflow:]
                    self.tail_bytes -= overflow

    def payload(self) -> dict[str, Any]:
        raw = bytes(self.head) + b"".join(self.tail)
        return {
            "text": raw.decode("utf-8", errors="replace"),
            "total_bytes": self.total_bytes,
            "truncated": self.total_bytes > self.limit,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utf8_size(value: str) -> int:
    return len(value.encode("utf-8"))


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


MANIFEST = load_manifest()
PROTOCOL_VERSION = int(MANIFEST["protocol_version"])
BROKER_VERSION = str(MANIFEST["broker_version"])
WORKSPACE_ID = str(MANIFEST["workspace_id"])
CONTAINER_NAME = str(MANIFEST["container_name"])
IMAGE_NAME = str(MANIFEST["image_name"])
BASE_IMAGE_DIGEST = str(MANIFEST["base_image_digest"])


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


async def drain_stream(reader: asyncio.StreamReader, capture: StreamCapture) -> None:
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            return
        capture.add(chunk)


async def run_capture(
    argv: list[str],
    *,
    stdin: bytes = b"",
    deadline: float | None = None,
) -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stdout_capture = StreamCapture()
    stderr_capture = StreamCapture()
    stdout_task = asyncio.create_task(drain_stream(process.stdout, stdout_capture))
    stderr_task = asyncio.create_task(drain_stream(process.stderr, stderr_capture))
    process.stdin.write(stdin)
    try:
        await process.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    process.stdin.close()
    timed_out = False
    try:
        if deadline is None:
            await process.wait()
        else:
            await asyncio.wait_for(process.wait(), deadline)
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        await process.wait()
    await asyncio.gather(stdout_task, stderr_task)
    return process.returncode, stdout_capture.payload(), stderr_capture.payload(), timed_out


async def podman(
    args: list[str], *, stdin: bytes = b"", deadline: float | None = 120.0
) -> tuple[int, dict[str, Any], dict[str, Any], bool]:
    return await run_capture([PODMAN, *args], stdin=stdin, deadline=deadline)


def stream_preview(stream: dict[str, Any]) -> dict[str, Any]:
    raw = str(stream.get("text", "")).encode("utf-8", errors="replace")
    clipped = bool(stream.get("truncated", False)) or len(raw) > MAX_INLINE_STREAM_BYTES
    if len(raw) > MAX_INLINE_STREAM_BYTES:
        half = MAX_INLINE_STREAM_BYTES // 2
        raw = raw[:half] + raw[-half:]
    return {
        "text": raw.decode("utf-8", errors="replace"),
        "total_bytes": int(stream.get("total_bytes", 0) or 0),
        "truncated": clipped,
    }


def public_command(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_id": record["command_id"],
        "workspace_id": record["workspace_id"],
        "status": record["status"],
        "cwd": record["cwd"],
        "exit_code": record.get("exit_code"),
        "started_at": record["started_at"],
        "finished_at": record.get("finished_at"),
        "timed_out": bool(record.get("timed_out", False)),
        "stdout": stream_preview(record.get("stdout") or {}),
        "stderr": stream_preview(record.get("stderr") or {}),
    }


def require_default(params: dict[str, Any]) -> None:
    if str(params.get("workspace_id", WORKSPACE_ID)) != WORKSPACE_ID:
        raise RpcFailure("invalid_argument", "only workspace_id='default' is supported")


def validate_linux_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RpcFailure("invalid_argument", "path must be a non-empty Linux path")
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise RpcFailure("invalid_argument", "Windows and host paths are not accepted")
    path = PurePosixPath(value)
    if str(path) in {"", "."}:
        raise RpcFailure("invalid_argument", "path must identify a file")
    return value


async def container_exists() -> bool:
    code, _, _, _ = await podman(["container", "exists", CONTAINER_NAME], deadline=15.0)
    return code == 0


async def container_running() -> bool:
    if not await container_exists():
        return False
    code, stdout, _, _ = await podman(
        ["inspect", "--format", "{{.State.Running}}", CONTAINER_NAME], deadline=15.0
    )
    return code == 0 and stdout["text"].strip().lower() == "true"


async def image_exists() -> bool:
    code, _, _, _ = await podman(["image", "exists", IMAGE_NAME], deadline=15.0)
    return code == 0


async def inspect_image_label(label: str) -> str:
    code, stdout, stderr, _ = await podman(
        ["image", "inspect", "--format", f"{{{{ index .Labels \"{label}\" }}}}", IMAGE_NAME],
        deadline=15.0,
    )
    if code != 0:
        raise RpcFailure("container_start_failed", "could not inspect workspace image", {"stderr": stderr["text"][-2048:]})
    return stdout["text"].strip()


async def ensure_container() -> dict[str, Any]:
    if not FIREWALL_MARKER.is_file():
        raise RpcFailure(
            "container_start_failed",
            "public_egress firewall is not active; refusing to start workspace",
        )

    if not await image_exists():
        code, _, stderr, timed_out = await podman(
            [
                "build",
                "--pull=missing",
                "--label",
                f"io.aicq.workspace.protocol={PROTOCOL_VERSION}",
                "--label",
                f"io.aicq.workspace.base-digest={BASE_IMAGE_DIGEST}",
                "--tag",
                IMAGE_NAME,
                str(IMAGE_CONTEXT),
            ],
            deadline=MAX_TIMEOUT_SECONDS,
        )
        if code != 0 or timed_out:
            raise RpcFailure(
                "container_start_failed",
                "workspace development image build failed",
                {"stderr": stderr["text"][-4096:], "timed_out": timed_out},
            )

    protocol_label = await inspect_image_label("io.aicq.workspace.protocol")
    digest_label = await inspect_image_label("io.aicq.workspace.base-digest")
    if protocol_label != str(PROTOCOL_VERSION) or digest_label != BASE_IMAGE_DIGEST:
        raise RpcFailure(
            "container_start_failed",
            "workspace image labels do not match the installed protocol manifest",
            {"protocol": protocol_label, "base_digest": digest_label},
        )

    created = False
    started = False
    exists = await container_exists()
    if not exists:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        limits = MANIFEST["limits"]
        code, _, stderr, _ = await podman(
            [
                "create",
                "--name",
                CONTAINER_NAME,
                "--hostname",
                "workspace",
                "--user",
                "0:0",
                "--workdir",
                "/workspace",
                "--cpus",
                str(limits["cpus"]),
                "--memory",
                str(limits["memory_bytes"]),
                "--pids-limit",
                str(limits["pids"]),
                "--network",
                "pasta",
                "--volume",
                f"{WORKSPACE_ROOT}:/workspace:rw",
                "--stop-timeout",
                "10",
                IMAGE_NAME,
            ],
            deadline=120.0,
        )
        if code != 0:
            raise RpcFailure(
                "container_start_failed", "could not create workspace container", {"stderr": stderr["text"][-4096:]}
            )
        created = True

    if not await container_running():
        code, _, stderr, _ = await podman(["start", CONTAINER_NAME], deadline=60.0)
        if code != 0:
            raise RpcFailure(
                "container_start_failed", "could not start workspace container", {"stderr": stderr["text"][-4096:]}
            )
        started = True

    return {
        "workspace_id": WORKSPACE_ID,
        "container_name": CONTAINER_NAME,
        "created": created,
        "started": started,
        "image_digest": BASE_IMAGE_DIGEST,
        "limits": MANIFEST["limits"],
    }


class Broker:
    def __init__(self) -> None:
        self.workspace_lock = asyncio.Lock()

    async def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "health":
            return await self.health()
        require_default(params)
        if method == "get_command":
            return await self.get_command(params)
        async with self.workspace_lock:
            if method == "ensure_default":
                return await ensure_container()
            if method == "exec":
                await ensure_container()
                return await self.exec(params)
            if method == "read_text":
                await ensure_container()
                return await self.read_text(params)
            if method == "write_text":
                await ensure_container()
                return await self.write_text(params)
        raise RpcFailure("invalid_argument", f"unsupported method: {method}")

    async def health(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "broker_version": BROKER_VERSION,
            "distro": "AICQ-Workspace",
            "container_exists": await container_exists(),
            "container_running": await container_running(),
            "image_digest": BASE_IMAGE_DIGEST,
            "firewall_active": FIREWALL_MARKER.is_file(),
        }

    async def exec(self, params: dict[str, Any]) -> dict[str, Any]:
        command = params.get("command")
        stdin = params.get("stdin", "")
        cwd = validate_linux_path(params.get("cwd", "/workspace"))
        if not isinstance(command, str) or not command:
            raise RpcFailure("invalid_argument", "command must be a non-empty string")
        if utf8_size(command) > MAX_COMMAND_BYTES:
            raise RpcFailure("invalid_argument", "command exceeds the 64 KiB limit")
        if not isinstance(stdin, str) or utf8_size(stdin) > MAX_STDIN_BYTES:
            raise RpcFailure("invalid_argument", "stdin exceeds the 1 MiB limit")
        try:
            timeout_seconds = float(params.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        except (TypeError, ValueError) as exc:
            raise RpcFailure("invalid_argument", "timeout_seconds must be numeric") from exc
        if timeout_seconds <= 0 or timeout_seconds > MAX_TIMEOUT_SECONDS:
            raise RpcFailure("invalid_argument", "timeout_seconds must be in (0, 900]")

        command_id = uuid.uuid4().hex
        record_path = COMMAND_ROOT / f"{command_id}.json"
        record = {
            "command_id": command_id,
            "workspace_id": WORKSPACE_ID,
            "command": command,
            "status": "running",
            "cwd": cwd,
            "exit_code": None,
            "started_at": utc_now(),
            "finished_at": None,
            "timed_out": False,
            "stdout": {"text": "", "total_bytes": 0, "truncated": False},
            "stderr": {"text": "", "total_bytes": 0, "truncated": False},
        }
        atomic_json(record_path, record)

        code, stdout, stderr, outer_timeout = await podman(
            [
                "exec",
                "--user",
                "0",
                "--workdir",
                cwd,
                "--interactive",
                CONTAINER_NAME,
                "/usr/bin/timeout",
                "--signal=TERM",
                "--kill-after=5s",
                f"{timeout_seconds}s",
                "/bin/bash",
                "-lc",
                command,
            ],
            stdin=stdin.encode("utf-8"),
            deadline=timeout_seconds + 15.0,
        )
        timed_out = outer_timeout or code in {124, 137}
        record.update(
            {
                "status": "timed_out" if timed_out else "completed",
                "exit_code": code,
                "finished_at": utc_now(),
                "timed_out": timed_out,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        atomic_json(record_path, record)
        public = public_command(record)
        if timed_out:
            raise RpcFailure(
                "command_timeout",
                f"command exceeded {timeout_seconds:g} seconds",
                public,
            )
        return public

    async def get_command(self, params: dict[str, Any]) -> dict[str, Any]:
        command_id = params.get("command_id")
        if not isinstance(command_id, str) or not re.fullmatch(r"[0-9a-f]{32}", command_id):
            raise RpcFailure("invalid_argument", "command_id must be a broker command id")
        path = COMMAND_ROOT / f"{command_id}.json"
        try:
            with path.open("r", encoding="utf-8") as handle:
                return public_command(json.load(handle))
        except FileNotFoundError as exc:
            raise RpcFailure("path_error", "command record was not found") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RpcFailure("internal_error", "command record could not be read") from exc

    async def read_text(self, params: dict[str, Any]) -> dict[str, Any]:
        path = validate_linux_path(params.get("path"))
        script = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            f"d=p.read_bytes(); sys.exit(65) if len(d)>{MAX_TEXT_BYTES} else None; "
            "d.decode('utf-8'); sys.stdout.buffer.write(d)"
        )
        code, stdout, stderr, _ = await podman(
            ["exec", "--user", "0", "--workdir", "/workspace", CONTAINER_NAME, "python3", "-c", script, path],
            deadline=30.0,
        )
        if code == 65:
            raise RpcFailure("path_error", "text file exceeds the 1 MiB limit")
        if code != 0:
            raise RpcFailure("path_error", "text file could not be read", {"stderr": stderr["text"][-2048:]})
        content = stdout["text"]
        return {"path": path, "content": content, "size_bytes": stdout["total_bytes"]}

    async def write_text(self, params: dict[str, Any]) -> dict[str, Any]:
        path = validate_linux_path(params.get("path"))
        content = params.get("content")
        if not isinstance(content, str) or utf8_size(content) > MAX_TEXT_BYTES:
            raise RpcFailure("invalid_argument", "content exceeds the 1 MiB UTF-8 limit")
        create_parents = bool(params.get("create_parents", False))
        script = (
            "import os,pathlib,sys,tempfile; p=pathlib.Path(sys.argv[1]); "
            "parents=sys.argv[2]=='1'; p.parent.mkdir(parents=True,exist_ok=True) if parents else None; "
            "data=sys.stdin.buffer.read(); data.decode('utf-8'); "
            "fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent)); "
            "f=os.fdopen(fd,'wb'); f.write(data); f.flush(); os.fsync(f.fileno()); f.close(); os.replace(tmp,p)"
        )
        code, _, stderr, _ = await podman(
            [
                "exec",
                "--user",
                "0",
                "--workdir",
                "/workspace",
                "--interactive",
                CONTAINER_NAME,
                "python3",
                "-c",
                script,
                path,
                "1" if create_parents else "0",
            ],
            stdin=content.encode("utf-8"),
            deadline=30.0,
        )
        if code != 0:
            raise RpcFailure("path_error", "text file could not be written", {"stderr": stderr["text"][-2048:]})
        return {"path": path, "size_bytes": utf8_size(content)}


BROKER = Broker()


def response_ok(request_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"version": PROTOCOL_VERSION, "request_id": request_id, "ok": True, "result": result}


def response_error(request_id: str, failure: RpcFailure) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": {"code": failure.code, "message": failure.message, "details": failure.details},
    }


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    request_id = "unknown"
    try:
        raw = await reader.readline()
        if not raw or len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            raise RpcFailure("invalid_argument", "request must be one bounded NDJSON record")
        try:
            request = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RpcFailure("invalid_argument", "request is not valid UTF-8 JSON") from exc
        if not isinstance(request, dict):
            raise RpcFailure("invalid_argument", "request must be an object")
        request_id = str(request.get("request_id", ""))
        if not request_id or len(request_id) > 128:
            request_id = "unknown"
            raise RpcFailure("invalid_argument", "request_id is required")
        if request.get("version") != PROTOCOL_VERSION:
            raise RpcFailure(
                "protocol_mismatch",
                "unsupported workspace protocol version",
                {"expected": PROTOCOL_VERSION, "received": request.get("version")},
            )
        method = request.get("method")
        params = request.get("params")
        if method not in ALLOWED_METHODS or not isinstance(params, dict):
            raise RpcFailure("invalid_argument", "unsupported method or invalid params")
        response = response_ok(request_id, await BROKER.dispatch(method, params))
    except RpcFailure as exc:
        response = response_error(request_id, exc)
    except Exception:
        response = response_error(request_id, RpcFailure("internal_error", "unhandled broker error"))
    try:
        writer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def main() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    COMMAND_ROOT.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(
        handle_client, path=str(SOCKET_PATH), limit=MAX_REQUEST_BYTES + 1
    )
    os.chmod(SOCKET_PATH, 0o600)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()
    SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
