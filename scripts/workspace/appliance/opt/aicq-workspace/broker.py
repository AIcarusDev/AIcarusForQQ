#!/usr/bin/python3
"""Rootless Podman broker for the internal AICQ Agent computer appliance."""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
import signal
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path("/opt/aicq-workspace")
MANIFEST_PATH = ROOT / "protocol-manifest.json"
IMAGE_CONTEXT = ROOT / "image"
STATE_ROOT = Path("/var/lib/aicq-workspace")
HOME_ROOT = STATE_ROOT / "home"
COMMAND_ROOT = STATE_ROOT / "commands"
SOCKET_PATH = Path("/run/aicq-workspace/broker.sock")
FIREWALL_MARKER = Path("/run/aicq-workspace/firewall.ready")
PODMAN = "/usr/bin/podman"
CONTAINER_COMMAND_ROOT = "/run/aicq-workspace/commands"
AGENT_HOME = "/home/agent"

MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_STDIN_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_PAGE_BYTES = 64 * 1024
MAX_MODEL_CONTENT_CHARS = 2000
CONTENT_TRUNCATION_MARKER = "[Content too long; truncated]"
MAX_TIMEOUT_SECONDS = 900.0

ALLOWED_METHODS = {
    "health",
    "ensure_default",
    "start_command",
    "wait_command",
    "poll_command",
    "stop_command",
    "read_file",
    "edit_file",
    "write_file",
    "find_files",
    "search",
}
TERMINAL_STATUSES = {"completed", "timed_out", "stopped", "interrupted"}


class RpcFailure(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"{code}: {message}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise RpcFailure("invalid_argument", "text must be valid UTF-8") from exc


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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


async def run_capture(
    argv: list[str], *, stdin: bytes = b"", deadline: float | None = 120.0
) -> tuple[int, bytes, bytes, bool]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout=deadline)
        return int(process.returncode or 0), stdout[:MAX_RESPONSE_BYTES], stderr[:MAX_RESPONSE_BYTES], False
    except asyncio.TimeoutError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        return int(process.returncode or 137), b"", b"", True
    except asyncio.CancelledError:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
        raise


async def podman(
    args: list[str], *, stdin: bytes = b"", deadline: float | None = 120.0
) -> tuple[int, bytes, bytes, bool]:
    return await run_capture([PODMAN, *args], stdin=stdin, deadline=deadline)


def require_default(params: dict[str, Any]) -> None:
    if str(params.get("workspace_id", WORKSPACE_ID)) != WORKSPACE_ID:
        raise RpcFailure("invalid_argument", "only the default Agent computer is supported")


def validate_linux_path(value: Any, *, name: str = "path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise RpcFailure("invalid_argument", f"{name} must be a non-empty Linux path")
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise RpcFailure("invalid_argument", "Windows and host paths are not accepted")
    utf8_size(value)
    path = PurePosixPath(value)
    if str(path) in {"", "."} and name != "cwd":
        raise RpcFailure("invalid_argument", f"{name} must identify a path")
    return posixpath.normpath(value if path.is_absolute() else posixpath.join(AGENT_HOME, value))


async def container_exists() -> bool:
    code, _, _, _ = await podman(["container", "exists", CONTAINER_NAME], deadline=15.0)
    return code == 0


async def container_running() -> bool:
    if not await container_exists():
        return False
    code, stdout, _, _ = await podman(
        ["inspect", "--format", "{{.State.Running}}", CONTAINER_NAME], deadline=15.0
    )
    return code == 0 and stdout.decode("utf-8", errors="replace").strip().lower() == "true"


async def image_exists() -> bool:
    code, _, _, _ = await podman(["image", "exists", IMAGE_NAME], deadline=15.0)
    return code == 0


async def inspect_image_label(label: str) -> str:
    code, stdout, stderr, _ = await podman(
        ["image", "inspect", "--format", f"{{{{ index .Labels \"{label}\" }}}}", IMAGE_NAME],
        deadline=15.0,
    )
    if code != 0:
        raise RpcFailure(
            "container_start_failed",
            "could not inspect the computer system image",
            {"stderr": stderr.decode("utf-8", errors="replace")[-2048:]},
        )
    return stdout.decode("utf-8", errors="replace").strip()


async def inspect_container_label(label: str) -> str:
    code, stdout, stderr, _ = await podman(
        ["inspect", "--format", f"{{{{ index .Config.Labels \"{label}\" }}}}", CONTAINER_NAME],
        deadline=15.0,
    )
    if code != 0:
        raise RpcFailure(
            "container_start_failed",
            "could not inspect the computer container",
            {"stderr": stderr.decode("utf-8", errors="replace")[-2048:]},
        )
    return stdout.decode("utf-8", errors="replace").strip()


async def apply_resource_limits() -> None:
    limits = MANIFEST["limits"]
    code, _, stderr, _ = await podman(
        [
            "update",
            "--cpus",
            str(limits["cpus"]),
            "--memory",
            str(limits["memory_bytes"]),
            "--pids-limit",
            str(limits["pids"]),
            CONTAINER_NAME,
        ],
        deadline=30.0,
    )
    if code != 0:
        raise RpcFailure(
            "container_start_failed",
            "could not apply the configured computer resource limits",
            {"stderr": stderr.decode("utf-8", errors="replace")[-4096:]},
        )


async def require_container() -> dict[str, Any]:
    if not FIREWALL_MARKER.is_file():
        raise RpcFailure(
            "container_start_failed",
            "public_egress firewall is not active; refusing to start the computer",
        )
    if not await image_exists():
        raise RpcFailure(
            "computer_not_built",
            "computer system image does not exist; install it from Web settings",
        )
    protocol_label = await inspect_image_label("io.aicq.workspace.protocol")
    digest_label = await inspect_image_label("io.aicq.workspace.base-digest")
    if protocol_label != str(PROTOCOL_VERSION) or digest_label != BASE_IMAGE_DIGEST:
        raise RpcFailure("computer_needs_upgrade", "computer system image is stale; update it from Web settings")

    started = False
    exists = await container_exists()
    if exists:
        container_protocol = await inspect_container_label("io.aicq.workspace.protocol")
        container_digest = await inspect_container_label("io.aicq.workspace.base-digest")
        if container_protocol != str(PROTOCOL_VERSION) or container_digest != BASE_IMAGE_DIGEST:
            raise RpcFailure(
                "computer_needs_upgrade",
                "computer system is stale; update it from Web settings",
            )
    if not exists:
        raise RpcFailure(
            "computer_not_built",
            "computer container does not exist; install it from Web settings",
        )
    if not await container_running():
        code, _, stderr, _ = await podman(["start", CONTAINER_NAME], deadline=60.0)
        if code != 0:
            raise RpcFailure(
                "container_start_failed",
                "could not start the computer container",
                {"stderr": stderr.decode("utf-8", errors="replace")[-4096:]},
            )
        started = True
    # Podman 4.x treats `podman update` limits as runtime-only. Reapply the
    # persisted desired state on every ensure so resource settings survive
    # stop/start without replacing the long-lived computer container.
    await apply_resource_limits()
    return {
        "workspace_id": WORKSPACE_ID,
        "container_name": CONTAINER_NAME,
        "created": False,
        "started": started,
        "image_digest": BASE_IMAGE_DIGEST,
        "limits": MANIFEST["limits"],
    }


def command_dir(command_id: str) -> Path:
    if not isinstance(command_id, str) or not re.fullmatch(r"[0-9a-f]{32}", command_id):
        raise RpcFailure("invalid_argument", "command_id must be a broker command id")
    return COMMAND_ROOT / command_id


def load_command(command_id: str) -> dict[str, Any]:
    try:
        with (command_dir(command_id) / "meta.json").open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise RpcFailure("command_not_found", "command record was not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RpcFailure("internal_error", "command record could not be read") from exc
    if not isinstance(data, dict):
        raise RpcFailure("internal_error", "command record is invalid")
    return data


def save_command(record: dict[str, Any]) -> None:
    atomic_json(command_dir(str(record["command_id"])) / "meta.json", record)


def public_command(record: dict[str, Any]) -> dict[str, Any]:
    truncated_marker = command_dir(str(record["command_id"])) / "truncated"
    return {
        "command_id": record["command_id"],
        "workspace_id": record["workspace_id"],
        "status": record["status"],
        "cwd": record["cwd"],
        "exit_code": record.get("exit_code"),
        "started_at": record["started_at"],
        "finished_at": record.get("finished_at"),
        "timed_out": record.get("status") == "timed_out",
        "truncated": bool(record.get("truncated", False) or truncated_marker.is_file()),
    }


def _decode_page(raw: bytes, *, final: bool) -> tuple[str, int]:
    if not raw:
        return "", 0
    if final:
        return raw.decode("utf-8", errors="replace"), len(raw)
    end = len(raw)
    for _ in range(4):
        try:
            return raw[:end].decode("utf-8"), end
        except UnicodeDecodeError as exc:
            if exc.end == end and exc.reason == "unexpected end of data" and end > 0:
                end -= 1
                continue
            return raw[:end].decode("utf-8", errors="replace"), end
    return raw.decode("utf-8", errors="replace"), len(raw)


def _model_content_preview(content: str) -> str:
    if len(content) <= MAX_MODEL_CONTENT_CHARS:
        return content
    separator = f"\n{CONTENT_TRUNCATION_MARKER}\n"
    retained_chars = MAX_MODEL_CONTENT_CHARS - len(separator)
    head_chars = retained_chars // 2
    tail_chars = retained_chars - head_chars
    return content[:head_chars] + separator + content[-tail_chars:]


def command_page(record: dict[str, Any], cursor: int) -> dict[str, Any]:
    command_id = str(record["command_id"])
    output_path = command_dir(command_id) / "merged.bin"
    total = output_path.stat().st_size if output_path.exists() else 0
    if cursor < 0 or cursor > total:
        raise RpcFailure("invalid_argument", "cursor is outside the stored command output")
    with output_path.open("rb") if output_path.exists() else open(os.devnull, "rb") as handle:
        handle.seek(cursor)
        raw = handle.read(MAX_PAGE_BYTES + 4)
    bounded = raw[:MAX_PAGE_BYTES]
    final_page = cursor + len(bounded) >= total
    content, consumed = _decode_page(bounded, final=final_page)
    next_cursor = cursor + consumed
    result = public_command(record)
    result.update(
        {
            "content": _model_content_preview(content),
            "cursor": next_cursor,
            "has_more": next_cursor < total,
        }
    )
    if len(content) > MAX_MODEL_CONTENT_CHARS:
        relative_path = PurePosixPath(".aicq") / "command-output" / command_id / f"{cursor}-{next_cursor}.log"
        content_path = HOME_ROOT.joinpath(*relative_path.parts)
        atomic_text(content_path, content)
        result["content_file"] = str(PurePosixPath(AGENT_HOME) / relative_path)
        result["content_chars"] = len(content)
    return result


class CommandSpool:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.lock = asyncio.Lock()
        self.total = 0
        self.truncated = False

    async def add(self, stream_name: str, chunk: bytes) -> None:
        async with self.lock:
            remaining = max(0, MAX_OUTPUT_BYTES - self.total)
            stored = chunk[:remaining]
            if stored:
                with (self.directory / f"{stream_name}.bin").open("ab") as stream_handle:
                    stream_handle.write(stored)
                with (self.directory / "merged.bin").open("ab") as merged_handle:
                    merged_handle.write(stored)
                self.total += len(stored)
            if len(stored) < len(chunk):
                self.truncated = True
                (self.directory / "truncated").touch(exist_ok=True)


async def drain_stream(reader: asyncio.StreamReader, spool: CommandSpool, stream_name: str) -> None:
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            return
        await spool.add(stream_name, chunk)


class Broker:
    def __init__(self) -> None:
        self.ensure_lock = asyncio.Lock()
        self.file_write_lock = asyncio.Lock()
        self.jobs: dict[str, asyncio.Task[None]] = {}
        self.done_events: dict[str, asyncio.Event] = {}
        self.stop_requested: set[str] = set()
        self._reconcile_interrupted()

    def _reconcile_interrupted(self) -> None:
        if not COMMAND_ROOT.is_dir():
            return
        for path in COMMAND_ROOT.glob("*/meta.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict) and record.get("status") == "running":
                    record.update({"status": "interrupted", "finished_at": utc_now(), "exit_code": None})
                    atomic_json(path, record)
            except Exception:
                continue

    async def dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "health":
            return await self.health()
        require_default(params)
        if method == "ensure_default":
            async with self.ensure_lock:
                return await require_container()
        if method == "start_command":
            return await self.start_command(params)
        if method == "wait_command":
            return await self.wait_command(params)
        if method == "poll_command":
            return await self.poll_command(params)
        if method == "stop_command":
            return await self.stop_command(params)
        if method in {"read_file", "edit_file", "write_file", "find_files", "search"}:
            return await self.file_operation(method, params)
        raise RpcFailure("invalid_argument", f"unsupported method: {method}")

    async def _ensure(self) -> None:
        async with self.ensure_lock:
            await require_container()

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

    async def start_command(self, params: dict[str, Any]) -> dict[str, Any]:
        command = params.get("command")
        stdin = params.get("stdin", "")
        cwd = validate_linux_path(params.get("cwd", AGENT_HOME), name="cwd")
        if not isinstance(command, str) or not command or "\x00" in command:
            raise RpcFailure("invalid_argument", "command must be a non-empty string")
        if utf8_size(command) > MAX_COMMAND_BYTES:
            raise RpcFailure("invalid_argument", "command exceeds the 64 KiB limit")
        if not isinstance(stdin, str) or utf8_size(stdin) > MAX_STDIN_BYTES:
            raise RpcFailure("invalid_argument", "stdin exceeds the 1 MiB limit")
        # Fail the model-facing call immediately when the user has not built
        # the container. The broker may start an existing stopped container,
        # but it never builds an image or creates a container.
        await self._ensure()
        command_id = uuid.uuid4().hex
        directory = command_dir(command_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "command.sh").write_text(command, encoding="utf-8")
        (directory / "stdin.bin").write_bytes(stdin.encode("utf-8"))
        for name in ("merged.bin", "stdout.bin", "stderr.bin"):
            (directory / name).touch()
        record = {
            "command_id": command_id,
            "workspace_id": WORKSPACE_ID,
            "status": "running",
            "cwd": cwd,
            "exit_code": None,
            "started_at": utc_now(),
            "finished_at": None,
            "truncated": False,
        }
        save_command(record)
        event = asyncio.Event()
        self.done_events[command_id] = event
        self.jobs[command_id] = asyncio.create_task(
            self._run_command(command_id), name=f"workspace-command-{command_id[:8]}"
        )
        return public_command(record)

    async def _run_command(self, command_id: str) -> None:
        directory = command_dir(command_id)
        record = load_command(command_id)
        spool = CommandSpool(directory)
        process: asyncio.subprocess.Process | None = None
        timed_out = False
        lifetime_started = asyncio.get_running_loop().time()
        try:
            await asyncio.wait_for(self._ensure(), timeout=MAX_TIMEOUT_SECONDS)
            remaining_lifetime = max(
                0.1,
                MAX_TIMEOUT_SECONDS - (asyncio.get_running_loop().time() - lifetime_started),
            )
            process = await asyncio.create_subprocess_exec(
                PODMAN,
                "exec",
                "--user",
                "agent",
                "--workdir",
                AGENT_HOME,
                CONTAINER_NAME,
                "/usr/local/bin/aicq-command-runner",
                command_id,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert process.stdout is not None and process.stderr is not None
            stdout_task = asyncio.create_task(drain_stream(process.stdout, spool, "stdout"))
            stderr_task = asyncio.create_task(drain_stream(process.stderr, spool, "stderr"))
            try:
                await asyncio.wait_for(process.wait(), timeout=remaining_lifetime)
            except asyncio.TimeoutError:
                timed_out = True
                await self._terminate_command(command_id)
                await process.wait()
            await asyncio.gather(stdout_task, stderr_task)
            status = (
                "stopped"
                if command_id in self.stop_requested
                else "timed_out"
                if timed_out
                else "completed"
            )
            record.update(
                {
                    "status": status,
                    "exit_code": process.returncode,
                    "finished_at": utc_now(),
                    "truncated": spool.truncated,
                }
            )
        except asyncio.TimeoutError:
            timed_out = True
            await spool.add(
                "stderr",
                b"computer command exceeded its 900 second lifecycle during initialization\n",
            )
            record.update(
                {
                    "status": "timed_out",
                    "exit_code": None,
                    "finished_at": utc_now(),
                    "truncated": spool.truncated,
                }
            )
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                await self._terminate_command(command_id)
            record.update(
                {
                    "status": "stopped" if command_id in self.stop_requested else "interrupted",
                    "exit_code": None,
                    "finished_at": utc_now(),
                    "truncated": spool.truncated,
                }
            )
            raise
        except Exception as exc:
            await spool.add(
                "stderr",
                f"computer command could not start: {exc}\n".encode("utf-8", errors="replace"),
            )
            record.update(
                {
                    "status": "interrupted",
                    "exit_code": None,
                    "finished_at": utc_now(),
                    "truncated": spool.truncated,
                }
            )
        finally:
            save_command(record)
            self.stop_requested.discard(command_id)
            self.jobs.pop(command_id, None)
            event = self.done_events.pop(command_id, None)
            if event is not None:
                event.set()

    async def _terminate_command(self, command_id: str) -> None:
        pid_path = command_dir(command_id) / "pid"
        for _ in range(20):
            if pid_path.is_file():
                break
            await asyncio.sleep(0.05)
        if not pid_path.is_file():
            return
        try:
            pid = int(pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return
        if pid <= 1:
            return
        await podman(["exec", "--user", "0", CONTAINER_NAME, "kill", "-TERM", "--", f"-{pid}"], deadline=5.0)
        await asyncio.sleep(5.0)
        code, _, _, _ = await podman(
            ["exec", "--user", "0", CONTAINER_NAME, "kill", "-0", "--", f"-{pid}"],
            deadline=5.0,
        )
        if code == 0:
            await podman(["exec", "--user", "0", CONTAINER_NAME, "kill", "-KILL", "--", f"-{pid}"], deadline=5.0)

    async def wait_command(self, params: dict[str, Any]) -> dict[str, Any]:
        command_id = str(params.get("command_id") or "")
        record = load_command(command_id)
        if record.get("status") in TERMINAL_STATUSES:
            return public_command(record)
        event = self.done_events.get(command_id)
        if event is None:
            record.update({"status": "interrupted", "finished_at": utc_now(), "exit_code": None})
            save_command(record)
            return public_command(record)
        await event.wait()
        return public_command(load_command(command_id))

    async def poll_command(self, params: dict[str, Any]) -> dict[str, Any]:
        command_id = str(params.get("command_id") or "")
        try:
            cursor = int(params.get("cursor", 0))
        except (TypeError, ValueError) as exc:
            raise RpcFailure("invalid_argument", "cursor must be an integer") from exc
        return command_page(load_command(command_id), cursor)

    async def stop_command(self, params: dict[str, Any]) -> dict[str, Any]:
        command_id = str(params.get("command_id") or "")
        record = load_command(command_id)
        if record.get("status") in TERMINAL_STATUSES:
            return public_command(record)
        self.stop_requested.add(command_id)
        await self._terminate_command(command_id)
        task = self.jobs.get(command_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=10.0)
            except asyncio.TimeoutError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        return public_command(load_command(command_id))

    async def file_operation(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        if operation in {"edit_file", "write_file"}:
            async with self.file_write_lock:
                return await self._run_file_operation(operation, params)
        return await self._run_file_operation(operation, params)

    async def _run_file_operation(self, operation: str, params: dict[str, Any]) -> dict[str, Any]:
        await self._ensure()
        payload = json.dumps(
            {"operation": operation, "params": params},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        code, stdout, stderr, timed_out = await podman(
            ["exec", "--user", "agent", "--workdir", AGENT_HOME, "--interactive", CONTAINER_NAME, "/usr/local/bin/aicq-file-ops"],
            stdin=payload,
            deadline=30.0,
        )
        if code != 0 or timed_out:
            raise RpcFailure(
                "internal_error",
                "computer file operation failed",
                {"stderr": stderr.decode("utf-8", errors="replace")[-2048:]},
            )
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RpcFailure("internal_error", "computer file operation returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RpcFailure("internal_error", "computer file operation returned invalid data")
        if response.get("ok") is not True:
            error = response.get("error") if isinstance(response.get("error"), dict) else {}
            raise RpcFailure(str(error.get("code") or "internal_error"), str(error.get("message") or "file operation failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise RpcFailure("internal_error", "computer file operation returned no result")
        return result


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
        raw_request_id = request.get("request_id")
        if not isinstance(raw_request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", raw_request_id):
            request_id = "unknown"
            raise RpcFailure("invalid_argument", "request_id is required")
        request_id = raw_request_id
        if request.get("version") != PROTOCOL_VERSION:
            raise RpcFailure(
                "protocol_mismatch",
                "unsupported computer protocol version",
                {"expected": PROTOCOL_VERSION, "received": request.get("version")},
            )
        method = request.get("method")
        params = request.get("params")
        if not isinstance(method, str) or method not in ALLOWED_METHODS or not isinstance(params, dict):
            raise RpcFailure("invalid_argument", "unsupported method or invalid params")
        response = response_ok(request_id, await BROKER.dispatch(str(method), params))
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
        try:
            await writer.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass


async def main() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    HOME_ROOT.mkdir(parents=True, exist_ok=True)
    COMMAND_ROOT.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOCKET_PATH.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(handle_client, path=str(SOCKET_PATH), limit=MAX_REQUEST_BYTES + 1)
    os.chmod(SOCKET_PATH, 0o600)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()
    for task in list(BROKER.jobs.values()):
        task.cancel()
    if BROKER.jobs:
        await asyncio.gather(*BROKER.jobs.values(), return_exceptions=True)
    SOCKET_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
