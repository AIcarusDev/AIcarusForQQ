#!/usr/bin/python3
"""Structured UTF-8 file, glob, and search operations for the workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


MAX_TEXT_BYTES = 1024 * 1024
MAX_READ_PAGE_BYTES = 256 * 1024
MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_LIST_BYTES = 64 * 1024


class OperationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def encode_utf8(value: str, name: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise OperationError("invalid_argument", f"{name} must be valid UTF-8 text") from exc


def resolve_path(value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise OperationError("invalid_argument", "path must be a non-empty Linux path")
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise OperationError("invalid_argument", "Windows and host paths are not accepted")
    encode_utf8(value, "path")
    pure = PurePosixPath(value)
    path = Path(value) if pure.is_absolute() else Path("/workspace") / Path(value)
    return path.resolve(strict=False)


def raw_text(path: Path) -> tuple[bytes, str, bool, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OperationError("path_error", f"file could not be read: {exc}") from exc
    bom = raw.startswith(b"\xef\xbb\xbf")
    body = raw[3:] if bom else raw
    if b"\x00" in body:
        raise OperationError("binary_file", "file contains NUL bytes and is not treated as text")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperationError("binary_file", "file is not valid UTF-8 text") from exc
    newline = "\r\n" if text.count("\r\n") > text.replace("\r\n", "").count("\n") else "\n"
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return raw, normalized, bom, newline


def revision(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def atomic_write(
    path: Path,
    raw: bytes,
    *,
    create_parents: bool = False,
    expected_revision: str | None = None,
    must_not_exist: bool = False,
) -> None:
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir():
        raise OperationError("path_error", "parent directory does not exist")
    previous = None
    try:
        previous = path.stat()
    except FileNotFoundError:
        pass
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        if previous is not None:
            os.fchmod(descriptor, previous.st_mode & 0o7777)
            try:
                os.fchown(descriptor, previous.st_uid, previous.st_gid)
            except PermissionError:
                pass
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if expected_revision is not None:
            try:
                current = path.read_bytes()
            except FileNotFoundError as exc:
                raise OperationError("stale_file", "file changed since it was read; read it again") from exc
            if revision(current) != expected_revision:
                raise OperationError("stale_file", "file changed since it was read; read it again")
        if must_not_exist:
            try:
                os.link(temp_name, path)
            except FileExistsError as exc:
                raise OperationError("stale_file", "file was created concurrently; read it before overwriting") from exc
            os.unlink(temp_name)
        else:
            os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def read_file(params: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(params.get("path"))
    start = int(params.get("start_line", 1))
    count = int(params.get("line_count", MAX_READ_LINES))
    if start < 1 or count < 1 or count > MAX_READ_LINES:
        raise OperationError("invalid_argument", "start_line must be >= 1 and line_count must be in 1..2000")
    raw, text, _, _ = raw_text(path)
    lines = text.splitlines()
    total = len(lines)
    selected: list[str] = []
    truncated_lines: list[int] = []
    used_bytes = 0
    index = start - 1
    end_limit = min(total, index + count)
    while index < end_limit:
        line_number = index + 1
        line = lines[index]
        if len(line) > MAX_LINE_CHARS:
            line = line[:MAX_LINE_CHARS] + "… [line truncated]"
            truncated_lines.append(line_number)
        rendered = f"{line_number}\t{line}"
        size = len((rendered + "\n").encode("utf-8"))
        if selected and used_bytes + size > MAX_READ_PAGE_BYTES:
            break
        if not selected and size > MAX_READ_PAGE_BYTES:
            rendered = rendered.encode("utf-8")[:MAX_READ_PAGE_BYTES].decode("utf-8", errors="ignore")
            truncated_lines.append(line_number)
            size = len(rendered.encode("utf-8"))
        selected.append(rendered)
        used_bytes += size
        index += 1
    end_line = index if selected else min(total, start - 1)
    has_more = index < total
    return {
        "path": str(path),
        "content": "\n".join(selected),
        "revision": revision(raw),
        "start_line": start,
        "end_line": end_line,
        "total_lines": total,
        "has_more": has_more,
        "next_line": index + 1 if has_more else None,
        "truncated_lines": sorted(set(truncated_lines)),
    }


def edit_file(params: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(params.get("path"))
    raw, text, bom, newline = raw_text(path)
    expected = str(params.get("expected_revision") or "")
    if not expected:
        raise OperationError("file_not_read", "read the file before editing it")
    if revision(raw) != expected:
        raise OperationError("stale_file", "file changed since it was read; read it again")
    edits = params.get("edits")
    if not isinstance(edits, list) or not edits:
        raise OperationError("invalid_argument", "edits must not be empty")
    working = text
    replacements = 0
    for item in edits:
        if not isinstance(item, dict):
            raise OperationError("invalid_argument", "each edit must be an object")
        old = item.get("old_text")
        new = item.get("new_text")
        if not isinstance(old, str) or not old:
            raise OperationError("invalid_argument", "old_text must be non-empty")
        if not isinstance(new, str):
            raise OperationError("invalid_argument", "new_text must be a string")
        if "\x00" in old or "\x00" in new:
            raise OperationError("invalid_argument", "edit text must not contain NUL")
        old = old.replace("\r\n", "\n").replace("\r", "\n")
        new = new.replace("\r\n", "\n").replace("\r", "\n")
        matches = working.count(old)
        if item.get("replace_all"):
            if matches < 1:
                raise OperationError("ambiguous_edit", "old_text was not found")
            working = working.replace(old, new)
            replacements += matches
        else:
            if matches != 1:
                raise OperationError("ambiguous_edit", f"old_text must match exactly once; found {matches}")
            working = working.replace(old, new, 1)
            replacements += 1
    encoded_body = encode_utf8(working.replace("\n", newline), "edited content")
    encoded = (b"\xef\xbb\xbf" if bom else b"") + encoded_body
    if len(encoded) > MAX_TEXT_BYTES:
        raise OperationError("invalid_argument", "edited file exceeds the 1 MiB limit")
    atomic_write(path, encoded, expected_revision=expected)
    return {
        "path": str(path),
        "revision": revision(encoded),
        "replacements": replacements,
        "size_bytes": len(encoded),
        "total_lines": line_count(working),
    }


def write_file(params: dict[str, Any]) -> dict[str, Any]:
    path = resolve_path(params.get("path"))
    content = params.get("content")
    if not isinstance(content, str):
        raise OperationError("invalid_argument", "content must be UTF-8 text")
    encoded = encode_utf8(content, "content")
    if len(encoded) > MAX_TEXT_BYTES:
        raise OperationError("invalid_argument", "content exceeds the 1 MiB limit")
    if "\x00" in content:
        raise OperationError("invalid_argument", "content must not contain NUL")
    exists = path.exists()
    expected = params.get("expected_revision")
    if exists:
        if not isinstance(expected, str) or not expected:
            raise OperationError("file_not_read", "read the complete file before overwriting it")
        current = path.read_bytes()
        if revision(current) != expected:
            raise OperationError("stale_file", "file changed since it was read; read it again")
    elif expected:
        raise OperationError("stale_file", "file no longer exists at the revision that was read")
    atomic_write(
        path,
        encoded,
        create_parents=bool(params.get("create_parents", False)),
        expected_revision=str(expected) if exists else None,
        must_not_exist=not exists,
    )
    return {
        "path": str(path),
        "revision": revision(encoded),
        "created": not exists,
        "size_bytes": len(encoded),
        "total_lines": line_count(content.replace("\r\n", "\n").replace("\r", "\n")),
    }


def paginate(lines: list[str], offset: int, limit: int, path: Path) -> dict[str, Any]:
    page = lines[:limit]
    selected: list[str] = []
    used = 0
    truncated = False
    for line in page:
        encoded_line = encode_utf8(line, "search result")
        size = len(encoded_line) + 1
        if selected and used + size > MAX_LIST_BYTES:
            truncated = True
            break
        if not selected and size > MAX_LIST_BYTES:
            suffix = "… [line truncated]"
            budget = max(0, MAX_LIST_BYTES - len(suffix.encode("utf-8")))
            line = encoded_line[:budget].decode("utf-8", errors="ignore") + suffix
            size = len(line.encode("utf-8"))
            truncated = True
        selected.append(line)
        used += size
    consumed = len(selected)
    next_offset = offset + consumed
    has_more = consumed < len(lines)
    return {
        "path": str(path),
        "content": "\n".join(selected),
        "count": consumed,
        "offset": offset,
        "next_offset": next_offset if has_more else None,
        "has_more": has_more,
        "truncated": truncated,
    }


def run_rg(argv: list[str], *, skip_lines: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    limited = False
    with tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=stderr_file)
        assert process.stdout is not None
        try:
            seen = 0
            for raw_line in process.stdout:
                if seen < skip_lines:
                    seen += 1
                    continue
                lines.append(raw_line.rstrip(b"\r\n").decode("utf-8", errors="replace"))
                if len(lines) >= max_lines:
                    limited = True
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                    break
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=5)
        finally:
            process.stdout.close()
        if not limited and returncode not in {0, 1}:
            stderr_file.seek(0)
            message = stderr_file.read().decode("utf-8", errors="replace").strip()
            raise OperationError("path_error", message or "ripgrep failed")
    return lines


def find_files(params: dict[str, Any]) -> dict[str, Any]:
    root = resolve_path(params.get("path", "/workspace"))
    pattern = params.get("pattern")
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 100))
    if not isinstance(pattern, str) or not pattern or "\x00" in pattern or offset < 0 or not 1 <= limit <= 500:
        raise OperationError("invalid_argument", "invalid pattern, offset, or limit")
    encode_utf8(pattern, "pattern")
    lines = run_rg(
        ["rg", "--files", "--hidden", "--sort", "path", "--glob", "!.git/**", "--glob", pattern, str(root)],
        skip_lines=offset,
        max_lines=limit + 1,
    )
    absolute = sorted(str(Path(line).resolve(strict=False)) for line in lines)
    return paginate(absolute, offset, limit, root)


def search(params: dict[str, Any]) -> dict[str, Any]:
    root = resolve_path(params.get("path", "/workspace"))
    pattern = params.get("pattern")
    mode = str(params.get("mode", "content"))
    offset = int(params.get("offset", 0))
    limit = int(params.get("limit", 250))
    before = int(params.get("context_before", 0))
    after = int(params.get("context_after", 0))
    if not isinstance(pattern, str) or not pattern or "\x00" in pattern or mode not in {"content", "files_with_matches", "count"}:
        raise OperationError("invalid_argument", "invalid search pattern or mode")
    encode_utf8(pattern, "pattern")
    if offset < 0 or not 1 <= limit <= 1000 or not 0 <= before <= 20 or not 0 <= after <= 20:
        raise OperationError("invalid_argument", "invalid search pagination or context")
    argv = ["rg", "--hidden", "--sort", "path", "--glob", "!.git/**"]
    if params.get("literal"):
        argv.append("--fixed-strings")
    if params.get("case_sensitive"):
        argv.append("--case-sensitive")
    else:
        argv.append("--ignore-case")
    if params.get("multiline"):
        argv.append("--multiline")
    glob = params.get("glob")
    if isinstance(glob, str) and glob:
        if "\x00" in glob:
            raise OperationError("invalid_argument", "glob must not contain NUL")
        encode_utf8(glob, "glob")
        argv.extend(["--glob", glob])
    if mode == "files_with_matches":
        argv.append("--files-with-matches")
    elif mode == "count":
        argv.append("--count")
    else:
        argv.extend(["--no-heading", "--line-number", "--max-columns", "2000", "--max-columns-preview"])
        if before:
            argv.extend(["--before-context", str(before)])
        if after:
            argv.extend(["--after-context", str(after)])
    if argv[-1] != "--":
        argv.append("--")
    argv.extend([pattern, str(root)])
    lines = run_rg(argv, skip_lines=offset, max_lines=limit + 1)
    return paginate(lines, offset, limit, root)


OPERATIONS = {
    "read_file": read_file,
    "edit_file": edit_file,
    "write_file": write_file,
    "find_files": find_files,
    "search": search,
}


def main() -> int:
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise OperationError("invalid_argument", "request must be an object")
        operation = str(request.get("operation") or "")
        handler = OPERATIONS.get(operation)
        if handler is None:
            raise OperationError("invalid_argument", "unknown file operation")
        result = handler(request.get("params") if isinstance(request.get("params"), dict) else {})
        json.dump({"ok": True, "result": result}, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        return 0
    except OperationError as exc:
        json.dump({"ok": False, "error": {"code": exc.code, "message": exc.message}}, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        return 0
    except Exception as exc:
        json.dump({"ok": False, "error": {"code": "internal_error", "message": str(exc)}}, sys.stdout, ensure_ascii=False, separators=(",", ":"))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
