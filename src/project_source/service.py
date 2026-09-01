"""Read-only access to static text inside the application checkout.

The security boundary is intentionally location-based. It protects declared
secret-bearing files and fields, but it does not inspect arbitrary source text
for strings that merely look like credentials.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal

import yaml


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAX_TEXT_BYTES = 8 * 1024 * 1024
MAX_READ_CHARS = 16_000
MAX_SEARCH_FILES = 10_000
MAX_EXCERPT_CHARS = 500

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_ENV_PUBLIC_SUFFIXES = (".example", ".sample", ".template")
_PROTECTED_FILE_NAMES = {
    ".launcher_env",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "service-account.json",
    "service_account.json",
    "id_rsa",
    "id_ed25519",
}
_PROTECTED_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".jks", ".keystore"}
_PROTECTED_DIRECTORY_NAMES = {".git", ".ssh", ".gnupg"}
_UNSUPPORTED_SUBTREES = {
    ("cache",),
    ("data", "image_cache"),
    ("data", "stickers"),
    ("data", "tts_cache"),
}
_UNSUPPORTED_SUFFIXES = {
    ".sqlite",
    ".sqlite3",
    ".db",
    ".db-shm",
    ".db-wal",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".svgz",
    ".mp3",
    ".wav",
    ".ogg",
    ".flac",
    ".m4a",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".zip",
    ".7z",
    ".rar",
    ".gz",
    ".bz2",
    ".xz",
    ".tar",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".pyc",
    ".pyo",
    ".class",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}
_STRUCTURED_MASKS: dict[str, tuple[tuple[str, ...], ...]] = {
    "config/config_user.yaml": (("tts", "secret_token"),),
}
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


@dataclass(frozen=True)
class AccessDecision:
    access: Literal["allowed", "filtered", "denied", "unsupported"]
    code: str = ""


class ProjectSourceError(RuntimeError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.message = message

    def result(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "error": self.message,
            "path": self.path,
        }


@dataclass(frozen=True)
class _LoadedText:
    path: str
    text: str
    revision: str
    filtered_fields: tuple[str, ...] = ()


class ProjectSourceService:
    """Browse and read text below one trusted project root."""

    def __init__(self, root: str | Path = DEFAULT_PROJECT_ROOT) -> None:
        self.root = Path(root).resolve(strict=True)

    def list_directory(self, path: str = ".", *, offset: int = 0, limit: int = 100) -> dict[str, Any]:
        try:
            logical, directory = self._resolve_existing(path)
            if not directory.is_dir():
                raise ProjectSourceError("not_a_directory", logical, "这个路径不是目录。")
            entries = sorted(
                directory.iterdir(),
                key=self._entry_sort_key,
            )
            page = entries[offset : offset + limit]
            rows = [self._entry_row(item) for item in page]
            next_offset = offset + len(page) if offset + len(page) < len(entries) else None
            result: dict[str, Any] = {
                "ok": True,
                "path": logical,
                "entries": rows,
                "total": len(entries),
            }
            if next_offset is not None:
                result["next_offset"] = next_offset
            return result
        except ProjectSourceError as exc:
            return exc.result()
        except OSError:
            return ProjectSourceError("list_failed", self._display_path(path), "无法列出这个目录。").result()

    def read_file(
        self,
        path: str,
        *,
        start_line: int = 1,
        line_count: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        try:
            loaded = self._load_text(path)
            total_lines = len(loaded.text.splitlines())
            if cursor:
                cursor_data = self._decode_cursor(cursor)
                if cursor_data.get("path") != loaded.path:
                    raise ProjectSourceError("invalid_cursor", loaded.path, "继续读取位置与文件不匹配。")
                if cursor_data.get("revision") != loaded.revision:
                    raise ProjectSourceError("file_changed", loaded.path, "文件已经变化，请重新读取。")
                offset = int(cursor_data.get("offset", -1))
                if offset < 0 or offset > len(loaded.text):
                    raise ProjectSourceError("invalid_cursor", loaded.path, "继续读取位置无效。")
            else:
                offset = self._offset_for_line(loaded.text, start_line, loaded.path)

            actual_start_line = loaded.text.count("\n", 0, offset) + 1
            end_by_lines = self._end_offset_for_lines(loaded.text, offset, line_count)
            end = min(end_by_lines, offset + MAX_READ_CHARS)
            content = loaded.text[offset:end]
            result: dict[str, Any] = {
                "ok": True,
                "path": loaded.path,
                "content": content,
                "start_line": actual_start_line,
                "end_line": self._end_line(actual_start_line, content),
                "total_lines": total_lines,
            }
            if loaded.filtered_fields:
                result["filtered_fields"] = list(loaded.filtered_fields)
            if end < len(loaded.text):
                result["next_cursor"] = self._encode_cursor(loaded.path, loaded.revision, end)
            return result
        except ProjectSourceError as exc:
            return exc.result()

    def search(
        self,
        query: str,
        *,
        path: str = ".",
        glob: str | None = None,
        mode: Literal["content", "path"] = "content",
        case_sensitive: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        try:
            logical_root, search_root = self._resolve_existing(path)
            if not search_root.is_dir():
                raise ProjectSourceError("not_a_directory", logical_root, "搜索路径不是目录。")
            if mode == "path":
                return self._search_paths(
                    query,
                    search_root,
                    logical_root=logical_root,
                    glob=glob,
                    case_sensitive=case_sensitive,
                    offset=offset,
                    limit=limit,
                )
            return self._search_content(
                query,
                search_root,
                logical_root=logical_root,
                glob=glob,
                case_sensitive=case_sensitive,
                offset=offset,
                limit=limit,
            )
        except ProjectSourceError as exc:
            return exc.result()
        except OSError:
            return ProjectSourceError("search_failed", self._display_path(path), "无法搜索这个目录。").result()

    def access_for(self, logical_path: str, *, is_directory: bool = False) -> AccessDecision:
        pure = PurePosixPath(logical_path)
        lowered_parts = tuple(part.casefold() for part in pure.parts if part not in ("", "."))
        if any(part in _PROTECTED_DIRECTORY_NAMES for part in lowered_parts):
            return AccessDecision("denied", "protected_source")
        if self._inside_subtree(lowered_parts, _UNSUPPORTED_SUBTREES):
            return AccessDecision("unsupported", "unsupported_source")
        if is_directory:
            return AccessDecision("allowed")

        name = pure.name.casefold()
        suffix = pure.suffix.casefold()
        suffixes = tuple(item.casefold() for item in pure.suffixes)
        if self._is_protected_env_name(name):
            return AccessDecision("denied", "protected_source")
        if name in _PROTECTED_FILE_NAMES or suffix in _PROTECTED_SUFFIXES:
            return AccessDecision("denied", "protected_source")
        if suffix in _UNSUPPORTED_SUFFIXES or "".join(suffixes[-2:]) in _UNSUPPORTED_SUFFIXES:
            return AccessDecision("unsupported", "unsupported_file_type")
        if pure.as_posix().casefold() in _STRUCTURED_MASKS:
            return AccessDecision("filtered")
        return AccessDecision("allowed")

    def _entry_row(self, path: Path) -> dict[str, Any]:
        logical = self._logical_from_path(path)
        try:
            stat_result = path.lstat()
        except OSError:
            return {"name": path.name, "path": logical, "kind": "unknown", "content_access": "denied"}
        is_link = self._is_reparse(stat_result) or path.is_symlink()
        if is_link:
            return {"name": path.name, "path": logical, "kind": "link", "content_access": "denied"}
        is_directory = stat.S_ISDIR(stat_result.st_mode)
        decision = self.access_for(logical, is_directory=is_directory)
        row: dict[str, Any] = {
            "name": path.name,
            "path": logical,
            "kind": "directory" if is_directory else "file",
            "content_access": decision.access,
        }
        if not is_directory:
            row["size"] = stat_result.st_size
        return row

    def _load_text(self, path: str) -> _LoadedText:
        logical, resolved = self._resolve_existing(path)
        if not resolved.is_file():
            raise ProjectSourceError("not_a_file", logical, "这个路径不是文件。")
        decision = self.access_for(logical)
        if decision.access == "denied":
            raise ProjectSourceError("permission_denied", logical, "没有权限读取这个文件。")
        if decision.access == "unsupported":
            raise ProjectSourceError(decision.code, logical, "不支持读取这个文件。")

        raw = self._read_file_bytes(resolved, logical)
        if raw.startswith(b"SQLite format 3\x00"):
            raise ProjectSourceError("unsupported_file_type", logical, "不支持读取这个文件。")
        if b"\x00" in raw:
            raise ProjectSourceError("unsupported_file_type", logical, "不支持读取这个文件。")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ProjectSourceError("unsupported_text_encoding", logical, "文件不是受支持的 UTF-8 文本。") from exc

        filtered_fields: tuple[str, ...] = ()
        masks = _STRUCTURED_MASKS.get(logical.casefold(), ())
        if masks:
            text, filtered_fields = self._mask_yaml_fields(text, masks, logical)
        try:
            stat_result = resolved.stat()
        except OSError as exc:
            raise ProjectSourceError("read_failed", logical, "无法读取这个文件。") from exc
        revision = f"{stat_result.st_size}:{stat_result.st_mtime_ns}"
        return _LoadedText(logical, text, revision, filtered_fields)

    def _read_file_bytes(self, path: Path, logical: str) -> bytes:
        try:
            size = path.stat().st_size
            if size > MAX_TEXT_BYTES:
                raise ProjectSourceError("file_too_large", logical, "文件超过可读取大小限制。")
            return path.read_bytes()
        except ProjectSourceError:
            raise
        except OSError as exc:
            raise ProjectSourceError("read_failed", logical, "无法读取这个文件。") from exc

    def _resolve_existing(self, path: str) -> tuple[str, Path]:
        logical = self._normalize_logical_path(path)
        candidate = self.root if logical == "." else self.root.joinpath(*PurePosixPath(logical).parts)
        current = self.root
        if logical != ".":
            for part in PurePosixPath(logical).parts:
                current = current / part
                try:
                    stat_result = current.lstat()
                except FileNotFoundError as exc:
                    raise ProjectSourceError("not_found", logical, "没有找到这个路径。") from exc
                except OSError as exc:
                    raise ProjectSourceError("path_unavailable", logical, "无法访问这个路径。") from exc
                if self._is_reparse(stat_result) or current.is_symlink():
                    raise ProjectSourceError("permission_denied", logical, "没有权限访问这个路径。")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root)
        except FileNotFoundError as exc:
            raise ProjectSourceError("not_found", logical, "没有找到这个路径。") from exc
        except (OSError, ValueError) as exc:
            raise ProjectSourceError("path_outside_project", logical, "路径不在当前项目中。") from exc
        return logical, resolved

    def _normalize_logical_path(self, path: str) -> str:
        if not isinstance(path, str):
            raise ProjectSourceError("invalid_path", "", "路径必须是项目相对路径。")
        raw = path.strip().replace("\\", "/")
        if not raw:
            raw = "."
        if "\x00" in raw or ":" in raw or raw.startswith("/") or raw.startswith("//") or _WINDOWS_DRIVE_RE.match(raw):
            raise ProjectSourceError("invalid_path", self._display_path(raw), "路径必须是项目相对路径。")
        pure = PurePosixPath(raw)
        if any(part == ".." for part in pure.parts):
            raise ProjectSourceError("path_outside_project", self._display_path(raw), "路径不在当前项目中。")
        parts = [part for part in pure.parts if part not in ("", ".")]
        return PurePosixPath(*parts).as_posix() if parts else "."

    def _logical_from_path(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _search_paths(
        self,
        query: str,
        root: Path,
        *,
        logical_root: str,
        glob: str | None,
        case_sensitive: bool,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        for path in self._walk(root, include_protected=True):
            logical = self._logical_from_path(path)
            haystack = logical if case_sensitive else logical.casefold()
            if needle not in haystack or not self._glob_matches(logical, glob):
                continue
            matches.append({
                "path": logical,
                "kind": "directory" if path.is_dir() else "file",
            })
            if len(matches) > offset + limit:
                break
        return self._paged_search_result(logical_root, matches, offset, limit)

    def _search_content(
        self,
        query: str,
        root: Path,
        *,
        logical_root: str,
        glob: str | None,
        case_sensitive: bool,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        needle = query if case_sensitive else query.casefold()
        matches: list[dict[str, Any]] = []
        skipped: Counter[str] = Counter()
        scanned_files = 0
        scan_truncated = False
        for path in self._walk(root, include_protected=False):
            if not path.is_file():
                continue
            logical = self._logical_from_path(path)
            if not self._glob_matches(logical, glob):
                continue
            if scanned_files >= MAX_SEARCH_FILES:
                scan_truncated = True
                break
            scanned_files += 1
            try:
                loaded = self._load_text(logical)
            except ProjectSourceError as exc:
                skipped[exc.code] += 1
                continue
            for line_number, line in enumerate(loaded.text.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                if needle not in haystack:
                    continue
                excerpt = line if len(line) <= MAX_EXCERPT_CHARS else line[:MAX_EXCERPT_CHARS]
                matches.append({"path": logical, "line": line_number, "text": excerpt})
                if len(matches) > offset + limit:
                    break
            if len(matches) > offset + limit:
                break
        result = self._paged_search_result(logical_root, matches, offset, limit)
        result["scanned_files"] = scanned_files
        if skipped:
            result["skipped"] = dict(sorted(skipped.items()))
        if scan_truncated:
            result["scan_truncated"] = True
        return result

    def _walk(self, root: Path, *, include_protected: bool) -> Iterable[Path]:
        for current_root, dir_names, file_names in os.walk(root, topdown=True, followlinks=False):
            current = Path(current_root)
            safe_dirs: list[str] = []
            for name in sorted(dir_names, key=str.casefold):
                child = current / name
                logical = self._logical_from_path(child)
                try:
                    stat_result = child.lstat()
                except OSError:
                    continue
                if self._is_reparse(stat_result) or child.is_symlink():
                    continue
                decision = self.access_for(logical, is_directory=True)
                if include_protected or decision.access == "allowed":
                    safe_dirs.append(name)
                if include_protected:
                    yield child
            dir_names[:] = safe_dirs
            for name in sorted(file_names, key=str.casefold):
                child = current / name
                try:
                    stat_result = child.lstat()
                except OSError:
                    continue
                if self._is_reparse(stat_result) or child.is_symlink():
                    continue
                yield child

    @staticmethod
    def _paged_search_result(path: str, matches: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
        page = matches[offset : offset + limit]
        result: dict[str, Any] = {"ok": True, "path": path, "matches": page}
        if len(matches) > offset + limit:
            result["next_offset"] = offset + limit
        return result

    @staticmethod
    def _glob_matches(logical: str, pattern: str | None) -> bool:
        if not pattern:
            return True
        pure = PurePosixPath(logical)
        if pure.match(pattern):
            return True
        return pattern.startswith("**/") and pure.match(pattern[3:])

    @staticmethod
    def _inside_subtree(parts: tuple[str, ...], subtrees: set[tuple[str, ...]]) -> bool:
        return any(len(parts) >= len(prefix) and parts[: len(prefix)] == prefix for prefix in subtrees)

    @staticmethod
    def _is_protected_env_name(name: str) -> bool:
        if name.endswith(_ENV_PUBLIC_SUFFIXES):
            return False
        return name == ".env" or name.startswith(".env.") or name.endswith(".env") or ".env." in name

    @staticmethod
    def _is_reparse(stat_result: os.stat_result) -> bool:
        return bool(getattr(stat_result, "st_file_attributes", 0) & _REPARSE_POINT)

    @staticmethod
    def _offset_for_line(text: str, start_line: int, logical: str) -> int:
        if start_line <= 1:
            return 0
        lines = text.splitlines(keepends=True)
        if start_line > len(lines):
            raise ProjectSourceError("invalid_range", logical, "起始行超出文件范围。")
        return sum(len(line) for line in lines[: start_line - 1])

    @staticmethod
    def _end_offset_for_lines(text: str, offset: int, line_count: int) -> int:
        remaining = text[offset:]
        lines = remaining.splitlines(keepends=True)
        return offset + len("".join(lines[:line_count]))

    @staticmethod
    def _end_line(start_line: int, content: str) -> int:
        if not content:
            return start_line
        newline_count = content.count("\n")
        return start_line + newline_count - (1 if content.endswith("\n") else 0)

    @staticmethod
    def _encode_cursor(path: str, revision: str, offset: int) -> str:
        payload = json.dumps(
            {"v": 1, "path": path, "revision": revision, "offset": offset},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_cursor(cursor: str) -> dict[str, Any]:
        try:
            padding = "=" * (-len(cursor) % 4)
            payload = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
            value = json.loads(payload.decode("utf-8"))
            if (
                not isinstance(value, dict)
                or value.get("v") != 1
                or not isinstance(value.get("path"), str)
                or not isinstance(value.get("revision"), str)
                or not isinstance(value.get("offset"), int)
                or isinstance(value.get("offset"), bool)
            ):
                raise ValueError
            return value
        except (ValueError, UnicodeError, json.JSONDecodeError, binascii.Error) as exc:
            raise ProjectSourceError("invalid_cursor", "", "继续读取位置无效。") from exc

    @staticmethod
    def _mask_yaml_fields(
        text: str,
        field_paths: tuple[tuple[str, ...], ...],
        logical: str,
    ) -> tuple[str, tuple[str, ...]]:
        try:
            roots = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
        except yaml.YAMLError as exc:
            raise ProjectSourceError("protected_source_invalid", logical, "受保护配置无法安全读取。") from exc
        replacements: dict[tuple[int, int], str] = {}
        filtered: list[str] = []
        for field_path in field_paths:
            nodes = [
                node
                for root in roots
                for node in ProjectSourceService._yaml_nodes_at(root, field_path)
            ]
            if not nodes:
                continue
            for node in nodes:
                replacements[(node.start_mark.index, node.end_mark.index)] = '"***"'
            filtered.append(".".join(field_path))
        for (start, end), replacement in sorted(replacements.items(), reverse=True):
            text = text[:start] + replacement + text[end:]
        return text, tuple(filtered)

    @staticmethod
    def _yaml_nodes_at(node: yaml.Node | None, parts: tuple[str, ...]) -> list[yaml.Node]:
        if not parts:
            return [] if node is None else [node]
        if not isinstance(node, yaml.MappingNode):
            return []
        matches: list[yaml.Node] = []
        part = parts[0]
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and str(key_node.value) == part:
                matches.extend(ProjectSourceService._yaml_nodes_at(value_node, parts[1:]))
        return matches

    @staticmethod
    def _entry_sort_key(path: Path) -> tuple[bool, str, str]:
        try:
            stat_result = path.lstat()
            is_directory = stat.S_ISDIR(stat_result.st_mode) and not (
                ProjectSourceService._is_reparse(stat_result) or path.is_symlink()
            )
        except OSError:
            is_directory = False
        return not is_directory, path.name.casefold(), path.name

    @staticmethod
    def _display_path(path: object) -> str:
        value = str(path or "").replace("\\", "/")
        return value[:512]


@lru_cache(maxsize=1)
def get_default_service() -> ProjectSourceService:
    return ProjectSourceService(DEFAULT_PROJECT_ROOT)


__all__ = ["ProjectSourceService", "get_default_service"]
