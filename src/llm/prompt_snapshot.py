"""Persist sanitized LLM request snapshots for offline debugging."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger("AICQ.llm.prompt_snapshot")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT_DIR = "logs/llm_prompts"
_SCHEMA_VERSION = 1

_WRITE_LOCK = threading.Lock()
_MAINTENANCE_LOCK = threading.Lock()
_last_maintenance_at = 0.0

_DATA_URI_RE = re.compile(
    r"data:(?P<mime>[a-zA-Z0-9_.+/-]+)?;base64,(?P<payload>[A-Za-z0-9+/\r\n]+={0,2})"
)
_RAW_B64_RE = re.compile(
    r"(?<![A-Za-z0-9+/])(?P<payload>[A-Za-z0-9+/]{4096,}={0,2})(?![A-Za-z0-9+/=])"
)


def normalize_prompt_snapshot_config(cfg: dict | None) -> dict:
    raw = dict(cfg or {})
    include_raw = raw.get("include")
    include = dict(include_raw) if isinstance(include_raw, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "root_dir": str(raw.get("root_dir") or _DEFAULT_ROOT_DIR),
        "include": {
            "main_round": bool(include.get("main_round", True)),
            "simple_text": bool(include.get("simple_text", False)),
            "forced_tool": bool(include.get("forced_tool", False)),
        },
        "compress_after_days": _bounded_int(raw.get("compress_after_days"), 1, 0, 3650),
        "bundle_after_days": _bounded_int(raw.get("bundle_after_days"), 14, 0, 3650),
        "delete_after_days": _bounded_int(raw.get("delete_after_days"), 60, 0, 3650),
        "max_total_mb": _bounded_int(raw.get("max_total_mb"), 2048, 0, 1024 * 1024),
        "maintenance_interval_seconds": _bounded_int(
            raw.get("maintenance_interval_seconds"), 3600, 0, 7 * 24 * 3600
        ),
    }


def save_prompt_snapshot(
    cfg: dict | None,
    *,
    request_kind: str,
    provider: str,
    model: str,
    messages: list,
    create_kwargs: dict,
    feature: str = "",
    subfeature: str = "",
    context: dict | None = None,
) -> str:
    """Write one sanitized request snapshot and return its id.

    The live API request is not modified. The persisted copy keeps message shape
    but replaces image URLs, data URIs, and very long base64 blocks with hashes.
    """
    normalized = normalize_prompt_snapshot_config(cfg)
    if not normalized["enabled"]:
        return ""
    if not normalized["include"].get(request_kind, False):
        return ""

    root = _resolve_root_dir(normalized["root_dir"])
    now = datetime.now().astimezone()
    snapshot_id = f"{now.strftime('%Y%m%dT%H%M%S%f')[:-3]}-{uuid.uuid4().hex[:8]}"
    stats = {
        "image_url_count": 0,
        "data_uri_count": 0,
        "raw_base64_count": 0,
    }
    record = {
        "schema_version": _SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": now.isoformat(timespec="milliseconds"),
        "request_kind": request_kind,
        "provider": provider,
        "model": model,
        "feature": feature,
        "subfeature": subfeature,
        "context": _json_safe(context or {}),
        "request": {
            **_sanitize_for_snapshot(create_kwargs, stats),
            "messages": _sanitize_for_snapshot(messages, stats),
        },
        "redactions": stats,
    }

    try:
        day_dir = root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        target = day_dir / f"{_safe_kind(request_kind)}.jsonl"
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with _WRITE_LOCK:
            with target.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.write("\n")
        _maybe_run_maintenance(root, normalized, current_path=target)
    except Exception:
        logger.debug("[prompt_snapshot] failed to persist snapshot", exc_info=True)
        return ""
    return snapshot_id


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _resolve_root_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _safe_kind(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "llm_request"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


def _sanitize_for_snapshot(value: Any, stats: dict[str, int]) -> Any:
    if isinstance(value, str):
        return _sanitize_text(value, stats)
    if isinstance(value, dict):
        if _is_image_url_part(value):
            return _sanitize_image_url_part(value, stats)
        return {
            str(k): _sanitize_for_snapshot(v, stats)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_for_snapshot(item, stats) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_snapshot(item, stats) for item in value]
    try:
        copy.deepcopy(value)
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _sanitize_text(text: str, stats: dict[str, int]) -> str:
    def _data_uri_repl(match: re.Match) -> str:
        stats["data_uri_count"] += 1
        meta = _data_uri_metadata(match.group(0))
        return (
            "[data-uri omitted:"
            f" sha256={meta['sha256']}; bytes={meta['byte_size']};"
            f" mime={meta.get('mime_type') or 'unknown'}]"
        )

    def _raw_b64_repl(match: re.Match) -> str:
        stats["raw_base64_count"] += 1
        payload = match.group("payload")
        meta = _base64_payload_metadata(payload)
        return f"[base64 omitted: sha256={meta['sha256']}; bytes={meta['byte_size']}]"

    text = _DATA_URI_RE.sub(_data_uri_repl, text)
    return _RAW_B64_RE.sub(_raw_b64_repl, text)


def _is_image_url_part(value: dict) -> bool:
    return value.get("type") == "image_url" or "image_url" in value and isinstance(
        value.get("image_url"), (dict, str)
    )


def _sanitize_image_url_part(value: dict, stats: dict[str, int]) -> dict:
    stats["image_url_count"] += 1
    image_url = value.get("image_url")
    detail = None
    url = ""
    if isinstance(image_url, dict):
        detail = image_url.get("detail")
        url = image_url.get("url") if isinstance(image_url.get("url"), str) else ""
    elif isinstance(image_url, str):
        url = image_url

    placeholder = _image_url_metadata(url)
    if detail is not None:
        placeholder["detail"] = _json_safe(detail)
    return {
        **{
            str(k): _sanitize_for_snapshot(v, stats)
            for k, v in value.items()
            if k not in {"image_url"}
        },
        "type": value.get("type", "image_url"),
        "image_url": placeholder,
    }


def _image_url_metadata(url: str) -> dict:
    if not url:
        return {
            "placeholder": "[image]",
            "source": "empty",
            "sha256": "",
            "byte_size": 0,
        }
    if url.startswith("data:"):
        meta = _data_uri_metadata(url)
        return {
            "placeholder": "[image]",
            "source": "data_url",
            "sha256": meta["sha256"],
            "byte_size": meta["byte_size"],
            "mime_type": meta.get("mime_type", ""),
            "hash_input": meta["hash_input"],
        }
    data = url.encode("utf-8", errors="replace")
    return {
        "placeholder": "[image]",
        "source": _url_source(url),
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_size": len(data),
        "hash_input": "url_text",
    }


def _url_source(url: str) -> str:
    lowered = url.lower()
    if lowered.startswith(("http://", "https://")):
        return "url"
    if lowered.startswith("file:"):
        return "file_url"
    return "opaque_url"


def _data_uri_metadata(data_uri: str) -> dict:
    match = _DATA_URI_RE.match(data_uri)
    if not match:
        data = data_uri.encode("utf-8", errors="replace")
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
            "mime_type": "",
            "hash_input": "data_uri_text",
        }
    payload_meta = _base64_payload_metadata(match.group("payload"))
    payload_meta["mime_type"] = match.group("mime") or ""
    payload_meta["hash_input"] = "decoded_bytes"
    return payload_meta


def _base64_payload_metadata(payload: str) -> dict:
    compact = re.sub(r"\s+", "", payload)
    try:
        data = base64.b64decode(compact, validate=False)
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        }
    except Exception:
        data = compact.encode("utf-8", errors="replace")
        return {
            "sha256": hashlib.sha256(data).hexdigest(),
            "byte_size": len(data),
        }


def _maybe_run_maintenance(root: Path, cfg: dict, *, current_path: Path) -> None:
    global _last_maintenance_at
    interval = int(cfg.get("maintenance_interval_seconds", 3600))
    now = time.time()
    if interval > 0 and now - _last_maintenance_at < interval:
        return
    if not _MAINTENANCE_LOCK.acquire(blocking=False):
        return
    try:
        if interval > 0 and now - _last_maintenance_at < interval:
            return
        _last_maintenance_at = now
        _run_maintenance(root, cfg, current_path=current_path)
    finally:
        _MAINTENANCE_LOCK.release()


def _run_maintenance(root: Path, cfg: dict, *, current_path: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        _compress_old_jsonl(root, cfg, current_path=current_path)
        _bundle_old_gzip_files(root, cfg)
        _delete_old_files(root, cfg, current_path=current_path)
        _enforce_total_size(root, cfg, current_path=current_path)
        _remove_empty_dirs(root)
    except Exception:
        logger.debug("[prompt_snapshot] maintenance failed", exc_info=True)


def _compress_old_jsonl(root: Path, cfg: dict, *, current_path: Path) -> None:
    days = int(cfg.get("compress_after_days", 1))
    if days < 0:
        return
    cutoff = time.time() - days * 86400
    for path in root.glob("*/*.jsonl"):
        if path == current_path:
            continue
        try:
            if path.stat().st_mtime > cutoff:
                continue
            gz_path = path.with_suffix(path.suffix + ".gz")
            with path.open("rb") as src, gzip.open(gz_path, "ab") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
        except Exception:
            logger.debug("[prompt_snapshot] compress failed: %s", path, exc_info=True)


def _bundle_old_gzip_files(root: Path, cfg: dict) -> None:
    days = int(cfg.get("bundle_after_days", 14))
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    archive_dir = root / "archive"
    for day_dir in root.iterdir() if root.exists() else []:
        if not day_dir.is_dir() or day_dir.name == "archive":
            continue
        gz_files = [
            path
            for path in day_dir.glob("*.jsonl.gz")
            if _mtime_or_now(path) <= cutoff
        ]
        if not gz_files:
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = archive_dir / f"{day_dir.name}.tar.gz"
        if bundle_path.exists():
            bundle_path = archive_dir / f"{day_dir.name}-{int(time.time())}.tar.gz"
        try:
            with tarfile.open(bundle_path, "w:gz") as tar:
                for path in gz_files:
                    tar.add(path, arcname=path.relative_to(root).as_posix())
            for path in gz_files:
                path.unlink()
        except Exception:
            logger.debug("[prompt_snapshot] bundle failed: %s", day_dir, exc_info=True)


def _delete_old_files(root: Path, cfg: dict, *, current_path: Path) -> None:
    days = int(cfg.get("delete_after_days", 60))
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    for path in _iter_snapshot_files(root):
        if path == current_path:
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                path.unlink()
        except Exception:
            logger.debug("[prompt_snapshot] delete old file failed: %s", path, exc_info=True)


def _enforce_total_size(root: Path, cfg: dict, *, current_path: Path) -> None:
    max_mb = int(cfg.get("max_total_mb", 2048))
    if max_mb <= 0:
        return
    max_bytes = max_mb * 1024 * 1024
    files = []
    total = 0
    for path in _iter_snapshot_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        files.append((stat.st_mtime, stat.st_size, path))
    if total <= max_bytes:
        return
    for _mtime, size, path in sorted(files):
        if path == current_path:
            continue
        try:
            path.unlink()
            total -= size
        except Exception:
            logger.debug("[prompt_snapshot] size pruning failed: %s", path, exc_info=True)
        if total <= max_bytes:
            break


def _iter_snapshot_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.name.endswith(".jsonl")
            or path.name.endswith(".jsonl.gz")
            or path.name.endswith(".tar.gz")
        )
    ]


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if path == root:
            continue
        try:
            path.rmdir()
        except OSError:
            pass


def _mtime_or_now(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()
