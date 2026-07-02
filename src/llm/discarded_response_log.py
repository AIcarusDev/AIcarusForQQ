"""Persist discarded model response fragments for local debugging."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import tarfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger("AICQ.llm.discarded_response")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT_DIR = "logs/llm_discards"
_SCHEMA_VERSION = 1

_WRITE_LOCK = threading.Lock()
_MAINTENANCE_LOCK = threading.Lock()
_last_maintenance_at = 0.0


def normalize_discarded_response_log_config(raw: dict | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    max_file_bytes_raw = cfg.get("max_file_bytes")
    max_file_bytes = (
        _bounded_int(max_file_bytes_raw, 64 * 1024 * 1024, 1, 1024 * 1024 * 1024)
        if max_file_bytes_raw is not None
        else _bounded_int(cfg.get("max_file_mb"), 64, 1, 1024) * 1024 * 1024
    )
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "root_dir": str(cfg.get("root_dir") or _DEFAULT_ROOT_DIR),
        "compress_after_days": _bounded_int(cfg.get("compress_after_days"), 1, 0, 3650),
        "bundle_after_days": _bounded_int(cfg.get("bundle_after_days"), 14, 0, 3650),
        "delete_after_days": _bounded_int(cfg.get("delete_after_days"), 60, 0, 3650),
        "max_total_mb": _bounded_int(cfg.get("max_total_mb"), 512, 0, 1024 * 1024),
        "max_file_bytes": max_file_bytes,
        "maintenance_interval_seconds": _bounded_int(
            cfg.get("maintenance_interval_seconds"), 3600, 0, 7 * 24 * 3600
        ),
    }


def save_cognition_prefill_discard(
    cfg: dict | None,
    *,
    provider: str,
    model: str,
    feature: str = "",
    subfeature: str = "",
    prompt_snapshot_id: str = "",
    agent_run_id: str = "",
    context: dict | None = None,
    retry_attempt: int = 1,
    similarity: float,
    matched_index: int,
    discarded_cognition: str,
    matched_cognition: str,
    chosen_prefill: str,
    visible_cognitions_count: int = 0,
    prefill_exclusions: list[str] | tuple[str, ...] = (),
    guard_config: dict | None = None,
) -> str:
    normalized = normalize_discarded_response_log_config(cfg)
    if not normalized["enabled"]:
        return ""

    now = datetime.now().astimezone()
    event_id = f"{now.strftime('%Y%m%dT%H%M%S%f')[:-3]}-{uuid.uuid4().hex[:8]}"
    record = {
        "schema_version": _SCHEMA_VERSION,
        "event_id": event_id,
        "created_at": now.isoformat(timespec="milliseconds"),
        "event_type": "cognition_prefill_discard",
        "provider": provider,
        "model": model,
        "feature": feature,
        "subfeature": subfeature,
        "prompt_snapshot_id": prompt_snapshot_id,
        "agent_run_id": agent_run_id,
        "context": _json_safe(context or {}),
        "retry_attempt": max(1, int(retry_attempt or 1)),
        "guard": {
            "similarity": round(float(similarity), 4),
            "matched_index": int(matched_index),
            "visible_cognitions_count": int(visible_cognitions_count or 0),
            "prefill_exclusions": list(prefill_exclusions or ()),
            "config": _json_safe(guard_config or {}),
        },
        "discarded": _text_payload(discarded_cognition),
        "matched": _text_payload(matched_cognition),
        "chosen_prefill": _text_payload(chosen_prefill),
    }

    try:
        root = _resolve_root_dir(normalized["root_dir"])
        day_dir = root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        target = day_dir / "cognition_prefill.jsonl"
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with _WRITE_LOCK:
            _rotate_if_large(target, normalized)
            with target.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.write("\n")
        _maybe_run_maintenance(root, normalized, current_path=target)
    except Exception:
        logger.debug("[discarded_response] failed to persist cognition discard", exc_info=True)
        return ""
    return event_id


def _text_payload(text: str) -> dict[str, Any]:
    value = str(text or "")
    return {
        "text": value,
        "chars": len(value),
        "sha256": hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest(),
    }


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


def _rotate_if_large(target: Path, cfg: dict[str, Any]) -> None:
    max_bytes = int(cfg.get("max_file_bytes") or 0)
    if max_bytes <= 0 or not target.exists():
        return
    try:
        if target.stat().st_size < max_bytes:
            return
        rotated = _next_rotated_path(target)
        with target.open("rb") as src, gzip.open(rotated, "wb") as dst:
            shutil.copyfileobj(src, dst)
        target.unlink()
    except Exception:
        logger.debug("[discarded_response] rotate failed: %s", target, exc_info=True)


def _next_rotated_path(target: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%H%M%S")
    base = target.with_name(f"{target.stem}-{stamp}.jsonl.gz")
    if not base.exists():
        return base
    for index in range(2, 1000):
        candidate = target.with_name(f"{target.stem}-{stamp}-{index}.jsonl.gz")
        if not candidate.exists():
            return candidate
    return target.with_name(f"{target.stem}-{stamp}-{int(time.time())}.jsonl.gz")


def _maybe_run_maintenance(root: Path, cfg: dict[str, Any], *, current_path: Path) -> None:
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


def _run_maintenance(root: Path, cfg: dict[str, Any], *, current_path: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        _compress_old_jsonl(root, cfg, current_path=current_path)
        _bundle_old_gzip_files(root, cfg)
        _delete_old_files(root, cfg, current_path=current_path)
        _enforce_total_size(root, cfg, current_path=current_path)
        _remove_empty_dirs(root)
    except Exception:
        logger.debug("[discarded_response] maintenance failed", exc_info=True)


def _compress_old_jsonl(root: Path, cfg: dict[str, Any], *, current_path: Path) -> None:
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
            logger.debug("[discarded_response] compress failed: %s", path, exc_info=True)


def _bundle_old_gzip_files(root: Path, cfg: dict[str, Any]) -> None:
    days = int(cfg.get("bundle_after_days", 14))
    if days <= 0 or not root.exists():
        return
    cutoff = time.time() - days * 86400
    archive_dir = root / "archive"
    for day_dir in root.iterdir():
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
            logger.debug("[discarded_response] bundle failed: %s", day_dir, exc_info=True)


def _delete_old_files(root: Path, cfg: dict[str, Any], *, current_path: Path) -> None:
    days = int(cfg.get("delete_after_days", 60))
    if days <= 0:
        return
    cutoff = time.time() - days * 86400
    for path in _iter_discard_files(root):
        if path == current_path:
            continue
        try:
            if path.stat().st_mtime <= cutoff:
                path.unlink()
        except Exception:
            logger.debug("[discarded_response] delete old file failed: %s", path, exc_info=True)


def _enforce_total_size(root: Path, cfg: dict[str, Any], *, current_path: Path) -> None:
    max_mb = int(cfg.get("max_total_mb", 512))
    if max_mb <= 0:
        return
    max_bytes = max_mb * 1024 * 1024
    files = []
    total = 0
    for path in _iter_discard_files(root):
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
            logger.debug("[discarded_response] size pruning failed: %s", path, exc_info=True)
        if total <= max_bytes:
            break


def _iter_discard_files(root: Path) -> list[Path]:
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
