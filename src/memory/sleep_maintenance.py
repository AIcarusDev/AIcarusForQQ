"""Sleep-time Memory consolidation orchestration."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from typing import Any

from .consolidation import run_mount_consolidation, run_preprocessing
from .summary_worker import run_summary_refresh_worker

_SLEEP_MAINTENANCE_TIMEOUT_DEFAULT = 300.0


def run_sleep_memory_maintenance(
    db_path: str | os.PathLike[str] | None = None,
    *,
    trigger: str = "sleep",
    config: dict[str, Any] | None = None,
    pause_event: Any = None,
) -> dict[str, Any]:
    """Run one bounded memory maintenance pass after sleep."""

    cfg = _memory_consolidation_cfg(config)
    path = os.fspath(db_path) if db_path else _default_db_path()
    con = sqlite3.connect(path, timeout=30.0)
    try:
        started_ms = int(time.time() * 1000)
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        preprocess_limit = _bounded_int(cfg.get("preprocess_limit", 5000), 5000, 100, 50_000)
        max_mounts = _bounded_int(cfg.get("max_mounts_per_sleep", 100), 100, 1, 1000)
        accept_threshold = _bounded_float(cfg.get("accept_threshold", 0.62), 0.62, 0.0, 1.0)
        summary_max_inputs = _bounded_int(cfg.get("summary_max_inputs_per_sleep", 32), 32, 1, 500)
        summary_max_bootstrap_clusters = _bounded_int(cfg.get("summary_max_bootstrap_clusters_per_sleep", 64), 64, 1, 1000)
        timeout_seconds = _bounded_float(
            cfg.get("sleep_maintenance_timeout_seconds", _SLEEP_MAINTENANCE_TIMEOUT_DEFAULT),
            _SLEEP_MAINTENANCE_TIMEOUT_DEFAULT,
            0.0,
            3600.0,
        )
        summary_deadline_ms = None if timeout_seconds <= 0 else started_ms + int(timeout_seconds * 1000)
        dry_run = bool(cfg.get("dry_run", True))
        solidify = bool(cfg.get("solidify", False))

        preprocess = run_preprocessing(
            con,
            limit=preprocess_limit,
            trigger=trigger,
        )
        con.commit()
        mounts = run_mount_consolidation(
            con,
            max_mounts=max_mounts,
            dry_run=dry_run,
            solidify=solidify,
            accept_threshold=accept_threshold,
        )
        con.commit()
        summaries = run_summary_refresh_worker(
            con,
            max_inputs=summary_max_inputs,
            max_bootstrap_clusters=summary_max_bootstrap_clusters,
            priority_packet_ids=mounts.get("summary_refresh_packet_ids_queued") or (),
            priority_cluster_ids=mounts.get("local_cluster_ids_written") or (),
            deadline_ms=summary_deadline_ms,
            should_continue=(None if pause_event is None else lambda: not pause_event.is_set()),
        )
        con.commit()
        return {
            "ok": True,
            "trigger": trigger,
            "dry_run": not (solidify and not dry_run),
            "solidify": solidify,
            "preprocess": preprocess,
            "mount_consolidation": mounts,
            "summary_worker": summaries,
        }
    finally:
        con.close()


async def run_sleep_memory_maintenance_async(
    db_path: str | os.PathLike[str] | None = None,
    *,
    trigger: str = "sleep",
    config: dict[str, Any] | None = None,
    pause_event: Any = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        run_sleep_memory_maintenance,
        db_path,
        trigger=trigger,
        config=config,
        pause_event=pause_event,
    )


def _memory_consolidation_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, dict):
        memory = config.get("memory", {}) if isinstance(config.get("memory", {}), dict) else {}
        cfg = memory.get("consolidation", {}) if isinstance(memory.get("consolidation", {}), dict) else {}
        return dict(cfg)
    try:
        import app_state

        cfg = getattr(app_state, "memory_consolidation_cfg", None)
        if isinstance(cfg, dict) and cfg:
            return dict(cfg)
        root = getattr(app_state, "config", {}) or {}
        memory = root.get("memory", {}) if isinstance(root, dict) else {}
        return dict(memory.get("consolidation", {}) or {}) if isinstance(memory, dict) else {}
    except Exception:
        return {}


def _default_db_path() -> str:
    import database

    return str(database.DB_PATH)


def _bounded_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        item = int(value)
    except (TypeError, ValueError):
        item = default
    return max(low, min(high, item))


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        item = float(value)
    except (TypeError, ValueError):
        item = default
    return max(low, min(high, item))


__all__ = [
    "run_sleep_memory_maintenance",
    "run_sleep_memory_maintenance_async",
]
