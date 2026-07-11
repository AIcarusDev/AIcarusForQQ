"""Sleep-time Memory consolidation orchestration."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from .consolidation import (
    EpisodeCandidateConsolidationReport,
    PreprocessReport,
    run_episode_candidate_consolidation,
    run_preprocessing,
)
from .summary_worker import SummaryRefreshReport, run_summary_refresh_worker

_SLEEP_MAINTENANCE_TIMEOUT_DEFAULT = 300.0


@dataclass(frozen=True)
class SleepMaintenanceConfig:
    preprocess_limit: int
    algorithmic_clustering_enabled: bool
    max_episode_candidates: int
    summary_max_inputs: int
    timeout_seconds: float
    dry_run: bool
    solidify: bool

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "SleepMaintenanceConfig":
        return cls(
            preprocess_limit=_bounded_int(raw.get("preprocess_limit", 5000), 5000, 100, 50_000),
            algorithmic_clustering_enabled=bool(raw.get("algorithmic_clustering_enabled", False)),
            max_episode_candidates=_bounded_int(
                raw.get("max_episode_candidates_per_sleep", 100), 100, 1, 1000
            ),
            summary_max_inputs=_bounded_int(raw.get("summary_max_inputs_per_sleep", 32), 32, 1, 500),
            timeout_seconds=_bounded_float(
                raw.get("sleep_maintenance_timeout_seconds", _SLEEP_MAINTENANCE_TIMEOUT_DEFAULT),
                _SLEEP_MAINTENANCE_TIMEOUT_DEFAULT,
                0.0,
                3600.0,
            ),
            dry_run=bool(raw.get("dry_run", True)),
            solidify=bool(raw.get("solidify", False)),
        )


@dataclass(frozen=True)
class EpisodeCandidateConsolidationPhase:
    report: EpisodeCandidateConsolidationReport

    @property
    def cluster_ids(self) -> tuple[str, ...]:
        return self.report.cluster_ids_written

    def log_fields(self) -> dict[str, Any]:
        return {
            "pending_episode_candidates": self.report.pending_candidates_loaded,
            "episode_clusters_written": self.report.clusters_written,
        }


@dataclass(frozen=True)
class SummaryRefreshPhase:
    report: SummaryRefreshReport = field(default_factory=SummaryRefreshReport)

    def log_fields(self) -> dict[str, Any]:
        return {
            "summary_tasks_queued": self.report.summary_tasks_queued,
            "summary_done": self.report.summary_tasks_done,
            "summaries_ready": self.report.summaries_ready,
        }


@dataclass(frozen=True)
class SleepMaintenanceReport:
    ok: bool
    trigger: str
    dry_run: bool
    solidify: bool
    preprocess: PreprocessReport
    episode_candidate_consolidation: EpisodeCandidateConsolidationPhase
    summary_refresh: SummaryRefreshPhase

    def log_summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "solidify": self.solidify,
            "algorithmic_clustering": self.preprocess.algorithmic_clustering_enabled,
            "algorithmic_clusters": len(self.preprocess.algorithmic_cluster_ids),
            **self.episode_candidate_consolidation.log_fields(),
            **self.summary_refresh.log_fields(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "trigger": self.trigger,
            "dry_run": self.dry_run,
            "solidify": self.solidify,
            "preprocess": self.preprocess.to_dict(),
            "episode_candidate_consolidation": self.episode_candidate_consolidation.report.to_dict(),
            "summary_worker": self.summary_refresh.report.to_dict(),
            "maintenance_summary": self.log_summary(),
        }


def run_sleep_memory_maintenance(
    db_path: str | os.PathLike[str] | None = None,
    *,
    trigger: str = "sleep",
    config: dict[str, Any] | None = None,
    pause_event: Any = None,
) -> dict[str, Any]:
    """Run one bounded memory maintenance pass after sleep."""

    cfg = SleepMaintenanceConfig.from_raw(_memory_consolidation_cfg(config))
    path = os.fspath(db_path) if db_path else _default_db_path()
    con = sqlite3.connect(path, timeout=30.0)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        return _run_sleep_memory_maintenance_on_connection(
            con,
            trigger=trigger,
            cfg=cfg,
            pause_event=pause_event,
        ).to_dict()
    finally:
        con.close()


def _run_sleep_memory_maintenance_on_connection(
    con: sqlite3.Connection,
    *,
    trigger: str,
    cfg: SleepMaintenanceConfig,
    pause_event: Any = None,
) -> SleepMaintenanceReport:
    started_ms = int(time.time() * 1000)
    summary_deadline_ms = None if cfg.timeout_seconds <= 0 else started_ms + int(cfg.timeout_seconds * 1000)
    should_continue = None if pause_event is None else lambda: not pause_event.is_set()

    preprocess = run_preprocessing(
        con,
        limit=cfg.preprocess_limit,
        trigger=trigger,
        algorithmic_clustering_enabled=cfg.algorithmic_clustering_enabled,
    )
    con.commit()
    candidate_phase = EpisodeCandidateConsolidationPhase(
        run_episode_candidate_consolidation(
            con,
            max_candidates=cfg.max_episode_candidates,
            dry_run=cfg.dry_run,
            solidify=cfg.solidify,
        )
    )
    con.commit()
    summary_phase = SummaryRefreshPhase(
        run_summary_refresh_worker(
            con,
            max_inputs=cfg.summary_max_inputs,
            cluster_ids=preprocess.algorithmic_cluster_ids,
            priority_cluster_ids=candidate_phase.cluster_ids,
            deadline_ms=summary_deadline_ms,
            should_continue=should_continue,
        )
    )
    con.commit()
    return SleepMaintenanceReport(
        ok=True,
        trigger=trigger,
        dry_run=not (cfg.solidify and not cfg.dry_run),
        solidify=cfg.solidify,
        preprocess=preprocess,
        episode_candidate_consolidation=candidate_phase,
        summary_refresh=summary_phase,
    )


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
    "EpisodeCandidateConsolidationPhase",
    "SleepMaintenanceConfig",
    "SleepMaintenanceReport",
    "SummaryRefreshPhase",
    "run_sleep_memory_maintenance",
    "run_sleep_memory_maintenance_async",
]
