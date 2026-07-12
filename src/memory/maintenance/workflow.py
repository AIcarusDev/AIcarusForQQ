"""Trigger-driven memory maintenance orchestration."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from .preprocessing import (
    CandidateStorylineConsolidationReport,
    PreprocessReport,
    run_candidate_storyline_consolidation,
    run_preprocessing,
)
from ..storyline_synthesis.workflow import StorylineSynthesisReport, run_storyline_synthesis

_MAINTENANCE_TIMEOUT_DEFAULT = 300.0


@dataclass(frozen=True)
class MemoryMaintenanceConfig:
    preprocess_limit: int
    algorithmic_storyline_enabled: bool
    max_candidate_storylines: int
    storyline_synthesis_max_inputs: int
    timeout_seconds: float
    dry_run: bool
    solidify: bool

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "MemoryMaintenanceConfig":
        return cls(
            preprocess_limit=_bounded_int(raw.get("preprocess_limit", 5000), 5000, 100, 50_000),
            algorithmic_storyline_enabled=bool(raw.get("algorithmic_storyline_enabled", False)),
            max_candidate_storylines=_bounded_int(
                raw.get("max_candidate_storylines_per_maintenance", 100), 100, 1, 1000
            ),
            storyline_synthesis_max_inputs=_bounded_int(
                raw.get("storyline_synthesis_max_inputs_per_maintenance", 32), 32, 1, 500
            ),
            timeout_seconds=_bounded_float(
                raw.get("maintenance_timeout_seconds", _MAINTENANCE_TIMEOUT_DEFAULT),
                _MAINTENANCE_TIMEOUT_DEFAULT,
                0.0,
                3600.0,
            ),
            dry_run=bool(raw.get("dry_run", True)),
            solidify=bool(raw.get("solidify", False)),
        )


@dataclass(frozen=True)
class CandidateStorylineConsolidationPhase:
    report: CandidateStorylineConsolidationReport

    @property
    def storyline_ids(self) -> tuple[str, ...]:
        return self.report.storyline_ids_written

    def log_fields(self) -> dict[str, Any]:
        return {
            "pending_candidate_storylines": self.report.pending_candidate_storylines_loaded,
            "candidate_storylines_written": self.report.candidate_storylines_written,
        }


@dataclass(frozen=True)
class StorylineSynthesisPhase:
    report: StorylineSynthesisReport = field(default_factory=StorylineSynthesisReport)

    def log_fields(self) -> dict[str, Any]:
        return {
            "summary_tasks_queued": self.report.summary_tasks_queued,
            "summary_done": self.report.summary_tasks_done,
            "summaries_ready": self.report.summaries_ready,
        }


@dataclass(frozen=True)
class MemoryMaintenanceReport:
    ok: bool
    trigger: str
    dry_run: bool
    solidify: bool
    preprocess: PreprocessReport
    candidate_storyline_consolidation: CandidateStorylineConsolidationPhase
    storyline_synthesis: StorylineSynthesisPhase

    def log_summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "solidify": self.solidify,
            "algorithmic_storyline": self.preprocess.algorithmic_storyline_enabled,
            "algorithmic_storylines": len(self.preprocess.algorithmic_storyline_ids),
            **self.candidate_storyline_consolidation.log_fields(),
            **self.storyline_synthesis.log_fields(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "trigger": self.trigger,
            "dry_run": self.dry_run,
            "solidify": self.solidify,
            "preprocess": self.preprocess.to_dict(),
            "candidate_storyline_consolidation": self.candidate_storyline_consolidation.report.to_dict(),
            "storyline_synthesis": self.storyline_synthesis.report.to_dict(),
            "maintenance_summary": self.log_summary(),
        }


def run_memory_maintenance(
    db_path: str | os.PathLike[str] | None = None,
    *,
    trigger: str = "sleep",
    config: dict[str, Any] | None = None,
    pause_event: Any = None,
) -> dict[str, Any]:
    """Run one bounded memory maintenance pass after sleep."""

    cfg = MemoryMaintenanceConfig.from_raw(_memory_processing_cfg(config))
    path = os.fspath(db_path) if db_path else _default_db_path()
    con = sqlite3.connect(path, timeout=30.0)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=30000")
        return _run_memory_maintenance_on_connection(
            con,
            trigger=trigger,
            cfg=cfg,
            pause_event=pause_event,
        ).to_dict()
    finally:
        con.close()


def _run_memory_maintenance_on_connection(
    con: sqlite3.Connection,
    *,
    trigger: str,
    cfg: MemoryMaintenanceConfig,
    pause_event: Any = None,
) -> MemoryMaintenanceReport:
    started_ms = int(time.time() * 1000)
    summary_deadline_ms = None if cfg.timeout_seconds <= 0 else started_ms + int(cfg.timeout_seconds * 1000)
    should_continue = None if pause_event is None else lambda: not pause_event.is_set()

    preprocess = run_preprocessing(
        con,
        limit=cfg.preprocess_limit,
        trigger=trigger,
        algorithmic_storyline_enabled=cfg.algorithmic_storyline_enabled,
    )
    con.commit()
    candidate_phase = CandidateStorylineConsolidationPhase(
        run_candidate_storyline_consolidation(
            con,
            max_candidate_storylines=cfg.max_candidate_storylines,
            dry_run=cfg.dry_run,
            solidify=cfg.solidify,
        )
    )
    con.commit()
    synthesis_phase = StorylineSynthesisPhase(
        run_storyline_synthesis(
            con,
            max_inputs=cfg.storyline_synthesis_max_inputs,
            storyline_ids=preprocess.algorithmic_storyline_ids,
            priority_storyline_ids=candidate_phase.storyline_ids,
            deadline_ms=summary_deadline_ms,
            should_continue=should_continue,
        )
    )
    con.commit()
    return MemoryMaintenanceReport(
        ok=True,
        trigger=trigger,
        dry_run=not (cfg.solidify and not cfg.dry_run),
        solidify=cfg.solidify,
        preprocess=preprocess,
        candidate_storyline_consolidation=candidate_phase,
        storyline_synthesis=synthesis_phase,
    )


async def run_memory_maintenance_async(
    db_path: str | os.PathLike[str] | None = None,
    *,
    trigger: str = "sleep",
    config: dict[str, Any] | None = None,
    pause_event: Any = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        run_memory_maintenance,
        db_path,
        trigger=trigger,
        config=config,
        pause_event=pause_event,
    )


def _memory_processing_cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, dict):
        memory = config.get("memory", {}) if isinstance(config.get("memory", {}), dict) else {}
        cfg = memory.get("processing", {}) if isinstance(memory.get("processing", {}), dict) else {}
        return dict(cfg)
    try:
        import app_state

        cfg = getattr(app_state, "memory_processing_cfg", None)
        if isinstance(cfg, dict) and cfg:
            return dict(cfg)
        root = getattr(app_state, "config", {}) or {}
        memory = root.get("memory", {}) if isinstance(root, dict) else {}
        return dict(memory.get("processing", {}) or {}) if isinstance(memory, dict) else {}
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
    "CandidateStorylineConsolidationPhase",
    "MemoryMaintenanceConfig",
    "MemoryMaintenanceReport",
    "StorylineSynthesisPhase",
    "run_memory_maintenance",
    "run_memory_maintenance_async",
]
