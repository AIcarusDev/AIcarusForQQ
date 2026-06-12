"""Async cognition-flow compression worker."""

from __future__ import annotations

import asyncio
import logging

import app_state
from consciousness.flow import extract_summary_block
from database import save_adapter_contents
from llm.core.daemon_thread import run_in_daemon_thread
from llm.core.provider import build_compression_adapter_cfg, create_adapter
from runtime.emergency_reset import is_runtime_epoch_stale

from .config import normalize_generation_config
from .prompt import COMPRESSION_PROMPT_SYS_TEMPLATE

logger = logging.getLogger("AICQ.llm.compression")


def schedule_cognition_compression() -> None:
    """Freeze a ready compression job and ensure the serial worker is running."""
    flow = app_state.consciousness_flow
    if flow is None:
        return

    gen = normalize_generation_config(app_state.GEN)
    coverage_end = _queued_coverage_end(flow)
    job = flow.build_compression_job(
        gen["cognition_compression_trigger_rounds"],
        coverage_end=coverage_end,
    )
    if job is not None:
        pending = getattr(app_state, "cognition_compression_pending_jobs", None)
        if pending is None:
            pending = []
            app_state.cognition_compression_pending_jobs = pending
        pending.append(job)
        logger.info(
            "[compression] 已冻结意识流压缩任务: rounds=%d coverage_end=%d detected_at=%s",
            job.round_count,
            job.coverage_end_seq,
            job.detected_at,
        )
    elif not (getattr(app_state, "cognition_compression_pending_jobs", None) or []):
        return

    running = getattr(app_state, "cognition_compression_task", None)
    if running is not None and not running.done():
        return

    app_state.cognition_compression_task = asyncio.create_task(
        _compression_worker_loop(int(getattr(app_state, "runtime_reset_epoch", 0)))
    )


async def _compression_worker_loop(worker_epoch: int | None = None) -> None:
    """Run compression jobs serially, each based on the latest generated summary."""
    if worker_epoch is None:
        worker_epoch = int(getattr(app_state, "runtime_reset_epoch", 0))
    while True:
        if is_runtime_epoch_stale(worker_epoch):
            logger.info("[compression] 旧压缩 worker 已因紧急恢复失效 epoch=%s", worker_epoch)
            return
        async with app_state.llm_lock:
            if is_runtime_epoch_stale(worker_epoch):
                logger.info("[compression] 旧压缩 worker 已因紧急恢复失效 epoch=%s", worker_epoch)
                return
            flow = app_state.consciousness_flow
            if flow is None:
                return
            pending = getattr(app_state, "cognition_compression_pending_jobs", None) or []
            if not pending:
                gen = normalize_generation_config(app_state.GEN)
                job = flow.build_compression_job(
                    gen["cognition_compression_trigger_rounds"],
                    coverage_end=_queued_coverage_end(flow),
                )
                if job is None:
                    app_state.cognition_compression_inflight_job = None
                    return
                logger.info(
                    "[compression] 已冻结意识流压缩任务: rounds=%d coverage_end=%d detected_at=%s",
                    job.round_count,
                    job.coverage_end_seq,
                    job.detected_at,
                )
            else:
                job = pending.pop(0)
                app_state.cognition_compression_pending_jobs = pending
            app_state.cognition_compression_inflight_job = job
            task_xml = flow.render_compression_job(job)
        if job is None:
            return

        should_continue = await _run_cognition_compression(
            task_xml,
            job.coverage_end_seq,
            job.round_count,
            expected_epoch=worker_epoch,
        )
        if is_runtime_epoch_stale(worker_epoch):
            logger.info("[compression] 压缩结果返回时已过期，跳过旧 worker 清理")
            return
        app_state.cognition_compression_inflight_job = None
        if not should_continue:
            app_state.cognition_compression_pending_jobs = []
            return


def _queued_coverage_end(flow) -> int:
    coverage_end = flow.compression_frontier_end_seq
    inflight = getattr(app_state, "cognition_compression_inflight_job", None)
    if inflight is not None:
        coverage_end = max(coverage_end, inflight.coverage_end_seq)
    for job in getattr(app_state, "cognition_compression_pending_jobs", None) or []:
        coverage_end = max(coverage_end, job.coverage_end_seq)
    return coverage_end


async def _run_cognition_compression(
    task_xml: str,
    coverage_end_seq: int,
    round_count: int,
    expected_epoch: int | None = None,
) -> bool:
    if expected_epoch is None:
        expected_epoch = int(getattr(app_state, "runtime_reset_epoch", 0))
    logger.info(
        "[compression] 调度意识流压缩: rounds=%d coverage_end=%d",
        round_count,
        coverage_end_seq,
    )
    try:
        text = await run_in_daemon_thread(
            _call_compressor,
            task_xml,
            thread_name="cognition-compression",
        )
    except Exception:
        logger.warning("[compression] 意识流压缩调用失败", exc_info=True)
        return False

    if is_runtime_epoch_stale(expected_epoch):
        logger.info("[compression] 压缩调用完成但运行时已紧急恢复，丢弃旧摘要")
        return False

    summary = extract_summary_block(text or "")
    if not summary:
        logger.warning("[compression] 压缩输出缺少可用 summary")
        return False

    async with app_state.llm_lock:
        if is_runtime_epoch_stale(expected_epoch):
            logger.info("[compression] 入队摘要前检测到紧急恢复，丢弃旧摘要")
            return False
        queued = app_state.consciousness_flow.queue_compression_summary(
            summary,
            coverage_end_seq,
        )
    if queued:
        try:
            data, timestamps = app_state.consciousness_flow.dump()
            await save_adapter_contents("flow", data, timestamps)
        except Exception:
            logger.warning("[compression] 压缩摘要持久化失败", exc_info=True)
        logger.info("[compression] 已缓存意识流压缩摘要 coverage_end=%d", coverage_end_seq)
        try:
            from memory.archiver import schedule_compression_archive

            schedule_compression_archive(summary, coverage_end_seq)
        except Exception:
            logger.warning("[compression] 调度压缩摘要记忆归档失败", exc_info=True)
        return True
    else:
        logger.info("[compression] 压缩摘要已过期，跳过 coverage_end=%d", coverage_end_seq)
        return True


def _call_compressor(task_xml: str) -> str | None:
    compression_cfg = (
        getattr(app_state, "cognition_compression_cfg", None)
        or app_state.config.get("cognition_compression", {})
        or {}
    )
    adapter = getattr(app_state, "cognition_compression_adapter", None)
    if adapter is None:
        if compression_cfg.get("provider") and compression_cfg.get("model"):
            adapter = create_adapter(
                build_compression_adapter_cfg(app_state.config, compression_cfg)
            )
        else:
            adapter = create_adapter(app_state.config)
    gen = normalize_generation_config(app_state.GEN)
    compression_gen = compression_cfg.get("generation", {})
    comp_gen = dict(gen)
    comp_gen["temperature"] = compression_gen.get(
        "temperature",
        gen.get("cognition_compression_temperature", 0.3),
    )
    comp_gen["max_output_tokens"] = compression_gen.get(
        "max_output_tokens",
        gen.get("cognition_compression_max_output_tokens", 2000),
    )
    return adapter.call_simple_text(
        COMPRESSION_PROMPT_SYS_TEMPLATE,
        task_xml,
        comp_gen,
        log_tag="cognition_compression",
    )
