"""confidence_scheduler.py — 置信度衰减后台定时任务 (Phase 3A)

每隔 decay_interval_hours 小时扫描一次 MemoryTriples，对超过
idle_days_threshold 天未被访问的记忆执行置信度衰减。

配置键（config.yaml → memory.scheduler）：
  decay_interval_hours : float  每次衰减间隔（小时），默认 6
  idle_days_threshold  : float  多少天未访问才触发衰减，默认 7
  decay_rate           : float  每次降低幅度，默认 0.01
  min_confidence       : float  置信度下限，默认 0.05
"""

import asyncio
import logging

logger = logging.getLogger("AICQ.scheduler")


async def start_confidence_scheduler() -> None:
    """后台循环任务，由 lifecycle.startup() 通过 asyncio.create_task() 启动。"""
    import app_state
    from database import decay_triple_confidence

    while True:
        cfg = {}
        if hasattr(app_state, "config"):
            cfg = app_state.config.get("memory", {}).get("scheduler", {})

        interval_hours = float(cfg.get("decay_interval_hours", 6.0))
        idle_days = float(cfg.get("idle_days_threshold", 7.0))
        decay_rate = float(cfg.get("decay_rate", 0.01))
        min_confidence = float(cfg.get("min_confidence", 0.05))

        # 先睡眠再执行，避免启动时立即降权（刚恢复的记忆 last_accessed 已是最新）
        await asyncio.sleep(interval_hours * 3600)

        try:
            count = await decay_triple_confidence(
                min_confidence=min_confidence,
                decay_rate=decay_rate,
                idle_days_threshold=idle_days,
            )
            if count:
                logger.info("[scheduler] 置信度衰减完成：%d 条记忆已降权", count)
            else:
                logger.debug("[scheduler] 置信度衰减：无需降权")
        except Exception:
            logger.exception("[scheduler] 置信度衰减异常，将在下次间隔后重试")
