"""watcher_core.py — Watcher（窥屏意识）调度逻辑

在主模型选择 loop_control.break 后，后台周期性唤醒一个轻量模型，
由它决定要不要重新激活主意识投入对话。

流程：
  break → schedule_watcher() → 后台 run_watcher_loop() 每隔 interval 秒...
    └→ pass    → 继续睡，等下一轮
    └→ activate → 把 mood/think/intent 注入 session.watcher_nudge
                  → 调用主模型完整运行一轮（含 loop_control）
"""

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime

import app_state
from watcher_prompt import build_watcher_system_prompt

logger = logging.getLogger("AICQ.watcher")


# ══════════════════════════════════════════════════════════
#  内部：调用 watcher 模型
# ══════════════════════════════════════════════════════════

def _build_watcher_gen() -> dict:
    """构建 watcher 的生成参数（轻量、低温度、无工具轮次）。"""
    watcher_cfg = app_state.watcher_cfg
    gen = watcher_cfg.get("generation", {})
    return {
        "temperature": gen.get("temperature", 0.7),
        "max_output_tokens": gen.get("max_output_tokens", 800),
        "max_tool_rounds": 0,
    }


def _call_watcher_model(
    session,
    previous_cycle_result: dict | None = None,
    previous_cycle_time: float = 0.0,
    previous_cycle_source: str = "watcher",
) -> dict | None:
    """同步调用 watcher 模型，返回解析后的结果字典，失败返回 None。"""
    from schema import WATCHER_SCHEMA

    adapter = app_state.watcher_adapter
    if adapter is None:
        return None

    watcher_cfg = app_state.watcher_cfg
    model_name = watcher_cfg.get("model_name", watcher_cfg.get("model", "watcher"))

    system_prompt = build_watcher_system_prompt(
        persona=session._persona,
        qq_name=session._qq_name,
        qq_id=session._qq_id,
        model_name=model_name,
        conv_type=session.conv_type,
        conv_name=session.conv_name,
        conv_id=session.conv_id,
        now=datetime.now(session._timezone) if session._timezone else None,
        break_time=session.watcher_break_time,
        break_reason=session.watcher_break_reason,
        previous_cycle_result=previous_cycle_result,
        previous_cycle_time=previous_cycle_time,
        previous_cycle_source=previous_cycle_source,
    )

    # watcher 无工具调用，prompt_builder 忽略 tool_budget 参数
    def prompt_builder(tool_budget=None, rounds_used=0, max_rounds=None, tool_budget_suffix=""):
        return system_prompt

    chat_log = session.build_chat_log_xml()
    gen = _build_watcher_gen()

    result, _, _, _, _ = adapter.call(
        prompt_builder,
        chat_log,
        gen,
        WATCHER_SCHEMA,
        tool_declarations=[],
        tool_registry={},
    )
    return result


# ══════════════════════════════════════════════════════════
#  Watcher 后台循环
# ══════════════════════════════════════════════════════════

async def run_watcher_loop(
    session,
    conv_key: str,
    group_id,
    user_id,
) -> None:
    """Watcher 后台循环：周期性窥屏，决定是否激活主意识。

    退出条件：
    - session.watcher_active 被外部清除（被动消息激活了主意识）
    - 决策 activate（激活后自行退出）
    """
    watcher_cfg = app_state.watcher_cfg
    interval: float = float(watcher_cfg.get("interval", 60))
    jitter: float = float(watcher_cfg.get("interval_jitter", 10))

    watch_round = 0
    logger.info(
        "[watcher] 启动窥屏循环 conv=%s interval=%.0fs",
        conv_key, interval,
    )

    # 初始化 previous 状态：若首次进入窥屏（watcher_last_cycle 为空），继承聊天的最后一轮输出
    from session import get_bot_previous_cycle
    _prev_cycle = session.watcher_last_cycle
    _prev_cycle_time = session.watcher_last_cycle_time
    _prev_source = "watcher"
    if _prev_cycle is None:
        _prev_cycle = get_bot_previous_cycle()
        _prev_cycle_time = session.watcher_break_time
        _prev_source = "chat" if _prev_cycle else ""

    while session.watcher_active:
        sleep_time = interval + random.uniform(-jitter, jitter)
        sleep_time = max(10.0, sleep_time)
        logger.debug("[watcher] 等待 %.1fs 后窥屏 (conv=%s)", sleep_time, conv_key)

        try:
            await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("[watcher] 窥屏循环被取消 conv=%s", conv_key)
            break

        if not session.watcher_active:
            logger.info("[watcher] watcher_active 已清除，退出 conv=%s", conv_key)
            break

        # 意识正忙（正在处理其它会话），跳过本轮
        if app_state.consciousness_lock.locked():
            logger.info("[watcher] 意识正忙，本轮跳过 conv=%s", conv_key)
            continue

        watch_round += 1
        logger.info("[watcher] 第 %d 轮窥屏开始 conv=%s", watch_round, conv_key)

        _t0 = time.monotonic()
        _activated = False
        async with app_state.consciousness_lock:
            app_state.current_focus = conv_key
            try:
                try:
                    await app_state.rate_limiter.acquire()
                    result = await asyncio.to_thread(
                        _call_watcher_model,
                        session,
                        _prev_cycle,
                        _prev_cycle_time,
                        _prev_source,
                    )
                except Exception:
                    logger.exception("[watcher] 模型调用失败 conv=%s", conv_key)
                    continue

                logger.info("[watcher] 窥屏耗时 %.2fs conv=%s", time.monotonic() - _t0, conv_key)

                if result is None:
                    logger.warning("[watcher] 模型返回为空，跳过 conv=%s", conv_key)
                    continue

                # 持久化本轮结果，并更新内存中的上轮快照
                _cycle_id = uuid.uuid4().hex
                _now_ts = time.time()
                try:
                    from database import save_watcher_cycle
                    await save_watcher_cycle(
                        cycle_id=_cycle_id,
                        conv_type=session.conv_type,
                        conv_id=session.conv_id,
                        result=result,
                    )
                except Exception:
                    logger.warning("[watcher] 保存 watcher_cycle 失败 conv=%s", conv_key)
                session.watcher_last_cycle = result
                session.watcher_last_cycle_time = _now_ts
                _prev_cycle = result
                _prev_cycle_time = _now_ts
                _prev_source = "watcher"

                action = (result.get("decision") or {}).get("action", "pass")
                motivation = (result.get("decision") or {}).get("motivation", "")
                logger.info("[watcher] 决策=%s motivation=%s conv=%s", action, motivation, conv_key)

                if action == "activate":
                    session.watcher_active = False
                    session.watcher_nudge = {
                        "result": result,
                        "time_iso": datetime.utcfromtimestamp(_now_ts).isoformat() + "Z",
                    }
                    logger.info("[watcher] 决定激活主意识 conv=%s", conv_key)
                    await _activate_from_watcher(session, conv_key, group_id, user_id)
                    _activated = True
            finally:
                app_state.current_focus = None
        if _activated:
            break

    session.watcher_active = False
    session.watcher_task = None
    logger.info("[watcher] 窥屏循环结束 conv=%s (rounds=%d)", conv_key, watch_round)


# ══════════════════════════════════════════════════════════
#  激活主意识
# ══════════════════════════════════════════════════════════

async def _activate_from_watcher(
    session,
    conv_key: str,
    group_id,
    user_id,
) -> None:
    """watcher 决定 activate 后，完整运行一轮主意识（含 loop_control）。

    调用者需已持有 consciousness_lock 并设置 current_focus。
    """
    # 延迟导入，避免与 napcat_handler 循环依赖
    from napcat_handler import send_and_commit_bot_messages, _run_active_loop
    from llm_core import call_model_and_process
    from database import save_bot_turn

    _t0 = time.monotonic()
    try:
        await app_state.rate_limiter.acquire()
        result, _, _, _, _, tool_calls_log = await asyncio.to_thread(
            call_model_and_process, session
        )
    except Exception:
        logger.exception("[watcher] 激活主意识失败 conv=%s", conv_key)
        return

    logger.info("[watcher] 主意识响应耗时 %.2fs conv=%s", time.monotonic() - _t0, conv_key)

    if result is None:
        logger.warning("[watcher] 主意识返回为空 conv=%s", conv_key)
        return

    await send_and_commit_bot_messages(
        session, result, group_id, user_id, time.monotonic() - _t0, conv_key,
    )
    await save_bot_turn(
        turn_id=uuid.uuid4().hex,
        conv_type=session.conv_type,
        conv_id=session.conv_id,
        result=result,
        tool_calls_log=tool_calls_log,
    )
    await _run_active_loop(session, conv_key, group_id, user_id, result)


# ══════════════════════════════════════════════════════════
#  公共接口
# ══════════════════════════════════════════════════════════

def schedule_watcher(session, conv_key: str, group_id, user_id) -> None:
    """在 loop_control.break 后调度 watcher 后台任务。"""
    if not app_state.watcher_cfg.get("enabled", False):
        return
    if app_state.watcher_adapter is None:
        return

    # 如果旧 watcher 还在跑，先取消
    if session.watcher_task and not session.watcher_task.done():
        logger.info("[watcher] 取消旧 watcher 任务 conv=%s", conv_key)
        session.watcher_active = False
        session.watcher_task.cancel()

    # 重置上轮 watcher 快照，让新一轮继承本次 break 时的聊天输出而非上次 watcher 输出
    session.watcher_last_cycle = None
    session.watcher_last_cycle_time = 0.0

    session.watcher_active = True
    session.watcher_task = asyncio.create_task(
        run_watcher_loop(session, conv_key, group_id, user_id)
    )
    logger.info("[watcher] 已调度窥屏任务 conv=%s", conv_key)


def stop_watcher(session) -> None:
    """取消会话的 watcher 任务（被动消息激活主意识时调用）。"""
    if session.watcher_task and not session.watcher_task.done():
        session.watcher_active = False
        session.watcher_task.cancel()
        session.watcher_task = None


def stop_all_watchers() -> None:
    """停止所有会话的 watcher 任务（意识介入时调用，确保单一意识流）。"""
    from session import sessions
    for s in sessions.values():
        if s.watcher_task and not s.watcher_task.done():
            s.watcher_active = False
            s.watcher_task.cancel()
            s.watcher_task = None
