"""retry.py — LLM 调用重试封装

提供 call_model_with_retry()：在 LLM 思考期间若有新消息到达，
自动丢弃本次结果（回滚 previous_cycle 状态）并重新调用一次，
确保 LLM 始终基于完整上下文做决策。

被丢弃的中间调用对外完全无痕（不写 DB、不入上下文、不更新 previous_cycle）。
"""

import asyncio
import logging
import time

import app_state
from .llm_core import call_model_and_process
from ..session import (
    get_bot_previous_cycle,
    set_bot_previous_cycle,
    get_bot_previous_cycle_time,
    set_bot_previous_cycle_time,
    get_bot_previous_tool_calls,
    set_bot_previous_tool_calls,
)

logger = logging.getLogger("AICQ.app")


async def call_model_with_retry(session, conv_key: str):
    """封装 LLM 调用（rate_limit + prefetch + call），含一次重调机会。

    若 LLM 思考期间有新消息到达（session.unread_count > 0），丢弃本次结果并回滚
    previous_cycle 状态，重新调用一次，确保 LLM 看到完整上下文。
    丢弃的中间结果不更新 previous_cycle 状态，也不写入 DB。

    返回 (result, grounding, system_prompt, chat_log_display, repaired, tool_calls_log, llm_elapsed)。
    """
    from ..prompt.quote_prefetch import prefetch_quoted_messages

    # 快照调用前的 previous_cycle 全局状态，供丢弃时回滚
    _snap_cycle = get_bot_previous_cycle()
    _snap_cycle_time = get_bot_previous_cycle_time()
    _snap_tool_calls = get_bot_previous_tool_calls()
    _snap_prev_json = getattr(session, "previous_cycle_json", None)

    _t0 = time.monotonic()
    await app_state.rate_limiter.acquire()
    await prefetch_quoted_messages(session, app_state.napcat_client)
    await session.prepare_memory_recall()
    result, grounding, system_prompt, chat_log_display, repaired, tool_calls_log = await asyncio.to_thread(
        call_model_and_process, session
    )

    _retry_enabled = app_state.config.get("generation", {}).get("retry_on_new_message", True)
    if _retry_enabled and result is not None and session.unread_count > 0 and not tool_calls_log:
        # LLM 思考期间有新消息到达，且本次未调用任何工具（有工具调用时 user prompt 已在每轮刷新，无需重调）
        # → 回滚 previous_cycle 状态并重新调用（仅一次）
        logger.info(
            "[retry] 会话 %s LLM 思考期间收到 %d 条新消息（无工具调用），丢弃本次结果重新调用",
            conv_key, session.unread_count,
        )
        set_bot_previous_cycle(_snap_cycle)
        set_bot_previous_cycle_time(_snap_cycle_time)
        set_bot_previous_tool_calls(_snap_tool_calls)
        session.previous_cycle_json = _snap_prev_json
        await app_state.rate_limiter.acquire()
        await prefetch_quoted_messages(session, app_state.napcat_client)
        await session.prepare_memory_recall()
        result, grounding, system_prompt, chat_log_display, repaired, tool_calls_log = await asyncio.to_thread(
            call_model_and_process, session
        )

    return result, grounding, system_prompt, chat_log_display, repaired, tool_calls_log, time.monotonic() - _t0
