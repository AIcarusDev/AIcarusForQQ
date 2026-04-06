"""retry.py — LLM 调用重试封装

提供 call_model_with_retry()：在 LLM 思考期间若有新消息到达，
自动丢弃本次结果并重新调用一次，确保 LLM 始终基于完整上下文做决策。
"""

import asyncio
import logging
import time

import app_state
from .llm_core import call_model_and_process

logger = logging.getLogger("AICQ.app")


async def call_model_with_retry(session, conv_key: str) -> tuple:
    """封装 LLM 调用（rate_limit + prefetch + call），含一次重调机会。

    若 LLM 思考期间有新消息到达（session.unread_count > 0）且本次无工具调用，
    丢弃本次结果重新调用一次。

    返回 (loop_action, tool_calls_log, system_prompt, elapsed)。
    """
    from ..prompt.quote_prefetch import prefetch_quoted_messages

    _t0 = time.monotonic()
    await app_state.rate_limiter.acquire()
    await prefetch_quoted_messages(session, app_state.napcat_client)
    loop_action, tool_calls_log, system_prompt = await asyncio.to_thread(
        call_model_and_process, session
    )

    _retry_enabled = app_state.config.get("generation", {}).get("retry_on_new_message", True)
    if _retry_enabled and loop_action is not None and session.unread_count > 0 and not tool_calls_log:
        # LLM 思考期间有新消息到达，且本次未调用任何工具
        # → 重新调用（仅一次）；_contents 中暂存的历史在下次调用时会被 user msg 刷新
        logger.info(
            "[retry] 会话 %s LLM 思考期间收到 %d 条新消息（无工具调用），丢弃本次结果重新调用",
            conv_key, session.unread_count,
        )
        await app_state.rate_limiter.acquire()
        await prefetch_quoted_messages(session, app_state.napcat_client)
        loop_action, tool_calls_log, system_prompt = await asyncio.to_thread(
            call_model_and_process, session
        )

    return loop_action, tool_calls_log, system_prompt, time.monotonic() - _t0
