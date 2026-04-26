"""retry.py — LLM 调用重试封装

提供 call_model_with_retry()：消费 provider 返回的内部重调信号。

- 第 1 轮响应后若检测到新消息，丢弃本次结果并重调一次
- 模型若完全未调用工具，将本轮视作无事发生并重调有限次数
"""

import asyncio
import logging
import time

import app_state
from .llm_core import call_model_and_process
from .provider import (
    LLMCallFailed,
    RETRY_ON_EMPTY_TOOL_CALL_ACTION,
    RETRY_ON_NEW_MESSAGE_ACTION,
)

logger = logging.getLogger("AICQ.llm.retry")

EMPTY_TOOL_CALL_MAX_RETRIES = 2


async def _call_model_once(session, *, allow_retry_on_new_message: bool = True):
    from ..prompt.quote_prefetch import prefetch_quoted_messages

    # 模型调用前做 FTS5 记忆召回 + 事件召回，结果存入 session
    try:
        await session.prepare_memory_recall()
    except Exception:
        logger.warning("[retry] prepare_memory_recall 失败，本轮跳过召回", exc_info=True)
    await app_state.rate_limiter.acquire()
    await prefetch_quoted_messages(session, app_state.napcat_client)
    return await asyncio.to_thread(
        call_model_and_process,
        session,
        allow_retry_on_new_message=allow_retry_on_new_message,
    )


async def call_model_with_retry(session, conv_key: str) -> tuple:
    """封装 LLM 调用（rate_limit + prefetch + call），消费内部重调信号。

    - provider 第 1 轮响应后若发现思考期间有新消息，会返回内部重调信号
    - 因新消息触发的整轮重调最多只发生一次；重调后的调用直接保留本轮结果
    - provider 若模型完全未调用工具，会返回空工具调用重调信号
    - 空工具调用连续超过上限后，抛出 LLMCallFailed，避免误入 sleep

    返回 (loop_action, tool_calls_log, system_prompt, elapsed)。
    """
    _t0 = time.monotonic()
    empty_tool_call_retries = 0
    allow_retry_on_new_message = True

    while True:
        loop_action, tool_calls_log, system_prompt = await _call_model_once(
            session,
            allow_retry_on_new_message=allow_retry_on_new_message,
        )
        action = (loop_action or {}).get("action")

        if action == RETRY_ON_NEW_MESSAGE_ACTION:
            if not allow_retry_on_new_message:
                raise LLMCallFailed("新消息重调已禁用后仍收到内部重调信号")
            allow_retry_on_new_message = False
            logger.info(
                "[retry] 会话 %s LLM 思考期间收到新消息，丢弃本次结果重新调用",
                conv_key,
            )
            continue

        if action == RETRY_ON_EMPTY_TOOL_CALL_ACTION:
            if empty_tool_call_retries >= EMPTY_TOOL_CALL_MAX_RETRIES:
                raise LLMCallFailed(
                    f"模型连续 {EMPTY_TOOL_CALL_MAX_RETRIES + 1} 次未调用任何工具，已中止本次 activation"
                )
            empty_tool_call_retries += 1
            logger.warning(
                "[retry] 会话 %s 第 %d 次遇到空工具调用，丢弃本次结果并重调",
                conv_key,
                empty_tool_call_retries,
            )
            continue

        return loop_action, tool_calls_log, system_prompt, time.monotonic() - _t0
