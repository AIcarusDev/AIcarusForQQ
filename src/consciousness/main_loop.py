"""consciousness/main_loop.py — 机器人意识主循环

常驻 asyncio task，永远运行，直到 ``app_state.shutdown_event`` 被置位。

每一 round = 一次 LLM 调用 + 本轮工具的真正执行 + 持久化。
sleep/wait/shift 是普通的耗时工具，分别由其 handler 内部阻塞 / 修改全局焦点；
它们与 send_message、web_search 等工具语义上**完全等价**。

主循环本身没有任何 if-action 分派；它就是：

    while not shutdown:
        session = get(current_focus)
        result  = call_one_round(session)
        persist(result)
        # 立刻再来一次

启动时若 ``current_focus`` 为空（数据库无记录），则等 ``first_input_event``。
"""

import asyncio
import logging
import time as _time
import uuid

import app_state
from database import save_adapter_contents, save_bot_turn
from llm.core.daemon_thread import run_in_daemon_thread
from llm.session import get_or_create_session
from llm.core.provider import LLMCallFailed, RoundResult
from llm.prompt.user_prompt_builder import build_main_user_prompt
from tools import build_tools

from .flow import ToolCall, ToolResponse

logger = logging.getLogger("AICQ.consciousness.main")

# 模型连续违规（不调任何工具）时强制注入的兜底休眠时长（分钟）
EMPTY_TOOL_CALL_FALLBACK_DURATION = 60


# ── 内部辅助 ────────────────────────────────────────────────────────────────

def _restore_latent_tools_from_flow(tool_collection) -> None:
    """根据意识流推断仍应保持可用的潜伏工具，并就地激活。"""
    latent_names = set(tool_collection.latent_names())
    if not latent_names:
        return
    flow = app_state.consciousness_flow
    if flow is None or flow.round_count <= 0:
        return
    recoverable = flow.get_recoverable_latent_tool_names(latent_names)
    if not recoverable:
        return
    restored: list[str] = []
    for name in list(tool_collection.latent_names()):
        if name in recoverable and tool_collection.activate(name) is not None:
            restored.append(name)
    if restored:
        logger.info("[main] 从意识流恢复潜伏工具: %s", ", ".join(restored))


def _maybe_reset_chat_window_view(session, conv_key: str) -> None:
    """跨会话切换时重置目标会话的历史浏览视口。"""
    prev = app_state.last_active_session
    if conv_key and prev and prev != conv_key and session.is_browsing_history():
        logger.info("[main] 焦点曾切换 (%s → %s)，重置目标会话的历史浏览视口", prev, conv_key)
        session.reset_chat_window_view()
    if conv_key:
        app_state.last_active_session = conv_key


def _build_tool_collection(session):
    """每 round 重建工具集（保证 system prompt / 工具白名单与当前焦点一致）。"""
    return build_tools(
        app_state.config,
        napcat_client=app_state.napcat_client,
        group_id=session.conv_id if session.conv_type == "group" else None,
        user_id=int(session.conv_id) if session.conv_type == "private" else None,
        session=session,
        vision_bridge=(
            app_state.vision_bridge
            if (
                app_state.vision_bridge
                and app_state.vision_bridge.enabled
                and not app_state.config.get("vision", True)
            )
            else None
        ),
        provider=app_state.adapter.provider,
    )


async def _persist_round(session, conv_key: str, result: RoundResult) -> None:
    """把本 round 的简要摘要写入 bot_turns 并触发意识流持久化。"""
    try:
        # NOTE: bot_turns.result 字段在新架构下不再有 action 语义，仅作可读摘要
        summary = {
            "tools": [c["function"] for c in result.tool_calls_log],
            "tokens": {"in": result.prompt_tokens, "out": result.output_tokens},
        }
        await save_bot_turn(
            turn_id=uuid.uuid4().hex,
            conv_type=session.conv_type,
            conv_id=session.conv_id,
            result=summary,
            tool_calls_log=result.tool_calls_log,
        )
    except Exception:
        logger.warning("[main] save_bot_turn 失败 conv=%s", conv_key, exc_info=True)
    # 持久化意识流（重启后可恢复）
    try:
        c_data, ts_data = app_state.consciousness_flow.dump()
        asyncio.create_task(save_adapter_contents("flow", c_data, ts_data))
    except Exception:
        logger.warning("[main] 意识流持久化失败", exc_info=True)


async def _synthesize_fallback_sleep(session) -> None:
    """模型连续违规时合成一个 sleep 调用：直接执行 + 写入意识流。"""
    flow = app_state.consciousness_flow
    duration = EMPTY_TOOL_CALL_FALLBACK_DURATION
    motivation = "（模型连续未调用任何工具，强制休眠以避免空转）"
    call_id = f"fallback-sleep-{uuid.uuid4().hex[:8]}"
    if flow:
        max_rounds = app_state.GEN.get("llm_contents_max_rounds", 10)
        flow.prune(max_rounds)

    from tools.sleep.sleep import build_sleep_result, sleep_until_woken
    logger.warning("[main] 模型违规兜底：注入 sleep(duration=%dm)", duration)
    sleep_started_at = _time.monotonic()
    reason = await sleep_until_woken(session, duration * 60)
    result = build_sleep_result(
        session,
        elapsed=round(_time.monotonic() - sleep_started_at),
        reason=reason,
    )
    if flow:
        flow.append_round(
            [ToolCall(name="sleep", args={"duration": duration, "motivation": motivation}, call_id=call_id)],
            [ToolResponse(name="sleep", response=result, call_id=call_id)],
        )


# ── 单 round 执行（含 retry 语义） ─────────────────────────────────────────

async def _run_one_round(session, conv_key: str) -> RoundResult:
    """跑一个 round，处理模型违规重调。

    - 模型一次工具都没调 → 重调一次；仍然不调 → 合成兜底 sleep。
    """
    from llm.prompt.quote_prefetch import prefetch_quoted_messages

    # 清理上一轮残留的 wait race-window 标记：本 round 即将把所有未读消息
    # 喂给 LLM，模型已经"看到"它，不应再用它去提前唤醒下一次 wait
    # （否则会出现 wait 一启动就 elapsed=0.0s 被秒触发的空转）
    session.pending_early_trigger = None

    await prefetch_quoted_messages(session, app_state.napcat_client)

    tool_collection = _build_tool_collection(session)
    _restore_latent_tools_from_flow(tool_collection)

    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(
            activated_names=activated_names, latent_names=latent_names
        )

    chat_log = build_main_user_prompt(session)

    await app_state.rate_limiter.acquire()
    async with app_state.llm_lock:
        result = await run_in_daemon_thread(
            app_state.adapter.call_one_round,
            system_prompt_builder,
            chat_log,
            app_state.GEN,
            tool_collection,
            app_state.consciousness_flow,
            thread_name="main-llm-round",
        )

        # ── 模型违规（不调任何工具）重调 1 次，再失败就硬塞 sleep ────────
        if not result.failed and not result.had_tool_call:
            logger.warning("[main] 模型未调任何工具，重调一次 conv=%s", conv_key)
            chat_log = build_main_user_prompt(session)
            tool_collection = _build_tool_collection(session)
            _restore_latent_tools_from_flow(tool_collection)
            result2 = await run_in_daemon_thread(
                app_state.adapter.call_one_round,
                system_prompt_builder,
                chat_log,
                app_state.GEN,
                tool_collection,
                app_state.consciousness_flow,
                None,
                thread_name="main-llm-round-retry",
            )
            if not result2.failed and not result2.had_tool_call:
                await _synthesize_fallback_sleep(session)
                result2.had_tool_call = True
                result2.tool_calls_log.append({
                    "function": "sleep",
                    "arguments": {
                        "duration": EMPTY_TOOL_CALL_FALLBACK_DURATION,
                        "motivation": "fallback",
                    },
                    "result": {"ok": True, "fallback": True},
                })
            result = result2

    return result


# ── 主循环入口 ────────────────────────────────────────────────────────────

async def consciousness_main_loop() -> None:
    """常驻意识主循环。永不主动退出，仅响应 ``shutdown_event``。"""
    logger.info("[main] 意识主循环已启动 (initial_focus=%s)", app_state.current_focus)

    try:
        # 启动时若 current_focus 为空，等首条外部消息（来自任意会话）
        if app_state.current_focus is None:
            logger.info("[main] 当前无焦点，等待首条外部消息触发")
            await app_state.first_input_event.wait()

        while not app_state.shutdown_event.is_set():
            focus = app_state.current_focus
            if not focus:
                # 极少见的兜底：若被 shift 到不存在的 key 或被外部清空
                logger.warning("[main] current_focus 为空，等待新输入")
                app_state.first_input_event.clear()
                await app_state.first_input_event.wait()
                continue

            session = get_or_create_session(focus)
            _maybe_reset_chat_window_view(session, focus)

            t0 = _time.monotonic()
            result: RoundResult | None = None
            try:
                result = await _run_one_round(session, focus)
            except LLMCallFailed as exc:
                logger.warning("[main] LLM 调用最终失败 conv=%s: %s", focus, exc)
                await _synthesize_fallback_sleep(session)
                continue
            except Exception:
                logger.exception("[main] round 执行异常 conv=%s", focus)
                await asyncio.sleep(5)  # 避免炸事件循环
                continue

            elapsed = _time.monotonic() - t0
            if result is not None and not result.failed:
                logger.info(
                    "[main] round 完成 elapsed=%.2fs focus=%s tools=%d",
                    elapsed, focus, len(result.tool_calls_log),
                )
                await _persist_round(session, focus, result)
            else:
                logger.warning(
                    "[main] round 失败/无结果 elapsed=%.2fs focus=%s",
                    elapsed, focus,
                )

    except asyncio.CancelledError:
        logger.info("[main] 意识主循环被取消")
        raise
    except Exception:
        logger.exception("[main] 意识主循环异常退出")
        raise


def trigger_first_activation(initial_focus: str | None = None) -> None:
    """供外部首条消息回调使用：设置初始焦点（如未设置）并唤醒主循环。"""
    if initial_focus and app_state.current_focus is None:
        app_state.current_focus = initial_focus
        logger.info("[main] 首次激活，焦点 → %s", initial_focus)
    app_state.first_input_event.set()
