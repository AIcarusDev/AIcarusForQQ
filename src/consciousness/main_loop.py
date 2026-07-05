"""consciousness/main_loop.py — 机器人意识主循环

常驻 asyncio task，永远运行，直到 ``app_state.shutdown_event`` 被置位。

每一 round = 一次 LLM 调用 + 本轮工具的真正执行 + 持久化。
runtime_manage/enter_qq_session 是普通工具，分别由其 handler 内部阻塞 / 修改全局焦点；
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
from agent_events import emit_agent_event
from database import save_adapter_contents, save_bot_turn
from llm.core.daemon_thread import run_in_daemon_thread
from llm.session import get_or_create_session, sessions
from llm.core.provider import LLMCallFailed, RoundResult
from llm.core.tool_execution_guard import build_qq_guard_snapshot
from llm.core.error_policy import LLMErrorDecision, normalize_llm_error
from llm.compression.config import normalize_generation_config
from llm.core.duplicate_response_guard import (
    build_duplicate_model_response_limit_error,
    normalize_duplicate_model_response_guard_config,
)
from llm.compression.worker import schedule_cognition_compression
from llm.prompt.user_prompt_builder import build_main_user_prompt
from platforms.focus import FocusRef, current_focus_key, focus_from_session_key, session_key_for_focus
from platforms.registry import get_platform
from platforms.qq.session_context import qq_surface_for_focus, resolve_current_qq_session
from runtime import core_restart
from runtime.emergency_reset import (
    is_runtime_epoch_stale,
    make_runtime_epoch_checker,
    mark_result_aborted_by_reset,
)
from tools import build_tools
from tools.namespaces import NamespaceRuntimeState

from .flow import ToolCall, ToolResponse

logger = logging.getLogger("AICQ.consciousness.main")

# 模型连续违规（不调任何工具）时强制注入的兜底休眠时长（分钟）
EMPTY_TOOL_CALL_FALLBACK_DURATION = 60


# ── 内部辅助 ────────────────────────────────────────────────────────────────

def _maybe_reset_transient_session_views(session, conv_key: str) -> None:
    """跨会话切换时清理会话内的临时浏览视图。"""
    prev = current_focus_key(app_state.last_active_session)
    if conv_key and prev and prev != conv_key:
        prev_session = sessions.get(prev)
        if prev_session is not None and prev_session is not session:
            if prev_session.is_browsing_history() or prev_session.is_browsing_forward():
                logger.info("[main] 焦点离开 %s，清理原会话临时浏览视图", prev)
                prev_session.reset_transient_views()

        if session.is_browsing_history() or session.is_browsing_forward():
            logger.info("[main] 焦点进入 %s，清理目标会话残留临时浏览视图", conv_key)
            session.reset_transient_views()
    if conv_key:
        app_state.last_active_session = focus_from_session_key(conv_key)


def _build_tool_collection(session):
    """每 round 重建工具集（保证 system prompt / 工具白名单与当前焦点一致）。"""
    if app_state.namespace_runtime_state is None:
        app_state.namespace_runtime_state = NamespaceRuntimeState()
    max_rounds = normalize_generation_config(app_state.GEN)["llm_contents_max_rounds"]
    flow = app_state.consciousness_flow
    current_round = int(getattr(flow, "next_seq", 0) or 0)
    qq_runtime = get_platform("qq")
    qq_client = getattr(qq_runtime, "client", None)
    return build_tools(
        app_state.config,
        namespace_state=app_state.namespace_runtime_state,
        current_round=current_round,
        default_ttl_rounds=max_rounds,
        flow=flow,
        qq_runtime=qq_runtime,
        qq_surface=qq_surface_for_focus(app_state.current_focus),
        qq_session_provider=resolve_current_qq_session,
        qq_client=qq_client,
        group_id=session.conv_id if session.conv_type == "group" else None,
        user_id=int(session.conv_id) if session.conv_type in {"private", "temp"} else None,
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


def _prompt_snapshot_context(session, conv_key: str, attempt: str) -> dict:
    return {
        "conv_type": getattr(session, "conv_type", ""),
        "conv_id": getattr(session, "conv_id", ""),
        "focus": conv_key,
        "attempt": attempt,
    }


async def _persist_round(
    session,
    conv_key: str,
    result: RoundResult,
    *,
    elapsed_ms: float | None = None,
) -> bool:
    """把本 round 的简要摘要写入 bot_turns 并触发意识流持久化。"""
    expected_epoch = getattr(result, "runtime_reset_epoch", 0)
    if is_runtime_epoch_stale(expected_epoch):
        logger.info("[main] 跳过过期 round 持久化 conv=%s epoch=%s", conv_key, expected_epoch)
        return False
    try:
        # NOTE: bot_turns.result 字段在新架构下不再有 action 语义，仅作可读摘要
        summary = {
            "tools": [c["function"] for c in result.tool_calls_log],
            "tokens": {"in": result.prompt_tokens, "out": result.output_tokens},
        }
        if elapsed_ms is not None:
            summary["elapsed_ms"] = elapsed_ms
        if result.cognition:
            summary["cognition"] = result.cognition
        if result.prompt_snapshot_id:
            summary["prompt_snapshot_id"] = result.prompt_snapshot_id
        turn_id = uuid.uuid4().hex
        await save_bot_turn(
            turn_id=turn_id,
            conv_type=session.conv_type,
            conv_id=session.conv_id,
            result=summary,
            tool_calls_log=result.tool_calls_log,
            world_xml=getattr(result, "world_xml", ""),
        )
        agent_run_id = getattr(result, "agent_run_id", "")
        if agent_run_id:
            emit_agent_event(
                "round_persisted",
                round_id=agent_run_id,
                turn_id=turn_id,
                session_key=conv_key,
                conv_type=session.conv_type,
                conv_id=session.conv_id,
                conv_name=getattr(session, "conv_name", "") or conv_key,
                focus=conv_key,
            )
    except Exception:
        logger.warning("[main] save_bot_turn 失败 conv=%s", conv_key, exc_info=True)
    if is_runtime_epoch_stale(expected_epoch):
        logger.info("[main] round 持久化后检测到紧急恢复，跳过 flow 保存 conv=%s", conv_key)
        return False
    # 持久化意识流（重启后可恢复）
    try:
        c_data, ts_data = app_state.consciousness_flow.dump()
        asyncio.create_task(save_adapter_contents("flow", c_data, ts_data))
    except Exception:
        logger.warning("[main] 意识流持久化失败", exc_info=True)
    return True


def _schedule_archive(session, tool_calls_log: list) -> None:
    """Legacy recent-window archive trigger; V2 archives raw compression ranges."""
    return


async def _synthesize_fallback_sleep(session, duration: int | None = None, response: dict | None = None) -> None:
    """模型连续违规时合成一个 runtime_manage sleep 调用并写入意识流。"""
    flow = app_state.consciousness_flow
    duration = int(duration or EMPTY_TOOL_CALL_FALLBACK_DURATION)
    call_id = f"fallback-runtime-manage-{uuid.uuid4().hex[:8]}"
    if flow:
        max_rounds = normalize_generation_config(app_state.GEN)["llm_contents_max_rounds"]
        flow.prune(max_rounds)

    from tools.core.runtime_manage import build_runtime_result, wait_until_attention

    logger.warning("[main] 模型违规兜底：注入 runtime_manage(action=sleep, minutes=%dm)", duration)
    request_started_at = _time.time()
    sleep_started_at = _time.monotonic()
    reason = await wait_until_attention(session, duration * 60)
    waited_seconds = _time.monotonic() - sleep_started_at
    result = build_runtime_result(
        session,
        action="sleep",
        requested_seconds=duration * 60,
        waited_seconds=waited_seconds,
        elapsed_since_request=_time.time() - request_started_at,
        reason=reason,
    )
    if flow:
        if response:
            result = dict(result)
            result["guard"] = response
        flow.append_round(
            [ToolCall(name="runtime_manage", args={"action": "sleep", "minutes": duration}, call_id=call_id)],
            [ToolResponse(name="runtime_manage", response=result, call_id=call_id)],
        )


async def _cooldown_after_llm_error(
    session,
    conv_key: str,
    decision: LLMErrorDecision,
) -> None:
    """Apply runtime backoff for provider/API failures without writing fake tool calls."""
    wait_seconds = max(0.0, float(decision.cooldown_seconds or 0.0))
    logger.warning(
        "[main] LLM 错误退避 conv=%s category=%s status=%s retryable=%s action=%s wait=%.1fs summary=%s",
        conv_key,
        decision.category,
        decision.status_code,
        decision.retryable,
        decision.action,
        wait_seconds,
        decision.summary,
    )
    emit_agent_event(
        "round_error",
        session_key=conv_key,
        focus=conv_key,
        conv_type=getattr(session, "conv_type", ""),
        conv_id=getattr(session, "conv_id", ""),
        conv_name=getattr(session, "conv_name", "") or conv_key,
        error=decision.summary,
        category=decision.category,
        status_code=decision.status_code,
        retryable=decision.retryable,
        cooldown_seconds=wait_seconds,
        stage="llm_error_policy",
    )
    if wait_seconds <= 0:
        return
    try:
        await asyncio.wait_for(app_state.shutdown_event.wait(), timeout=wait_seconds)
    except asyncio.TimeoutError:
        return


# ── 单 round 执行（含 retry 语义） ─────────────────────────────────────────

async def _run_one_round(session, conv_key: str) -> RoundResult:
    """跑一个 round，处理模型违规重调。

    - 模型一次工具都没调 → 重调一次；仍然不调 → 合成兜底 runtime_manage.sleep。
    """
    round_epoch = int(getattr(app_state, "runtime_reset_epoch", 0))
    stale_checker = make_runtime_epoch_checker(round_epoch)

    async def _safe_memory_recall() -> None:
        try:
            await session.prepare_memory_recall()
        except Exception:
            logger.warning("[main] prepare_memory_recall 失败，本 round 跳过召回", exc_info=True)

    async def _safe_quote_prefetch() -> None:
        runtime = get_platform(session.get_platform_key())
        prefetch = getattr(runtime, "prefetch_quoted_messages", None)
        try:
            if prefetch is not None:
                await prefetch(session)
            else:
                from platforms.chat.quote_prefetch import prefetch_quoted_messages

                await prefetch_quoted_messages(session)
        except Exception:
            logger.warning("[main] prefetch_quoted_messages 失败，本 round 跳过引用预取", exc_info=True)

    await asyncio.gather(
        _safe_memory_recall(),
        _safe_quote_prefetch(),
    )

    tool_collection = _build_tool_collection(session)

    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(
            activated_names=activated_names, latent_names=latent_names
        )

    duplicate_retry_count = 0
    cognition_prefill_retry_count = 0
    assistant_prefill = ""
    used_cognition_prefills: list[str] = []
    agent_run_id = uuid.uuid4().hex
    agent_context = {
        "session_key": conv_key,
        "focus": conv_key,
        "conv_type": getattr(session, "conv_type", ""),
        "conv_id": getattr(session, "conv_id", ""),
        "conv_name": getattr(session, "conv_name", "") or conv_key,
    }
    emit_agent_event(
        "round_start",
        round_id=agent_run_id,
        runtime_reset_epoch=round_epoch,
        **agent_context,
    )

    await app_state.rate_limiter.acquire()
    async with app_state.llm_lock:
        if stale_checker():
            return mark_result_aborted_by_reset(RoundResult(), round_epoch)
        while True:
            chat_log = build_main_user_prompt(session)
            decision_guard_snapshot = build_qq_guard_snapshot(session)

            def current_world_provider():
                return build_main_user_prompt(session, consume_unread=False)

            def current_guard_snapshot_provider():
                return build_qq_guard_snapshot(session)

            usage_feature = "main_round"

            result = await run_in_daemon_thread(
                app_state.adapter.call_one_round,
                system_prompt_builder,
                chat_log,
                app_state.GEN,
                tool_collection,
                app_state.consciousness_flow,
                usage_feature=usage_feature,
                prompt_snapshot_context=_prompt_snapshot_context(
                    session, conv_key, usage_feature
                ),
                runtime_stale_checker=stale_checker,
                current_world_provider=current_world_provider,
                decision_guard_snapshot=decision_guard_snapshot,
                current_guard_snapshot_provider=current_guard_snapshot_provider,
                agent_run_id=agent_run_id,
                agent_context=agent_context,
                assistant_prefill=assistant_prefill,
                prefill_exclusions=tuple(used_cognition_prefills),
                thread_name="main-llm-round",
            )
            result.runtime_reset_epoch = round_epoch
            result.agent_run_id = agent_run_id

            if stale_checker() or getattr(result, "aborted_by_runtime_reset", False):
                emit_agent_event(
                    "round_error",
                    round_id=agent_run_id,
                    error="runtime_reset",
                    stage="main_round",
                    **agent_context,
                )
                return mark_result_aborted_by_reset(result, round_epoch)

            if getattr(result, "cognition_prefill_retry", False):
                cognition_prefill_retry_count += 1
                guard_cfg = normalize_duplicate_model_response_guard_config(
                    app_state.GEN.get("duplicate_model_response_guard")
                )
                prefill_cfg = guard_cfg.get("prefill_guidance") or {}
                next_prefill = str(getattr(result, "cognition_prefill", "") or "")
                next_prefill_body = str(getattr(result, "cognition_prefill_body", "") or "")
                if (
                    cognition_prefill_retry_count >= int(prefill_cfg.get("max_retries") or 2)
                    or not next_prefill
                ):
                    response = dict(getattr(result, "cognition_prefill_retry_error", {}) or {})
                    response.update({
                        "error": "REPEATED_COGNITION_PREFILL_LIMIT",
                        "message": "模型连续输出与可见意识流高度重复的 cognition，已停止重试并进入 runtime_manage.sleep。",
                        "retryable": False,
                        "fallback": "runtime_manage.sleep",
                    })
                    await _synthesize_fallback_sleep(
                        session,
                        duration=guard_cfg["fallback_sleep_minutes"],
                        response=response,
                    )
                    result.had_tool_call = True
                    result.cognition = ""
                    result.raw_response = ""
                    result.prompt_snapshot_id = ""
                    result.discarded_cognition = ""
                    result.tool_calls_log.append({
                        "function": "runtime_manage",
                        "arguments": {"action": "sleep", "minutes": guard_cfg["fallback_sleep_minutes"]},
                        "result": {
                            "ok": True,
                            "fallback": True,
                            "reason": "repeated_cognition_prefill",
                        },
                    })
                    break
                assistant_prefill = next_prefill
                if next_prefill_body:
                    used_cognition_prefills.append(next_prefill_body)
                logger.warning(
                    "[main] cognition 重复，使用 prefill 重调 conv=%s count=%s",
                    conv_key,
                    cognition_prefill_retry_count,
                )
                emit_agent_event(
                    "round_retry",
                    round_id=agent_run_id,
                    reason="repeated_cognition_prefill",
                    retry_count=cognition_prefill_retry_count,
                    **agent_context,
                )
                tool_collection = _build_tool_collection(session)
                continue

            assistant_prefill = ""

            if getattr(result, "duplicate_model_response", False):
                duplicate_retry_count += 1
                guard_cfg = normalize_duplicate_model_response_guard_config(
                    app_state.GEN.get("duplicate_model_response_guard")
                )
                if duplicate_retry_count >= guard_cfg["max_retries"]:
                    await _synthesize_fallback_sleep(
                        session,
                        duration=guard_cfg["fallback_sleep_minutes"],
                        response=build_duplicate_model_response_limit_error(
                            duplicate_count=duplicate_retry_count
                        ),
                    )
                    result.had_tool_call = True
                    result.tool_calls_log.append({
                        "function": "runtime_manage",
                        "arguments": {"action": "sleep", "minutes": guard_cfg["fallback_sleep_minutes"]},
                        "result": {"ok": True, "fallback": True, "reason": "duplicate_model_response"},
                    })
                    break
                logger.warning(
                    "[main] 模型完整输出重复，重调一次 conv=%s count=%s",
                    conv_key,
                    duplicate_retry_count,
                )
                emit_agent_event(
                    "round_retry",
                    round_id=agent_run_id,
                    reason="duplicate_model_response",
                    retry_count=duplicate_retry_count,
                    **agent_context,
                )
                tool_collection = _build_tool_collection(session)
                continue

            break

        # ── 模型违规（不调任何工具）重调 1 次，再失败就硬塞 runtime_manage.sleep ────────
        if not result.failed and not result.had_tool_call:
            logger.warning("[main] 模型未调任何工具，重调一次 conv=%s", conv_key)
            emit_agent_event(
                "round_retry",
                round_id=agent_run_id,
                reason="no_tool_call",
                retry_count=1,
                **agent_context,
            )
            chat_log = build_main_user_prompt(session)
            decision_guard_snapshot = build_qq_guard_snapshot(session)

            def retry_current_world_provider():
                return build_main_user_prompt(session, consume_unread=False)

            def retry_current_guard_snapshot_provider():
                return build_qq_guard_snapshot(session)

            tool_collection = _build_tool_collection(session)
            result2 = await run_in_daemon_thread(
                app_state.adapter.call_one_round,
                system_prompt_builder,
                chat_log,
                app_state.GEN,
                tool_collection,
                app_state.consciousness_flow,
                usage_feature="main_round_retry_no_tool",
                prompt_snapshot_context=_prompt_snapshot_context(
                    session, conv_key, "main_round_retry_no_tool"
                ),
                runtime_stale_checker=stale_checker,
                current_world_provider=retry_current_world_provider,
                decision_guard_snapshot=decision_guard_snapshot,
                current_guard_snapshot_provider=retry_current_guard_snapshot_provider,
                agent_run_id=agent_run_id,
                agent_context=agent_context,
                thread_name="main-llm-round-retry",
            )
            result2.runtime_reset_epoch = round_epoch
            result2.agent_run_id = agent_run_id
            if stale_checker() or getattr(result2, "aborted_by_runtime_reset", False):
                emit_agent_event(
                    "round_error",
                    round_id=agent_run_id,
                    error="runtime_reset",
                    stage="main_round_retry",
                    **agent_context,
                )
                return mark_result_aborted_by_reset(result2, round_epoch)
            if not result2.failed and not result2.had_tool_call:
                if stale_checker():
                    return mark_result_aborted_by_reset(result2, round_epoch)
                await _synthesize_fallback_sleep(session)
                if stale_checker():
                    return mark_result_aborted_by_reset(result2, round_epoch)
                result2.had_tool_call = True
                result2.tool_calls_log.append({
                    "function": "runtime_manage",
                    "arguments": {
                        "action": "sleep",
                        "minutes": EMPTY_TOOL_CALL_FALLBACK_DURATION,
                    },
                    "result": {"ok": True, "fallback": True},
                })
            result = result2

    result.runtime_reset_epoch = round_epoch
    result.agent_run_id = agent_run_id
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
            focus_ref = app_state.current_focus
            focus = current_focus_key(focus_ref)
            if not focus:
                # 极少见的兜底：若被焦点切换到不存在的 key 或被外部清空
                logger.warning("[main] current_focus 为空，等待新输入")
                app_state.first_input_event.clear()
                await app_state.first_input_event.wait()
                continue

            session = get_or_create_session(focus_ref if isinstance(focus_ref, FocusRef) else focus)
            _maybe_reset_transient_session_views(session, focus)

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
            elapsed_ms = round(elapsed * 1000, 3)
            if result is not None and not result.failed:
                if is_runtime_epoch_stale(getattr(result, "runtime_reset_epoch", 0)):
                    logger.info("[main] round 已被紧急恢复失效，跳过后续处理 focus=%s", focus)
                    continue
                logger.info(
                    "[main] round 完成 elapsed=%.2fs focus=%s tools=%d",
                    elapsed, focus, len(result.tool_calls_log),
                )
                emit_agent_event(
                    "round_done",
                    round_id=getattr(result, "agent_run_id", ""),
                    session_key=focus,
                    focus=focus,
                    conv_type=getattr(session, "conv_type", ""),
                    conv_id=getattr(session, "conv_id", ""),
                    conv_name=getattr(session, "conv_name", "") or focus,
                    elapsed_ms=elapsed_ms,
                    tool_count=len(result.tool_calls_log),
                    prompt_tokens=result.prompt_tokens,
                    output_tokens=result.output_tokens,
                    failed=False,
                )
                if not await _persist_round(session, focus, result, elapsed_ms=elapsed_ms):
                    continue
                if await core_restart.shutdown_after_round_if_requested():
                    return
                if is_runtime_epoch_stale(getattr(result, "runtime_reset_epoch", 0)):
                    logger.info("[main] round 后处理前检测到紧急恢复，跳过压缩/归档 focus=%s", focus)
                    continue
                schedule_cognition_compression()
                _schedule_archive(session, result.tool_calls_log)
            else:
                llm_error = normalize_llm_error(getattr(result, "llm_error", None))
                logger.warning(
                    "[main] round 失败/无结果 elapsed=%.2fs focus=%s llm_error=%s",
                    elapsed, focus, llm_error.category if llm_error else "",
                )
                if result is not None:
                    emit_agent_event(
                        "round_error",
                        round_id=getattr(result, "agent_run_id", ""),
                        session_key=focus,
                        focus=focus,
                        conv_type=getattr(session, "conv_type", ""),
                        conv_id=getattr(session, "conv_id", ""),
                        conv_name=getattr(session, "conv_name", "") or focus,
                        elapsed_ms=round(elapsed * 1000, 3),
                        error="round_failed",
                        stage="main_loop",
                    )
                if llm_error is not None:
                    await _cooldown_after_llm_error(session, focus, llm_error)
                else:
                    await asyncio.sleep(5)

    except asyncio.CancelledError:
        logger.info("[main] 意识主循环被取消")
        raise
    except Exception:
        logger.exception("[main] 意识主循环异常退出")
        raise


def trigger_first_activation(initial_focus: str | FocusRef | None = None) -> None:
    """供外部首条消息回调使用：设置初始焦点（如未设置）并唤醒主循环。"""
    if initial_focus and app_state.current_focus is None:
        app_state.current_focus = initial_focus if isinstance(initial_focus, FocusRef) else focus_from_session_key(initial_focus)
        logger.info("[main] 首次激活，焦点 → %s", current_focus_key(app_state.current_focus))
    app_state.first_input_event.set()


