"""llm_core.py — LLM 调用核心逻辑

提供 call_model_and_process()，供 NapCat 处理器调用。
"""

import logging
import time as _time

import app_state
from tools import build_tools
from ..prompt.user_prompt_builder import build_main_user_prompt

logger = logging.getLogger("AICQ.llm.core")


def _complete_pending_deferred_results(session) -> None:
    """补完前一轮 activation 留下的 deferred 工具返回。

    sleep → 用当前激活上下文（休眠时长、激活原因、当前会话）填充。
    wait → 通常已在 _run_active_loop 中补完，这里做兜底。
    """
    flow = app_state.consciousness_flow
    if flow is None or flow.round_count <= 0:
        return

    # ── sleep 延迟返回 ──
    sleep_ts = flow.get_deferred_timestamp("sleep")
    if sleep_ts is not None:
        elapsed = round(_time.time() - sleep_ts)
        result: dict = {"ok": True, "slept_seconds": elapsed}
        wake_reason = (session.last_wake_reason or "").strip()
        if wake_reason:
            result["woke_up_because"] = wake_reason
        session.last_wake_reason = ""
        result["current_session"] = {
            "type": session.conv_type,
            "id": session.conv_id,
            "name": session.conv_name,
        }
        flow.complete_deferred_response("sleep", result)
        logger.info("[app] 补完 sleep 延迟返回: slept=%ds", elapsed)

    # ── wait 兜底（正常情况已在 _run_active_loop 补完） ──
    wait_ts = flow.get_deferred_timestamp("wait")
    if wait_ts is not None:
        elapsed_w = round(_time.time() - wait_ts)
        flow.complete_deferred_response("wait", {
            "ok": True,
            "resumed": "unknown",
            "elapsed_seconds": elapsed_w,
        })
        logger.warning("[app] wait 延迟返回未被正常补完，兜底填充: elapsed=%ds", elapsed_w)


def _restore_latent_tools_from_flow(
    tool_collection,
) -> None:
    """根据当前保留的意识流历史，恢复仍应保持可用的潜伏工具。"""
    latent_names = set(tool_collection.latent_names())
    if not latent_names:
        return

    flow = app_state.consciousness_flow
    if flow is None or flow.round_count <= 0:
        return

    recoverable_names = flow.get_recoverable_latent_tool_names(latent_names)
    if not recoverable_names:
        return

    restored_names: list[str] = []
    for name in list(tool_collection.latent_names()):
        if name not in recoverable_names:
            continue
        if tool_collection.activate(name) is not None:
            restored_names.append(name)

    if restored_names:
        logger.info(
            "[app] 从意识流恢复潜伏工具 count=%d names=%s",
            len(restored_names),
            ", ".join(restored_names),
        )


def call_model_and_process(session, *, allow_retry_on_new_message: bool = True):
    """调用主模型（纯 function calling 路径），返回 (loop_action, tool_calls_log, system_prompt)。"""
    _complete_pending_deferred_results(session)

    # 视口生命周期：bot 离开本会话后再回来时（focus 改变）重置历史浏览视口。
    # last_active_session 记录上一次 call_model_and_process 处理的会话 key；
    # 仅当前一次激活属于"另一个会话"时（即 bot 真的离开过本会话），才需要重置。
    conv_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
    prev_active = app_state.last_active_session
    if conv_key and prev_active and prev_active != conv_key:
        if session.is_browsing_history():
            logger.info(
                "[app] 焦点曾切换 (%s → %s)，重置目标会话的历史浏览视口",
                prev_active, conv_key,
            )
            session.reset_chat_window_view()
    if conv_key:
        app_state.last_active_session = conv_key

    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(activated_names=activated_names, latent_names=latent_names)

    chat_log = build_main_user_prompt(session)

    logger.info("[app] 构建工具集开始 conv_type=%s", session.conv_type)
    tool_collection = build_tools(
        app_state.config,
        napcat_client=app_state.napcat_client,
        group_id=session.conv_id if session.conv_type == "group" else None,
        user_id=int(session.conv_id) if session.conv_type == "private" else None,
        session=session,
        vision_bridge=(
            app_state.vision_bridge
            if (app_state.vision_bridge and app_state.vision_bridge.enabled and not app_state.config.get("vision", True))
            else None
        ),
        provider=app_state.adapter.provider,
    )
    _restore_latent_tools_from_flow(tool_collection)
    logger.info(
        "[app] 构建工具集完成 tools_count=%d latent_count=%d",
        len(tool_collection.active_names()),
        len(tool_collection.latent_names()),
    )

    def _user_content_refresher():
        return build_main_user_prompt(session)

    logger.info("[app] LLM 调用开始 model=%s provider=%s", app_state.MODEL, app_state.adapter.provider)
    retry_on_new_message = (
        app_state.config.get("generation", {}).get("retry_on_new_message", True)
        and allow_retry_on_new_message
    )
    loop_action, tool_calls_log, system_prompt = app_state.adapter.call(
        system_prompt_builder,
        chat_log,
        app_state.GEN,
        tool_collection=tool_collection,
        user_content_refresher=_user_content_refresher,
        flow=app_state.consciousness_flow,
        new_message_checker=(lambda: session.unread_count > 0) if retry_on_new_message else None,
    )

    if loop_action is None:
        logger.warning("[app] LLM 调用失败，返回 None")
    else:
        logger.info("[app] LLM 调用完成 action=%s tool_calls=%d", loop_action.get("action"), len(tool_calls_log))

    return loop_action, tool_calls_log, system_prompt
