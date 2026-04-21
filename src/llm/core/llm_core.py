"""llm_core.py — LLM 调用核心逻辑

提供 call_model_and_process()，供 NapCat 处理器调用。
"""

import logging
import time as _time

import app_state
from tools import build_tools
from ..prompt.user_prompt_builder import build_main_user_prompt

logger = logging.getLogger("AICQ.app")


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
        from ..prompt.activity_log import get_current as _get_current_activity

        elapsed = round(_time.time() - sleep_ts)
        result: dict = {"ok": True, "slept_seconds": elapsed}

        current = _get_current_activity()
        if current:
            if current.enter_remark:
                result["woke_up_because"] = current.enter_remark
            elif current.enter_motivation:
                result["woke_up_because"] = current.enter_motivation
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
    tool_declarations: list,
    tool_registry: dict,
    latent_registry: dict,
) -> None:
    """根据当前保留的意识流历史，恢复仍应保持可用的潜伏工具。"""
    if not latent_registry:
        return

    flow = app_state.consciousness_flow
    if flow is None or flow.round_count <= 0:
        return

    recoverable_names = flow.get_recoverable_latent_tool_names(set(latent_registry.keys()))
    if not recoverable_names:
        return

    restored_names: list[str] = []
    for name in list(latent_registry.keys()):
        if name not in recoverable_names:
            continue
        decl, handler = latent_registry.pop(name)
        tool_declarations.append(decl)
        tool_registry[name] = handler
        restored_names.append(name)

    if restored_names:
        logger.info(
            "[app] 从意识流恢复潜伏工具 count=%d names=%s",
            len(restored_names),
            ", ".join(restored_names),
        )


def call_model_and_process(session):
    """调用主模型（纯 function calling 路径），返回 (loop_action, tool_calls_log, system_prompt)。"""
    _complete_pending_deferred_results(session)

    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(activated_names=activated_names, latent_names=latent_names)

    chat_log = build_main_user_prompt(session)

    logger.info("[app] 构建工具集开始 conv_type=%s", session.conv_type)
    tool_declarations, tool_registry, latent_registry = build_tools(
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
    _restore_latent_tools_from_flow(tool_declarations, tool_registry, latent_registry)
    logger.info("[app] 构建工具集完成 tools_count=%d latent_count=%d", len(tool_declarations), len(latent_registry))

    def _user_content_refresher():
        return build_main_user_prompt(session)

    logger.info("[app] LLM 调用开始 model=%s provider=%s", app_state.MODEL, app_state.adapter.provider)
    loop_action, tool_calls_log, system_prompt = app_state.adapter.call(
        system_prompt_builder,
        chat_log,
        app_state.GEN,
        tool_declarations=tool_declarations,
        tool_registry=tool_registry,
        latent_registry=latent_registry,
        user_content_refresher=_user_content_refresher,
        flow=app_state.consciousness_flow,
        new_message_checker=lambda: session.unread_count > 0,
    )

    if loop_action is None:
        logger.warning("[app] LLM 调用失败，返回 None")
    else:
        logger.info("[app] LLM 调用完成 action=%s tool_calls=%d", loop_action.get("action"), len(tool_calls_log))

    return loop_action, tool_calls_log, system_prompt
