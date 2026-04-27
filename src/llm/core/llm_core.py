"""llm_core.py — Web 测试页专用的单 round LLM 调用封装

NOTE: 主循环（NapCat 集成）已迁移至 ``consciousness.main_loop``。本模块只剩
web/routes_chat.py 一个调用点，仅做"用户发一条消息 → bot 跑一 round → 返回结果"
的简化路径，**不进入意识永动循环**。
"""

import logging

import app_state
from tools import build_tools
from ..prompt.user_prompt_builder import build_main_user_prompt

logger = logging.getLogger("AICQ.llm.core")


def _restore_latent_tools_from_flow(tool_collection) -> None:
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
            "[web-llm] 从意识流恢复潜伏工具 count=%d names=%s",
            len(restored_names),
            ", ".join(restored_names),
        )


def call_model_and_process(session, **_unused):
    """跑一个 round，返回 ``(summary_dict, tool_calls_log, system_prompt)``。

    summary_dict 仅供 web 测试页 / 持久化使用，含 ``action`` 字段以兼容前端：
      - 调了 sleep / wait / shift 之一 → action 取该工具名
      - 否则 → action="respond"
    """
    conv_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
    if conv_key:
        app_state.last_active_session = conv_key

    tool_collection = build_tools(
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
    _restore_latent_tools_from_flow(tool_collection)

    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(
            activated_names=activated_names, latent_names=latent_names
        )

    chat_log = build_main_user_prompt(session)

    result = app_state.adapter.call_one_round(
        system_prompt_builder,
        chat_log,
        app_state.GEN,
        tool_collection,
        app_state.consciousness_flow,
        None,
    )

    if result.failed:
        return None, result.tool_calls_log, result.system_prompt

    tool_names = [c["function"] for c in result.tool_calls_log]
    summary_action = "respond"
    for special in ("sleep", "wait", "shift"):
        if special in tool_names:
            summary_action = special
            break
    summary = {"action": summary_action, "tools": tool_names}

    return summary, result.tool_calls_log, result.system_prompt
