"""llm_core.py — LLM 调用核心逻辑

提供 call_model_and_process()，供 NapCat 处理器调用。
"""

import logging

import app_state
from tools import build_tools
from ..prompt.unread_builder import prepare_chat_log_with_unread
from ..prompt.final_reminder import append_final_reminder

logger = logging.getLogger("AICQ.app")


def call_model_and_process(session):
    """调用主模型（纯 function calling 路径），返回 (loop_action, tool_calls_log, system_prompt)。"""
    def system_prompt_builder(activated_names=None, latent_names=None):
        return session.build_system_prompt(activated_names=activated_names, latent_names=latent_names)

    chat_log = prepare_chat_log_with_unread(session)
    chat_log = append_final_reminder(chat_log, session)

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
    logger.info("[app] 构建工具集完成 tools_count=%d latent_count=%d", len(tool_declarations), len(latent_registry))

    def _user_content_refresher():
        fresh = prepare_chat_log_with_unread(session)
        return append_final_reminder(fresh, session)

    logger.info("[app] LLM 调用开始 model=%s provider=%s", app_state.MODEL, app_state.adapter.provider)
    loop_action, tool_calls_log, system_prompt = app_state.adapter.call(
        system_prompt_builder,
        chat_log,
        app_state.GEN,
        tool_declarations=tool_declarations,
        tool_registry=tool_registry,
        latent_registry=latent_registry,
        user_content_refresher=_user_content_refresher,
    )

    if loop_action is None:
        logger.warning("[app] LLM 调用失败，返回 None")
    else:
        logger.info("[app] LLM 调用完成 action=%s tool_calls=%d", loop_action.get("action"), len(tool_calls_log))

    return loop_action, tool_calls_log, system_prompt
