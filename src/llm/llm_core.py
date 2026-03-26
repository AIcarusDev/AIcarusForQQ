# Copyright (C) 2026  AIcarusDev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""llm_core.py — LLM 调用核心逻辑

提供 call_model_and_process() 和 commit_bot_messages_web()，
供 Web 路由和 NapCat 处理器共用。
"""

import logging
import uuid
from datetime import datetime

import app_state
from .schema import RESPONSE_SCHEMA
from tools import build_tools
from .session import (
    extract_bot_messages,
    set_bot_previous_cycle,
    set_bot_previous_cycle_time,
    set_bot_previous_tool_calls,
)

logger = logging.getLogger("AICQ.app")


def call_model_and_process(session):
    """调用模型，返回原始结果。

    注意：此函数**不再**将 bot 消息写入上下文。
    NapCat 端由调用者在发送完成后入上下文（使用 QQ 平台真实 ID）；
    Web 端由调用者自行入上下文（使用本地生成 ID）。

    返回 (result, grounding, system_prompt, user_prompt_display, repaired, tool_calls_log)。
    """
    def system_prompt_builder(tool_budget, rounds_used=0, max_rounds=None, tool_budget_suffix=""):
        return session.build_system_prompt(tool_budget=tool_budget, rounds_used=rounds_used, max_rounds=max_rounds, tool_budget_suffix=tool_budget_suffix)

    from .unread_builder import prepare_chat_log_with_unread
    chat_log = prepare_chat_log_with_unread(session)
    chat_log_display = session.get_chat_log_display()

    logger.info("[app] 构建工具集开始 conv_type=%s", session.conv_type)
    tool_declarations, tool_registry = build_tools(
        app_state.config,
        napcat_client=app_state.napcat_client,
        group_id=session.conv_id if session.conv_type == "group" else None,
        session=session,
        vision_bridge=app_state.vision_bridge if (app_state.vision_bridge and app_state.vision_bridge.enabled and not app_state.config.get("vision", True)) else None,
        provider=app_state.adapter.provider,
    )
    logger.info("[app] 构建工具集完成 tools_count=%d", len(tool_declarations))

    logger.info("[app] LLM 调用开始 model=%s provider=%s", app_state.MODEL, app_state.adapter.provider)
    result, grounding, repaired, tool_calls_log, system_prompt = app_state.adapter.call(
        system_prompt_builder,
        chat_log,
        app_state.GEN,
        RESPONSE_SCHEMA,
        tool_declarations=tool_declarations,
        tool_registry=tool_registry,
    )

    if result is None:
        logger.warning("[app] LLM 调用失败，返回 None")
        return None, None, system_prompt, chat_log_display, False, tool_calls_log

    logger.info("[app] LLM 调用完成 repaired=%s tool_calls=%d", repaired, len(tool_calls_log))

    session.previous_cycle_json = result
    set_bot_previous_cycle(result)
    set_bot_previous_cycle_time(datetime.now(app_state.TIMEZONE).isoformat())
    set_bot_previous_tool_calls(tool_calls_log)
    return result, grounding, system_prompt, chat_log_display, repaired, tool_calls_log


def commit_bot_messages_web(session, result: dict) -> None:
    """Web 端：从 LLM 结果提取 bot 消息并入上下文（使用本地生成 ID）。"""
    now_ts = datetime.now(app_state.TIMEZONE).isoformat()
    bot_sender_id = session._qq_id or "bot"
    bot_sender_name = session._qq_name or app_state.BOT_NAME
    for bot_msg in extract_bot_messages(result):
        session.add_to_context({
            "role": "bot",
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "sender_id": bot_sender_id,
            "sender_name": bot_sender_name,
            "sender_role": "",
            "timestamp": now_ts,
            "content": bot_msg["text"],
            "content_type": "text",
            "content_segments": bot_msg["content_segments"],
        })
