"""IS/core.py — 中断哨兵（Interruption Sentinel）核心

check_interruption() 是唯一对外接口：
  - 构建哨兵 system/user prompt
  - 同步调用 IS 模型（在线程池中执行，不阻塞事件循环）
  - 返回 (should_interrupt: bool, reason: str)

IS 模型输出格式由 config/schema/is.json 约束：{"continue": bool, "reason": str}
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import app_state
from .chat_log_builder import build_sentinel_chat_log
from .prompt import SENTINEL_PROMPT_SYS_TEMPLATE, SENTINEL_PROMPT_USER_TEMPLATE
from ..prompt.prompt import get_formatted_time_for_llm
from ..prompt.xml_builder import _inject_images_by_ref, _resolve_sentinels

logger = logging.getLogger("AICQ.is")

# IS 结构化输出 schema（从文件加载）
_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "schema" / "is.json"
with open(_SCHEMA_PATH, encoding="utf-8") as _f:
    IS_SCHEMA = json.load(_f)

# IS 默认生成参数
_DEFAULT_IS_GEN = {
    "temperature": 0.3,
    "max_output_tokens": 300,
}


def _get_is_gen() -> dict:
    """返回 IS 生成参数，优先使用 is_cfg 中的配置。"""
    base = dict(_DEFAULT_IS_GEN)
    cfg_gen = app_state.is_cfg.get("generation", {})
    if cfg_gen:
        base.update(cfg_gen)
    return base


def _build_context_description(session) -> str:
    """根据会话类型构建上下文描述字符串。"""
    if session.conv_type == "group":
        name = session.conv_name or session.conv_id
        return f'群聊"{name}"中聊天'
    elif session.conv_type == "private":
        name = session.conv_name or session.conv_id
        return f'与{name}私聊'
    return "聊天"


def _format_plan_message_list(send_messages: list[dict]) -> str:
    """将 send_messages 格式化为 IS prompt 中的 messages 段落。

    格式：
     - "你好呀"
     - "起这么早呀"
    """
    lines: list[str] = []
    for msg in send_messages:
        parts: list[str] = []
        for seg in msg.get("segments", []):
            cmd = seg.get("command", "")
            params = seg.get("params", {})
            if cmd == "text":
                parts.append(params.get("content", ""))
            elif cmd == "sticker":
                parts.append("[动画表情]")
            elif cmd == "at":
                parts.append(f"@{params.get('user_id', '')}")
        text = "".join(parts)
        lines.append(f' - "{text}"')
    return "\n".join(lines)


def _is_plan_msg_sticker_only(msg: dict) -> bool:
    """判断计划消息是否为纯动画表情（无文字）。"""
    segments = msg.get("segments", [])
    if not segments:
        return False
    return all(seg.get("command") == "sticker" for seg in segments)


def _is_adapter_vision_enabled() -> bool:
    """IS 模型是否支持读图（多模态）。"""
    adapter = app_state.is_adapter or app_state.adapter
    return bool(getattr(adapter, "_vision_enabled", False))


def _call_is_model_sync(
    system_prompt: str,
    user_content: "str | list",
) -> tuple[bool, str]:
    """同步调用 IS 模型，返回 (continue_sending, reason)。失败时默认 continue=True。"""
    adapter = app_state.is_adapter or app_state.adapter
    if adapter is None:
        logger.warning("[IS] 无可用适配器，默认继续发送")
        return True, "无适配器，默认继续"

    gen = _get_is_gen()

    def _prompt_builder(activated_names=None, latent_names=None):
        return system_prompt

    try:
        result, _, _, _, _ = adapter.call(
            _prompt_builder,
            user_content,
            gen,
            IS_SCHEMA,
            tool_declarations=None,
            tool_registry=None,
            latent_registry=None,
            user_content_refresher=None,
        )
        if result is None:
            logger.warning("[IS] 模型返回 None，默认继续发送")
            return True, "模型返回空，默认继续"

        should_continue = bool(result.get("continue", True))
        reason = str(result.get("reason", ""))
        logger.info("[IS] 判断结果: continue=%s, reason=%s", should_continue, reason)
        return should_continue, reason
    except Exception as e:
        logger.warning("[IS] 模型调用异常，默认继续发送: %s", e)
        return True, f"调用异常: {e}"


async def check_interruption(
    session,
    result: dict,
    sent_count: int,
    trigger_entry: dict,
    remaining_plan_msgs: list[dict],
    sent_this_round_ids: set[str],
) -> tuple[bool, str]:
    """判断是否中断后续消息发送。

    参数：
        session:              当前会话对象。
        result:               本轮 LLM 完整输出 dict。
        sent_count:           本轮已成功发送的消息条数。
        trigger_entry:        触发 IS 的消息 context entry（已在 context 中）。
        remaining_plan_msgs:  尚未发送的计划消息（send_messages 后半段）。
        sent_this_round_ids:  本轮已发送消息的 message_id 集合。

    返回：
        (should_interrupt: bool, reason: str)
        True = 停止发送；False = 继续发送。
    """
    # 检查 IS 功能是否已禁用
    if not app_state.is_cfg.get("enabled", True):
        return False, "IS 已禁用"

    # 过滤 ①：触发消息本身是纯表情 → 跳过 IS
    if trigger_entry.get("content_type") == "sticker":
        logger.debug("[IS] 跳过：触发消息是纯表情")
        return False, "触发消息是纯表情"

    # 过滤 ②：剩余唯一一条纯动画表情 → 跳过 IS（社交上可接受）
    if len(remaining_plan_msgs) == 1 and _is_plan_msg_sticker_only(remaining_plan_msgs[0]):
        logger.debug("[IS] 跳过：剩余仅单条纯动画表情")
        return False, "剩余仅单条纯动画表情"

    # ── 构建 system prompt ────────────────────────────────────────
    now = datetime.now(session._timezone) if session._timezone else datetime.now()
    time_str = get_formatted_time_for_llm(now)

    decision = result.get("decision") or {}
    send_messages = decision.get("send_messages") or []

    trigger_sender = trigger_entry.get("sender_name", "")
    trigger_content = trigger_entry.get("content", "")

    system_prompt = SENTINEL_PROMPT_SYS_TEMPLATE.format(
        persona=session._persona,
        time=time_str,
        qq_name=session._qq_name,
        qq_id=session._qq_id,
        context=_build_context_description(session),
        mood=result.get("mood", ""),
        think=result.get("think", ""),
        intent=result.get("intent", ""),
        message_count=len(send_messages),
        messages=_format_plan_message_list(send_messages),
        motivation=decision.get("motivation", ""),
        expected=result.get("expected", ""),
        quantity_sent_count=sent_count,
        user_name=trigger_sender,
        user_message=trigger_content,
    )

    # ── 构建 user content（哨兵聊天记录）────────────────────────────
    conv_meta = {
        "type": session.conv_type,
        "id": session.conv_id,
        "name": session.conv_name,
        "bot_id": session._qq_id,
        "bot_name": session._qq_name,
    }
    trigger_id = str(trigger_entry.get("message_id", ""))
    chat_log_xml, _images = build_sentinel_chat_log(
        context_messages=session.context_messages,
        trigger_id=trigger_id,
        sent_this_round_ids=sent_this_round_ids,
        remaining_plan_msgs=remaining_plan_msgs,
        conv_meta=conv_meta,
    )

    _full_user_text = SENTINEL_PROMPT_USER_TEMPLATE.format(
        chat_log_for_sentinel=chat_log_xml,
    )
    # 根据 IS 模型是否支持视觉，选择不同的图片处理方式：
    #   支持视觉：将图片 base64 直接嵌入多模态 parts。
    #   不支持：将 vision bridge 描述嵌入 <description> 文本块。
    if _images and _is_adapter_vision_enabled():
        user_content = _inject_images_by_ref(_full_user_text, _images)
    elif _images:
        user_content = _resolve_sentinels(_full_user_text, _images)
    else:
        user_content = _full_user_text

    # ── 在线程池中同步调用模型 ────────────────────────────────────
    _t0 = time.monotonic()
    should_continue, reason = await asyncio.to_thread(
        _call_is_model_sync, system_prompt, user_content
    )
    elapsed = time.monotonic() - _t0
    logger.info("[IS] 耗时 %.2fs, continue=%s", elapsed, should_continue)

    # should_continue=True 表示继续 → should_interrupt=False
    return not should_continue, reason
