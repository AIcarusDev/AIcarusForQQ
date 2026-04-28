"""intent 路由与内在声音调用逻辑。"""

import logging

import app_state

from .prompts import PROMPTS

logger = logging.getLogger("AICQ.tools.think_deeply")

INTENTS: list[str] = ["affirmation", "criticism", "solving", "inspiration", "simulate"]


def _get_gen() -> dict:
    """返回 slow_thinking 生成参数，优先使用配置中的 slow_thinking.generation。"""
    base: dict = {"temperature": 1.0}
    st_cfg: dict = getattr(app_state, "slow_thinking_cfg", {})
    cfg_gen = st_cfg.get("generation", {})
    if isinstance(cfg_gen, dict) and cfg_gen:
        base.update(cfg_gen)
    return base


def _build_world_context(session) -> str:
    """为 simulate intent 构建 <world> 上下文块。"""
    if session is None:
        return ""
    conv_type: str = getattr(session, "conv_type", "")
    conv_id: str = getattr(session, "conv_id", "")
    conv_name: str = getattr(session, "conv_name", "") or conv_id
    count: int = getattr(session, "conv_member_count", 0)

    if conv_type == "group":
        return f'<world>当前在群聊"{conv_name}"中（共 {count} 名成员）。</world>'
    if conv_type == "private":
        return f'<world>当前在与 {conv_name} 的私聊中。</world>'
    return ""


def call_inner_voice(intent: str, content: str, session=None) -> str:
    """调用对应 intent 的内在声音模型，返回自然语言结论。失败时返回占位字符串。"""
    adapter = getattr(app_state, "slow_thinking_adapter", None) or app_state.adapter
    if adapter is None:
        logger.warning("[think_deeply] 无可用适配器，intent=%s", intent)
        return "(内在声音无响应：无适配器)"

    system_prompt = PROMPTS[intent]
    if intent == "simulate":
        world = _build_world_context(session)
        if world:
            system_prompt = system_prompt + "\n\n" + world

    gen = _get_gen()
    logger.info("[think_deeply] 调用内在声音 intent=%s", intent)

    result = adapter.call_simple_text(
        system_prompt,
        content,
        gen,
        log_tag=f"think_deeply/{intent}",
    )
    if result:
        return result
    logger.warning("[think_deeply] 内在声音未返回内容，intent=%s", intent)
    return "(内在声音无响应)"