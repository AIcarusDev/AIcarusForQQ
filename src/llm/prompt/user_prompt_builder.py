"""user_prompt_builder.py — 主模型 user prompt 总装

统一组装主模型每轮调用的 user content。
当前包括：
- <memory> 块
- <goals> 块
- <skills> 块（仅在 active namespace 绑定主 skill 时出现）
- <world> 顶层包裹
- <current_time> 块
- <attention_events> 块
- <unread_info> 块
- <platform> 内层包裹
- <des> 平台说明块
- 聊天记录 XML / 多模态内容
- <system_reminder> 末尾附加块
"""

import html
import logging

import browser
from platforms.attention import build_attention_events_xml
from platforms.base import PlatformWorldBlock
from platforms.registry import get_platform
from skills import build_skill_block_for_namespaces
from tools.namespaces import load_namespace_registry

from .final_reminder import append_final_reminder
from platforms.chat.history_window import has_previous_messages, load_history_window
from platforms.chat.xml_builder import build_forward_browser_content, build_multimodal_content
from ..compression.config import (
    DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
    normalize_generation_config,
    normalize_world_multimodal_image_limit,
)

logger = logging.getLogger("AICQ.llm.prompt.user_prompt_builder")


def _build_prompt_block(tag: str, content: str) -> str:
    """构建一个简单的 XML 文本块。"""
    body = content.strip("\r\n")
    if body.strip():
        return f"<{tag}>\n{body}\n</{tag}>"
    return f"<{tag}>\n</{tag}>"


def _prepend_text_block(content: "str | list", text: str) -> "str | list":
    """给 user prompt 前部插入纯文本块。"""
    if isinstance(content, str):
        return text + "\n" + content
    return [{"type": "text", "text": text + "\n"}] + content


def _build_active_skill_prompt_block() -> str:
    try:
        import app_state

        state = getattr(app_state, "namespace_runtime_state", None)
        if state is None:
            return ""
        registry = load_namespace_registry()
        active_namespaces = state.active_namespaces(registry)
        return build_skill_block_for_namespaces(active_namespaces, registry)
    except Exception:
        logger.warning("构建 active skill prompt block 失败", exc_info=True)
        return ""


def _append_text_part(parts: list, text: str) -> None:
    if not text:
        return
    if parts and isinstance(parts[-1], dict) and parts[-1].get("type") == "text":
        parts[-1] = {**parts[-1], "text": parts[-1].get("text", "") + text}
    else:
        parts.append({"type": "text", "text": text})


def _platform_open_tag(
    platform_name: str,
    attrs: dict[str, str] | None = None,
) -> str:
    safe_platform = html.escape(str(platform_name or "qq"), quote=True)
    rendered = [f'name="{safe_platform}"']
    for key, value in (attrs or {}).items():
        safe_key = html.escape(str(key), quote=True)
        safe_value = html.escape(str(value or ""), quote=True)
        rendered.append(f'{safe_key}="{safe_value}"')
    return f"<platform {' '.join(rendered)}>"


def _platform_self_closing_tag(
    platform_name: str,
    attrs: dict[str, str] | None = None,
) -> str:
    rendered: list[str] = []
    if str(platform_name or "").strip():
        safe_platform = html.escape(str(platform_name), quote=True)
        rendered.append(f'name="{safe_platform}"')
    for key, value in (attrs or {}).items():
        safe_key = html.escape(str(key), quote=True)
        safe_value = html.escape(str(value or ""), quote=True)
        rendered.append(f'{safe_key}="{safe_value}"')
    return f"<platform {' '.join(rendered)}/>" if rendered else "<platform/>"


def _is_image_url_part(part: dict) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def _limit_multimodal_image_parts(content: "str | list", limit: int) -> "str | list":
    """Keep at most the last ``limit`` real image_url parts in the prompt."""
    if isinstance(content, str) or limit < 0:
        return content
    image_count = sum(1 for part in content if _is_image_url_part(part))
    overflow = image_count - limit
    if overflow <= 0:
        return content

    limited: list = []
    remaining_to_drop = overflow
    for part in content:
        if _is_image_url_part(part) and remaining_to_drop > 0:
            remaining_to_drop -= 1
            continue
        limited.append(part)
    return limited


def _world_multimodal_image_limit() -> int:
    """Read the runtime cap for real multimodal images inside <world>."""
    try:
        import app_state

        cfg = getattr(app_state, "config", {}) or {}
        if not bool(cfg.get("vision", True)):
            return -1
        gen = getattr(app_state, "GEN", None) or cfg.get("generation")
        return normalize_generation_config(gen).get(
            "world_multimodal_image_limit",
            DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
        )
    except Exception:
        return DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT


def _chat_log_multimodal_image_hint(limit: int) -> int:
    """Return the legacy per-chat-log hint that avoids avoidable old image work."""
    return -1 if limit < 0 else limit


def _wrap_platform_block_with_world(
    block: PlatformWorldBlock,
    current_time: str,
) -> "str | list":
    """Wrap a platform-provided content block in the stable world shell."""
    current_time_block = f"<current_time>{current_time}</current_time>"
    attention_events_block = build_attention_events_xml(current_platform=block.name)
    if block.content is None:
        platform_tag = _platform_self_closing_tag(block.name, block.attrs)
        return f"<world>\n{current_time_block}\n{attention_events_block}\n{platform_tag}\n</world>"

    platform_open = _platform_open_tag(block.name, block.attrs)
    content = block.content or ""
    if isinstance(content, str):
        return (
            f"<world>\n{current_time_block}\n{attention_events_block}\n{platform_open}\n"
            f"{content}\n"
            "</platform>\n</world>"
        )

    new_parts: list = [
        {
            "type": "text",
            "text": f"<world>\n{current_time_block}\n{attention_events_block}\n{platform_open}\n",
        }
    ]
    new_parts.extend(content)
    _append_text_part(new_parts, "\n</platform>")
    _append_text_part(new_parts, "\n</world>")
    return new_parts


def _merge_platform_content(content: "str | list", extra: "str | list") -> "str | list":
    if not extra:
        return content
    if isinstance(content, str) and isinstance(extra, str):
        return f"{content}\n{extra}"
    parts: list = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    _append_text_part(parts, "\n")
    if isinstance(extra, str):
        _append_text_part(parts, extra)
    else:
        parts.extend(extra)
    return parts


def _fallback_platform_block(
    session,
    chat_log: "str | list",
    forward_content: "str | list",
) -> PlatformWorldBlock:
    return PlatformWorldBlock(
        name=session.get_platform_key(),
        content=_merge_platform_content(chat_log, forward_content),
    )


def _strip_world_close(content: "str | list") -> tuple["str | list", str]:
    suffix = "\n</world>"
    if isinstance(content, str):
        if content.endswith(suffix):
            return content[: -len(suffix)], suffix
        if content.endswith("</world>"):
            return content[: -len("</world>")], "</world>"
        return content, ""

    parts = list(content)
    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = str(part.get("text", ""))
        marker = text.rfind("</world>")
        if marker < 0 or text[marker + len("</world>"):].strip():
            continue
        before = text[:marker]
        if before.endswith("\n"):
            before = before[:-1]
            close = "\n</world>"
        else:
            close = "</world>"
        parts = parts[: index + 1]
        if before:
            parts[index] = {**part, "text": before}
        else:
            parts = parts[:index]
        return parts, close
    return parts, ""


def _append_browser_content_to_world(
    content: "str | list",
    browser_content: "str | list",
) -> "str | list":
    if not browser_content:
        return content

    opened, close = _strip_world_close(content)
    close = close or "\n</world>"
    if isinstance(opened, str) and isinstance(browser_content, str):
        return f"{opened}\n{browser_content}{close}"

    parts: list = [{"type": "text", "text": opened}] if isinstance(opened, str) else list(opened)
    _append_text_part(parts, "\n")
    if isinstance(browser_content, str):
        _append_text_part(parts, browser_content)
    else:
        parts.extend(browser_content)
    _append_text_part(parts, close)
    return parts


def _build_unread_bubble_text(unread: int) -> str:
    """构建浏览态底部未读气泡文案。"""
    if unread <= 0:
        return ""
    unread_text = "99+" if unread > 99 else str(unread)
    return f"当前会话有 {unread_text} 条未读新消息"


def _build_current_chat_log(session) -> "str | list":
    """最新窗口聊天记录构建：统一输出 current 模式与 has_previous 状态。"""
    conv_meta = session._get_conv_meta()
    world_image_limit = _world_multimodal_image_limit()
    return build_multimodal_content(
        session.context_messages,
        conv_meta,
        max_images=_chat_log_multimodal_image_hint(world_image_limit),
        quoted_extra=session.quoted_extra,
        chat_logs_mode="current",
        has_previous=has_previous_messages(session, browsing=False),
    )


def _has_current_session(session) -> bool:
    """当前 QQ platform 是否已经打开具体会话窗口。"""
    focus = getattr(session, "focus", None)
    if focus is None:
        return False
    try:
        from platforms.qq.session_context import is_qq_home_focus

        if is_qq_home_focus(focus):
            return False
    except Exception:
        pass
    return True


def _build_browsing_chat_log(session) -> "str | list":
    """浏览态聊天记录构建：统一输出 history 模式、has_previous 与未读气泡。"""
    view = session.chat_window_view
    top_db_id = view.get("top_db_id")
    page_size = int(view.get("page_size", 10))
    if not top_db_id:
        # 状态异常：兜底回 live 渲染，避免空 prompt
        return _build_current_chat_log(session)

    msgs = load_history_window(session, int(top_db_id), page_size)
    if not msgs:
        return _build_current_chat_log(session)

    unread = session.consume_visible_unread_messages(msgs)

    conv_meta = session._get_conv_meta()
    world_image_limit = _world_multimodal_image_limit()
    return build_multimodal_content(
        msgs,
        conv_meta,
        max_images=_chat_log_multimodal_image_hint(world_image_limit),
        quoted_extra=session.quoted_extra,
        chat_logs_mode="history",
        has_previous=has_previous_messages(session, browsing=True, top_db_id=int(top_db_id)),
        bubble_text=_build_unread_bubble_text(unread),
    )


def build_main_user_prompt(session, *, consume_unread: bool = True) -> "str | list":
    """组装主模型本轮 user prompt。

    浏览态（session.is_browsing_history() 为真）下：
    - 聊天记录 XML 统一输出 <chat_logs mode="..." has_previous="...">
    - 浏览态不消费 unread_count，未读新消息以 <bubble> 出现在 <chat_logs> 内
    - 聊天记录从 DB 加载历史窗口，而非渲染最新 context
    """
    browsing = session.is_browsing_history()

    if consume_unread and not browsing:
        session.clear_unread_messages()

    if browsing:
        chat_log = _build_browsing_chat_log(session)
    elif not _has_current_session(session):
        chat_log = "<current_session/>"
    else:
        chat_log = _build_current_chat_log(session)
    forward_content = build_forward_browser_content(session)
    dynamic_blocks = session.build_dynamic_prompt_blocks()
    browser_content = browser.build_browser_world_content()
    runtime = get_platform(session.get_platform_key())
    if runtime is not None:
        platform_block = runtime.world_block(
            session,
            current_time=dynamic_blocks["current_time"],
            chat_log=chat_log,
            forward_content=forward_content,
        )
    else:
        platform_block = _fallback_platform_block(session, chat_log, forward_content)
    user_prompt = _wrap_platform_block_with_world(
        platform_block,
        dynamic_blocks["current_time"],
    )
    user_prompt = _limit_multimodal_image_parts(
        user_prompt,
        normalize_world_multimodal_image_limit(_world_multimodal_image_limit()),
    )
    user_prompt = _append_browser_content_to_world(user_prompt, browser_content)
    prefix_parts = [
        _build_prompt_block("memory", dynamic_blocks["memory"]),
        _build_prompt_block("goals", dynamic_blocks["goals"]),
    ]
    if skill_block := _build_active_skill_prompt_block():
        prefix_parts.append(skill_block)
    prefix = "\n".join(prefix_parts)
    user_prompt = _prepend_text_block(user_prompt, prefix)
    return append_final_reminder(user_prompt, session)
