"""user_prompt_builder.py — 主模型 user prompt 总装

统一组装主模型每轮调用的 user content。
当前包括：
- <memory> 块
- <goals> 块
- <style> 块
- <social_tips> 块
- <world> 顶层包裹
- <current_time> 块
- <unread_info> 块
- <qq> 内层包裹
- 聊天记录 XML / 多模态内容
- <system_reminder> 末尾附加块
"""

import browser

from .final_reminder import append_final_reminder
from .history_window import has_previous_messages, load_history_window
from .unread_builder import build_unread_info_xml
from .xml_builder import build_forward_browser_content, build_multimodal_content
from ..compression.config import (
    DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
    normalize_generation_config,
    normalize_world_multimodal_image_limit,
)
from ..session import sessions


def _build_prompt_block(tag: str, content: str) -> str:
    """构建一个简单的 XML 文本块。"""
    normalized = content.strip()
    if normalized:
        return f"<{tag}>\n{normalized}\n</{tag}>"
    return f"<{tag}>\n</{tag}>"


def _prepend_text_block(content: "str | list", text: str) -> "str | list":
    """给 user prompt 前部插入纯文本块。"""
    if isinstance(content, str):
        return text + "\n" + content
    return [{"type": "text", "text": text + "\n"}] + content


def _append_text_part(parts: list, text: str) -> None:
    if not text:
        return
    if parts and isinstance(parts[-1], dict) and parts[-1].get("type") == "text":
        parts[-1] = {**parts[-1], "text": parts[-1].get("text", "") + text}
    else:
        parts.append({"type": "text", "text": text})


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


def _wrap_chat_log_with_world(
    chat_log: "str | list",
    unread_xml: str,
    current_time: str,
    forward_content: "str | list" = "",
    browser_content: "str | list" = "",
) -> "str | list":
    """将聊天记录用 <world><qq> 包裹，并在前面插入 unread_info 块。"""
    unread_block = unread_xml if unread_xml else "<unread_info/>"
    current_time_block = f"<current_time>{current_time}</current_time>"
    if (
        isinstance(chat_log, str)
        and not isinstance(forward_content, list)
        and not isinstance(browser_content, list)
    ):
        forward_block = f"\n{forward_content}" if forward_content else ""
        browser_block = f"\n{browser_content}" if browser_content else ""
        return f"<world>\n{current_time_block}\n<qq>\n{unread_block}\n{chat_log}{forward_block}\n</qq>{browser_block}\n</world>"

    new_parts: list = [{"type": "text", "text": f"<world>\n{current_time_block}\n<qq>\n{unread_block}\n"}]
    if isinstance(chat_log, str):
        _append_text_part(new_parts, chat_log)
    else:
        new_parts.extend(chat_log)
    if forward_content:
        _append_text_part(new_parts, "\n")
        if isinstance(forward_content, str):
            _append_text_part(new_parts, forward_content)
        else:
            new_parts.extend(forward_content)
    _append_text_part(new_parts, "\n</qq>")
    if browser_content:
        _append_text_part(new_parts, "\n")
        if isinstance(browser_content, str):
            _append_text_part(new_parts, browser_content)
        else:
            new_parts.extend(browser_content)
    _append_text_part(new_parts, "\n</world>")
    return new_parts


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
    current_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
    browsing = session.is_browsing_history()

    if consume_unread and not browsing:
        session.clear_unread_messages()

    unread_xml = build_unread_info_xml(sessions, current_key)
    if browsing:
        chat_log = _build_browsing_chat_log(session)
    else:
        chat_log = _build_current_chat_log(session)
    forward_content = build_forward_browser_content(session)
    dynamic_blocks = session.build_dynamic_prompt_blocks()
    browser_content = browser.build_browser_world_content()
    user_prompt = _wrap_chat_log_with_world(
        chat_log,
        unread_xml,
        dynamic_blocks["current_time"],
        forward_content,
    )
    user_prompt = _limit_multimodal_image_parts(
        user_prompt,
        normalize_world_multimodal_image_limit(_world_multimodal_image_limit()),
    )
    user_prompt = _append_browser_content_to_world(user_prompt, browser_content)
    prefix = "\n".join([
        _build_prompt_block("memory", dynamic_blocks["memory"]),
        _build_prompt_block("goals", dynamic_blocks["goals"]),
        _build_prompt_block("style", session._style_prompt),
        _build_prompt_block("social_tips", session.get_social_tips()),
    ])
    user_prompt = _prepend_text_block(user_prompt, prefix)
    return append_final_reminder(user_prompt, session)
