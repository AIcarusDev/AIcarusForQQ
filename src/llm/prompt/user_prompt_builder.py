"""user_prompt_builder.py — 主模型 user prompt 总装

统一组装主模型每轮调用的 user content。
当前包括：
- <style> 块
- <social_tips> 块
- <world> 顶层包裹
- <unread_info> 块
- <qq> 内层包裹
- 聊天记录 XML / 多模态内容
- <system_reminder> 末尾附加块
"""

from .final_reminder import append_final_reminder
from .history_window import has_previous_messages, load_history_window
from .unread_builder import build_unread_info_xml
from .xml_builder import build_multimodal_content
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


def _wrap_chat_log_with_world(chat_log: "str | list", unread_xml: str) -> "str | list":
    """将聊天记录用 <world><qq> 包裹，并在前面插入 unread_info 块。"""
    unread_block = unread_xml if unread_xml else "<unread_info/>"
    if isinstance(chat_log, str):
        return f"<world>\n<qq>\n{unread_block}\n{chat_log}\n</qq>\n</world>"

    new_parts: list = [{"type": "text", "text": f"<world>\n<qq>\n{unread_block}\n"}]
    new_parts.extend(chat_log)
    last = new_parts[-1]
    if isinstance(last, dict) and last.get("type") == "text":
        new_parts[-1] = {**last, "text": last["text"] + "\n</qq>\n</world>"}
    else:
        new_parts.append({"type": "text", "text": "\n</qq>\n</world>"})
    return new_parts


def _build_unread_bubble_text(unread: int) -> str:
    """构建浏览态底部未读气泡文案。"""
    if unread <= 0:
        return ""
    unread_text = "99+" if unread > 99 else str(unread)
    return f"当前会话有 {unread_text} 条未读新消息"


def _build_current_chat_log(session) -> "str | list":
    """最新窗口聊天记录构建：统一输出 current 模式与 has_previous 状态。"""
    conv_meta = session._get_conv_meta()
    return build_multimodal_content(
        session.context_messages,
        conv_meta,
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
    return build_multimodal_content(
        msgs,
        conv_meta,
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
    user_prompt = _wrap_chat_log_with_world(chat_log, unread_xml)
    prefix = "\n".join([
        _build_prompt_block("style", session._style_prompt),
        _build_prompt_block("social_tips", session.get_social_tips()),
    ])
    user_prompt = _prepend_text_block(user_prompt, prefix)
    return append_final_reminder(user_prompt, session)
