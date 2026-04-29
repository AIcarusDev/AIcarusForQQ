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
from .history_window import load_history_window
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


def _build_window_status_tag(unread: int) -> str:
    """构建 <window_status> 标签，显示历史浏览模式及底部未读数。"""
    if unread <= 0:
        return '<window_status mode="history"/>'
    unread_text = "99+" if unread > 99 else str(unread)
    return (
        f'<window_status mode="history" unread_below="{unread_text}">'
        f'该会话有 {unread_text} 条未读新消息</window_status>'
    )


def _inject_before_conversation_close(chat_log: "str | list", tag: str) -> "str | list":
    """在 </conversation> 闭合标签前插入 tag 行。"""
    marker = "</conversation>"
    if isinstance(chat_log, str):
        idx = chat_log.rfind(marker)
        if idx >= 0:
            return chat_log[:idx] + tag + "\n" + chat_log[idx:]
        return chat_log + "\n" + tag
    # 多模态 list：修改最后一个 text part
    for i in range(len(chat_log) - 1, -1, -1):
        part = chat_log[i]
        if isinstance(part, dict) and part.get("type") == "text":
            text = part["text"]
            idx = text.rfind(marker)
            if idx >= 0:
                new_text = text[:idx] + tag + "\n" + text[idx:]
            else:
                new_text = text + "\n" + tag
            return chat_log[:i] + [{**part, "text": new_text}] + chat_log[i + 1:]
    return chat_log + [{"type": "text", "text": "\n" + tag}]


def _build_browsing_chat_log(session) -> "str | list":
    """浏览态聊天记录构建：从 DB 取 page_size 条历史消息后走与 live 一致的渲染路径。"""
    view = session.chat_window_view
    top_db_id = view.get("top_db_id")
    page_size = int(view.get("page_size", 10))
    if not top_db_id:
        # 状态异常：兜底回 live 渲染，避免空 prompt
        return session.build_chat_log_xml()

    msgs = load_history_window(session, int(top_db_id), page_size)
    if not msgs:
        return session.build_chat_log_xml()

    unread = session.consume_visible_unread_messages(msgs)

    conv_meta = session._get_conv_meta()
    chat_log = build_multimodal_content(msgs, conv_meta, quoted_extra=session.quoted_extra)
    return _inject_before_conversation_close(chat_log, _build_window_status_tag(unread))


def build_main_user_prompt(session, *, consume_unread: bool = True) -> "str | list":
    """组装主模型本轮 user prompt。

    浏览态（session.is_browsing_history() 为真）下：
    - 不消费 unread_count，<window_status> 显示当前会话的未读新消息
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
        chat_log = session.build_chat_log_xml()
    user_prompt = _wrap_chat_log_with_world(chat_log, unread_xml)
    prefix = "\n".join([
        _build_prompt_block("style", session._style_prompt),
        _build_prompt_block("social_tips", session.get_social_tips()),
    ])
    user_prompt = _prepend_text_block(user_prompt, prefix)
    return append_final_reminder(user_prompt, session)
