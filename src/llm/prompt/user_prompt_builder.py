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
from .unread_builder import build_unread_info_xml
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


def build_main_user_prompt(session, *, consume_unread: bool = True) -> "str | list":
    """组装主模型本轮 user prompt。"""
    current_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
    if consume_unread:
        session.unread_count = 0

    unread_xml = build_unread_info_xml(sessions, current_key)
    chat_log = session.build_chat_log_xml()
    user_prompt = _wrap_chat_log_with_world(chat_log, unread_xml)
    prefix = "\n".join([
        _build_prompt_block("style", session._style_prompt),
        _build_prompt_block("social_tips", session.get_social_tips()),
    ])
    user_prompt = _prepend_text_block(user_prompt, prefix)
    return append_final_reminder(user_prompt, session)