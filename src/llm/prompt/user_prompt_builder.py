"""user_prompt_builder.py — 主模型 user prompt 总装

统一组装主模型每轮调用的 user content。
当前包括：
- <unread_info> 块
- <qq> 顶层包裹
- 聊天记录 XML / 多模态内容
- error_logger / final_reminder 等末尾附加块
"""

from .final_reminder import append_final_reminder
from .unread_builder import build_unread_info_xml
from ..session import sessions


def _wrap_chat_log_with_qq(chat_log: "str | list", unread_xml: str) -> "str | list":
    """将聊天记录用 <qq> 包裹，并在前面插入 unread_info 块。"""
    unread_block = unread_xml if unread_xml else "<unread_info/>"
    if isinstance(chat_log, str):
        return f"<qq>\n{unread_block}\n{chat_log}\n</qq>"

    new_parts: list = [{"type": "text", "text": f"<qq>\n{unread_block}\n"}]
    new_parts.extend(chat_log)
    last = new_parts[-1]
    if isinstance(last, dict) and last.get("type") == "text":
        new_parts[-1] = {**last, "text": last["text"] + "\n</qq>"}
    else:
        new_parts.append({"type": "text", "text": "\n</qq>"})
    return new_parts


def build_main_user_prompt(session, *, consume_unread: bool = True) -> "str | list":
    """组装主模型本轮 user prompt。"""
    current_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
    if consume_unread:
        session.unread_count = 0

    unread_xml = build_unread_info_xml(sessions, current_key)
    chat_log = session.build_chat_log_xml()
    user_prompt = _wrap_chat_log_with_qq(chat_log, unread_xml)
    return append_final_reminder(user_prompt, session)