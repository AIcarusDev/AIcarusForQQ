"""user prompt 末尾预留的空系统提醒块。"""


_EMPTY_SYSTEM_REMINDER = "<system_reminder/>"


def append_final_reminder(chat_log: "str | list", session) -> "str | list":
    """在 chat_log 末尾保留空的 <system_reminder/> 逻辑位置。"""
    _ = session  # 保留接口，之后重新安排末尾提醒内容时使用。
    if isinstance(chat_log, str):
        return chat_log + "\n" + _EMPTY_SYSTEM_REMINDER

    # chat_log 为多模态 list 时（聊天记录含图片），将纯文本块合并到末尾文本块
    last = chat_log[-1] if chat_log else None
    if isinstance(last, dict) and last.get("type") == "text":
        return chat_log[:-1] + [
            {**last, "text": last["text"] + "\n" + _EMPTY_SYSTEM_REMINDER}
        ]
    return chat_log + [{"type": "text", "text": "\n" + _EMPTY_SYSTEM_REMINDER}]
