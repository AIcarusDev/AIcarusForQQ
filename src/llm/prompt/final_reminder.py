"""final_reminder.py — user prompt 末尾系统提醒块生成。"""


_FINAL_REMINDER_TEMPLATE_BROWSING = """\
<final_reminder>

# 提醒：

- 你正在翻阅该聊天窗口的历史记录，`<chat_logs>` 中显示的不是最新消息，而是当前窗口的历史聊天记录。
</final_reminder>"""

_FORWARD_BROWSER_REMINDER_TEMPLATE = """\
<forward_browser_reminder>

# 提醒：

- 当前打开着一个合并转发浏览窗口，你可以根据需要用 `browse_forward` 翻页，或用 `browse_forward` 的 action=open 打开嵌套合并转发。
- 如果这个合并转发窗口已经用不到了，记得用 `browse_forward` 关闭它；只退出当前层用 `back`，全部关闭用 `close_all`。
- 如果打开了`<current_session>`中的其它合并转发窗口，当前窗口会自动关闭并被替代。
- 如果你使用 `enter_qq_session` 进入其它 QQ 会话，当前合并转发浏览窗口会自动关闭。
</forward_browser_reminder>"""


def build_browsing_reminder(session) -> str:
    """浏览态提醒：告知模型当前在翻阅历史。"""
    if not session.is_browsing_history():
        return ""
    return _FINAL_REMINDER_TEMPLATE_BROWSING


def build_forward_browser_reminder(session) -> str:
    """合并转发浏览窗口提醒：只要窗口打开，就注入到末尾 <system_reminder>。"""
    if not session.is_browsing_forward():
        return ""
    return _FORWARD_BROWSER_REMINDER_TEMPLATE


def _build_error_logger_block(session) -> str:
    """若 pending_error_logger 非空，返回日志块并清空字段；否则返回空字符串。"""
    content = getattr(session, "pending_error_logger", "")
    if not content:
        return ""
    session.pending_error_logger = ""
    return f"<error_logger>\n```log\n{content}\n```\n</error_logger>"


def _build_system_reminder_block(*blocks: str) -> str:
    """将附加提醒统一包裹进 <system_reminder>。"""
    parts = [block for block in blocks if block]
    if not parts:
        return "<system_reminder/>"
    return "<system_reminder>\n" + "\n\n".join(parts) + "\n</system_reminder>"


def append_final_reminder(chat_log: "str | list", session) -> "str | list":
    """将 <system_reminder> 追加到 chat_log 末尾并返回。

    历史浏览提醒、错误日志和合并转发浏览提醒可并列显示。
    """
    error_block = _build_error_logger_block(session)
    browsing_block = build_browsing_reminder(session)
    forward_browser_block = build_forward_browser_reminder(session)

    system_reminder = _build_system_reminder_block(
        error_block,
        browsing_block,
        forward_browser_block,
    )
    if isinstance(chat_log, str):
        return chat_log + "\n" + system_reminder

    # chat_log 为多模态 list 时（聊天记录含图片），将纯文本块合并到末尾文本块
    last = chat_log[-1] if chat_log else None
    if isinstance(last, dict) and last.get("type") == "text":
        return chat_log[:-1] + [{**last, "text": last["text"] + "\n" + system_reminder}]
    return chat_log + [{"type": "text", "text": "\n" + system_reminder}]
