"""final_reminder.py — 连发提醒块生成

当 bot 在当前会话末尾连续发送了多条消息、而最后一条用户消息距今很近时，
在 user prompt 末尾插入 <final_reminder> 块，提示 bot 不要继续催促对方。
"""

from datetime import datetime, timezone


_FINAL_REMINDER_TEMPLATE = """\
<final_reminder>

# 提醒：

- 在当前聊天窗口的最后，你已经**连续发送了{N}条消息**，目前似乎没有人回应你
- 最后一条来自其它用户的消息距现在**仅过了{time}**
- 别人可能正在打字/休息/忙于其它事情/本来就没打算一直聊，或是话题已经告一段落

# 建议：

- **不要发送消息催促他人回复**，不要发送例如"怎么不说话了"/"人呢"这样令人厌恶，显得自己毫无分寸的内容
- **不建议继续发送消息**，如果你想主动提起另一个话题，或觉得有什么必须要说的，请确保一切是恰当合适的
- 如果还需要专注当前会话，考虑选择"wait"
- 如果对话已经告一段落，考虑选择"idle"
- 如果想看看别的会话，考虑选择"shift"

</final_reminder>"""

# 触发阈值
_MIN_TRAILING_BOT_MESSAGES = 2
_MAX_ELAPSED_SECONDS = 300  # 5 分钟


def _count_trailing_bot_messages(messages: list) -> int:
    """从上下文末尾统计连续 bot 消息数（note 条目透明跳过，遇到其它 role 则终止）。"""
    count = 0
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "bot":
            count += 1
        elif role == "note":
            continue
        else:
            break
    return count


def _get_last_user_message_elapsed_seconds(messages: list, now: datetime) -> "float | None":
    """返回最后一条非 bot 非 note 消息距 now 的秒数，找不到则返回 None。"""
    _now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role in ("bot", "note"):
            continue
        ts_str = msg.get("timestamp", "")
        if not ts_str:
            return None
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (_now - ts).total_seconds()
        except Exception:
            return None
    return None


def _format_elapsed_seconds(seconds: float) -> str:
    """将秒数格式化为人类可读的短描述（供 final_reminder 使用）。"""
    if seconds < 10:
        return "几秒钟"
    if seconds < 40:
        return "几十秒"
    if seconds < 60:
        return f"约{int(seconds)}秒"
    return f"约{int(seconds // 60)}分钟"


def build_final_reminder(session) -> str:
    """若满足条件返回 final_reminder 字符串，否则返回空字符串。

    触发条件（同时满足）：
    - 配置中 generation.final_reminder 未被关闭
    - 末尾连续 bot 消息数 >= _MIN_TRAILING_BOT_MESSAGES
    - 最后一条非 bot 消息距现在 < _MAX_ELAPSED_SECONDS 秒
    """
    import app_state
    if not app_state.config.get("generation", {}).get("final_reminder", True):
        return ""

    messages = session.context_messages
    trailing = _count_trailing_bot_messages(messages)
    if trailing < _MIN_TRAILING_BOT_MESSAGES:
        return ""

    tz = getattr(session, "_timezone", None)
    now = datetime.now(tz) if tz else datetime.now(timezone.utc)
    elapsed = _get_last_user_message_elapsed_seconds(messages, now)
    if elapsed is None or elapsed >= _MAX_ELAPSED_SECONDS:
        return ""

    return _FINAL_REMINDER_TEMPLATE.format(N=trailing, time=_format_elapsed_seconds(elapsed))


def append_final_reminder(chat_log: "str | list", session) -> "str | list":
    """若条件满足，将 final_reminder 追加到 chat_log 末尾并返回；否则原样返回。"""
    reminder = build_final_reminder(session)
    if not reminder:
        return chat_log

    if isinstance(chat_log, str):
        return chat_log + "\n" + reminder

    # chat_log 为多模态 list 时（聊天记录含图片），将纯文本 reminder 合并到末尾文本块
    last = chat_log[-1] if chat_log else None
    if isinstance(last, dict) and last.get("type") == "text":
        return chat_log[:-1] + [{**last, "text": last["text"] + "\n" + reminder}]
    return chat_log + [{"type": "text", "text": "\n" + reminder}]
