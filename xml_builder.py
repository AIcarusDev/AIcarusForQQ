"""xml_builder.py — 聊天记录格式化

将内部上下文消息列表转为紧凑 JSON，供 LLM 上下文使用。
日志/前端展示使用 format_chat_log_for_display() 获取可读版本。
"""

import json
from datetime import datetime, timezone


def _format_relative_time(iso_timestamp: str) -> str:
    """将 ISO 格式时间戳转换为相对时间字符串。"""
    try:
        past = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(timezone.utc)
        if past.tzinfo is None:
            past = past.replace(tzinfo=timezone.utc)
        delta = (now - past).total_seconds()
    except (ValueError, TypeError):
        return iso_timestamp

    if delta < 10:
        return "刚刚"
    if delta < 60:
        return f"{int(delta)}秒前"
    minutes = delta / 60
    if minutes < 60:
        return f"{int(minutes)}分钟前"
    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}小时前"
    days = hours / 24
    if days < 2:
        return "昨天"
    if days < 30:
        return f"{int(days)}天前"
    months = days / 30
    if months < 12:
        return f"{int(months)}个月前"
    return f"{int(days / 365)}年前"


def _msg_to_record(msg: dict) -> dict:
    """内部消息字典 → 发送给模型的精简记录。"""
    return {
        "id": msg["message_id"],
        "sender_id": msg["sender_id"],
        "sender_name": msg["sender_name"],
        "timestamp": _format_relative_time(msg["timestamp"]),
        "content": msg["content"],
    }


def build_chat_log(context_messages: list[dict]) -> str:
    """将上下文消息列表转为紧凑 JSON 数组字符串，直接发给模型。"""
    if not context_messages:
        return "[]"
    records = [_msg_to_record(m) for m in context_messages]
    return json.dumps(records, ensure_ascii=False, separators=(",", ":"))


def build_multimodal_content(
    context_messages: list[dict],
    max_images: int = 5,
) -> "str | list":
    """将上下文消息列表转为 LLM 可用的内容。

    图片精准内嵌：每条携带图片的消息，其图片 content part 紧跟该消息的 JSON 之后，
    模型注意力自然落在"消息文字 → 对应图片"的顺序上。

    只嵌入最新的 max_images 张图片；更早的图片消息保留 [图片] 文字占位符。
    无图片时退回纯字符串，与原有逻辑完全兼容。
    """
    if not context_messages:
        return "[]"

    # 找出所有有图片的消息下标，取最新的 max_images 个作为"有效嵌入集"
    image_indices = [
        i for i, m in enumerate(context_messages) if m.get("images")
    ]
    eligible: set[int] = set(image_indices[-max_images:]) if image_indices else set()

    if not eligible:
        return build_chat_log(context_messages)

    parts: list[dict] = []
    text_buf: list[str] = []

    for i, msg in enumerate(context_messages):
        line = json.dumps(_msg_to_record(msg), ensure_ascii=False, separators=(",", ":"))
        text_buf.append(line)

        if i in eligible:
            # 将当前文字缓冲区作为一个 text part 输出，然后紧接图片 parts
            parts.append({"type": "text", "text": "\n".join(text_buf)})
            text_buf = []
            for img in msg["images"]:  # type: ignore[index]
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mime']};base64,{img['base64']}"
                        },
                    }
                )

    # 收尾：将剩余文字缓冲区输出
    if text_buf:
        parts.append({"type": "text", "text": "\n".join(text_buf)})
    return parts


def format_chat_log_for_display(context_messages: list[dict]) -> str:
    """将上下文消息格式化为可读 JSON，用于前端/日志展示。

    图片字段不嵌入 base64，只显示数量提示。
    """
    if not context_messages:
        return "[]"
    records = []
    for m in context_messages:
        d = _msg_to_record(m)
        if m.get("images"):
            d["images"] = f"[{len(m['images'])}张图片]"
        records.append(d)
    return json.dumps(records, ensure_ascii=False, indent=2)
