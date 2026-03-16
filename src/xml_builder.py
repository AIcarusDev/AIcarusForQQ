"""xml_builder.py — 聊天记录 → XML 转化

将内部上下文消息列表转为 XML 格式，供 LLM 上下文使用。
日志/前端展示使用 format_chat_log_for_display() 获取可读版本。
"""

import html
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


def build_chat_log_xml(context_messages: list[dict]) -> str:
    """将上下文消息列表转为 XML 字符串。

    每条消息格式:
        {"message_id", "sender_id", "sender_name", "timestamp", "content"}

    输出示例:
        <chat_log>
          <message id="msg_001" sender_id="10001"
                   sender_name="张三" timestamp="3分钟前">
            早上好各位
          </message>
        </chat_log>
    """
    if not context_messages:
        return "<chat_log>\n</chat_log>"

    lines = ["<chat_log>"]
    for msg in context_messages:
        safe_content = html.escape(msg["content"], quote=False)
        safe_name = html.escape(msg["sender_name"])
        rel_time = _format_relative_time(msg["timestamp"])
        lines.append(
            f'  <message id="{msg["message_id"]}" '
            f'sender_id="{msg["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{rel_time}">'
        )
        lines.append(f"    {safe_content}")
        lines.append("  </message>")
    lines.append("</chat_log>")
    return "\n".join(lines)


def build_multimodal_content(
    context_messages: list[dict],
    max_images: int = 5,
) -> "str | list":
    """将上下文消息列表转为 LLM 可用的内容。

    图片精准内嵌：每条携带图片的消息，其图片 content part 紧跟该消息的 XML 标签之后，
    模型注意力自然落在"消息文字 → 对应图片"的顺序上。

    只嵌入最新的 max_images 张图片；更早的图片消息保留 [图片] 文字占位符。
    无图片时退回纯字符串，与原有逻辑完全兼容。
    """
    if not context_messages:
        return "<chat_log>\n</chat_log>"

    # 找出所有有图片的消息下标，取最新的 max_images 个作为"有效嵌入集"
    image_indices = [
        i for i, m in enumerate(context_messages) if m.get("images")
    ]
    eligible: set[int] = set(image_indices[-max_images:]) if image_indices else set()

    if not eligible:
        # 无需多模态，直接返回纯 XML
        return build_chat_log_xml(context_messages)

    parts: list[dict] = []
    text_buf: list[str] = ["<chat_log>"]

    for i, msg in enumerate(context_messages):
        safe_content = html.escape(msg["content"], quote=False)
        safe_name = html.escape(msg["sender_name"])
        rel_time = _format_relative_time(msg["timestamp"])
        text_buf.append(
            f'  <message id="{msg["message_id"]}" '
            f'sender_id="{msg["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{rel_time}">'
        )
        text_buf.append(f"    {safe_content}")
        text_buf.append("  </message>")

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

    # 收尾：关闭 chat_log 标签，将剩余文字缓冲区输出
    text_buf.append("</chat_log>")
    parts.append({"type": "text", "text": "\n".join(text_buf)})
    return parts


def format_chat_log_for_display(context_messages: list[dict]) -> str:
    """将上下文消息格式化为可读 XML，用于前端/日志展示。

    图片字段不嵌入 base64，只显示数量提示。
    """
    if not context_messages:
        return "<chat_log>\n</chat_log>"

    lines = ["<chat_log>"]
    for msg in context_messages:
        safe_content = html.escape(msg["content"], quote=False)
        safe_name = html.escape(msg["sender_name"])
        rel_time = _format_relative_time(msg["timestamp"])
        img_hint = ""
        if msg.get("images"):
            img_hint = f" [{len(msg['images'])}张图片]"
        lines.append(
            f'  <message id="{msg["message_id"]}" '
            f'sender_id="{msg["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{rel_time}">'
        )
        lines.append(f"    {safe_content}{img_hint}")
        lines.append("  </message>")
    lines.append("</chat_log>")
    return "\n".join(lines)
