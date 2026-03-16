"""xml_builder.py — 聊天记录 → XML 转化

将内部上下文消息列表转为结构化 XML，供 LLM 上下文使用。

输出格式根据会话类型自动适配：
  - 群聊：完整 sender / role / mention / quote
  - 私聊：精简，去掉每条消息的 sender 块，bot 消息用 from="self" 标记
  - Web：兜底的通用格式

日志/前端展示使用 format_chat_log_for_display() 获取可读版本。
"""

import html
from datetime import datetime, timezone


# ── 时间格式化 ────────────────────────────────────────────

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


# ── 内容段渲染 ────────────────────────────────────────────

def _render_content_segments(segments: list[dict]) -> str:
    """将结构化 content_segments 渲染为 XML 内联内容。

    支持: text / mention / emoji / image / file / 其他占位
    """
    parts: list[str] = []
    for seg in segments:
        seg_type = seg.get("type", "")
        if seg_type == "text":
            parts.append(html.escape(seg.get("text", ""), quote=False))
        elif seg_type == "mention":
            uid = html.escape(str(seg.get("uid", "")))
            display = html.escape(seg.get("display", ""))
            parts.append(f'<mention uid="{uid}">{display}</mention>')
        elif seg_type == "emoji":
            eid = html.escape(str(seg.get("id", "")))
            name = html.escape(seg.get("name", ""))
            parts.append(f'<emoji id="{eid}" name="{name}"/>')
        elif seg_type == "image":
            parts.append("[图片]")
        elif seg_type == "file":
            fn = html.escape(seg.get("filename", "未知"))
            parts.append(f"[文件:{fn}]")
        else:
            label = seg.get("label", seg_type)
            parts.append(f"[{html.escape(label)}]")
    return "".join(parts)


def _render_content_text(content: str) -> str:
    """兜底：没有 content_segments 时用纯文本渲染。"""
    return html.escape(content, quote=False)


def _render_content(msg: dict) -> str:
    """选择结构化或纯文本渲染，返回 XML 内联字符串。"""
    segments = msg.get("content_segments")
    if segments:
        return _render_content_segments(segments)
    return _render_content_text(msg.get("content", ""))


# ── 回复引用 ─────────────────────────────────────────────

def _build_quote_xml(ref_id: str, context_messages: list[dict], indent: str) -> str | None:
    """根据 ref_id 在上下文中查找被引用的消息，构建 <quote> 标签。"""
    for m in context_messages:
        if str(m.get("message_id", "")) == ref_id:
            name = html.escape(m.get("sender_name", ""))
            raw = m.get("content", "")
            preview_text = raw[:50] + ("..." if len(raw) > 50 else "")
            preview = html.escape(preview_text, quote=False)
            return (
                f'{indent}<quote ref_id="{html.escape(ref_id)}">\n'
                f"{indent}  <preview>{name}: {preview}</preview>\n"
                f"{indent}</quote>"
            )
    # 引用的消息不在当前上下文窗口内
    return (
        f'{indent}<quote ref_id="{html.escape(ref_id)}">\n'
        f"{indent}  <preview>[上下文之外的消息]</preview>\n"
        f"{indent}</quote>"
    )


# ── conversation 开闭标签 ─────────────────────────────────

def _conv_open_tag(conv_meta: dict) -> str:
    """生成 <conversation ...> 开标签。"""
    conv_type = conv_meta.get("type", "")
    conv_id = html.escape(str(conv_meta.get("id", "")))
    conv_name = html.escape(conv_meta.get("name", ""))

    if conv_type == "group":
        attrs = f'type="group" id="{conv_id}"'
        if conv_name:
            attrs += f' name="{conv_name}"'
        return f"<conversation {attrs}>"
    elif conv_type == "private":
        attrs = f'type="private" id="{conv_id}"'
        if conv_name:
            attrs += f' nickname="{conv_name}"'
        return f"<conversation {attrs}>"
    else:
        return "<conversation>"


def _self_tag(meta: dict) -> str | None:
    """当有 bot_id 时生成 <self> 标签，供 LLM 交叉验证 from="self" 身份。"""
    bot_id = html.escape(str(meta.get("bot_id", "")))
    bot_name = html.escape(meta.get("bot_name", ""))
    if not bot_id:
        return None
    if bot_name:
        return f'<self id="{bot_id}" name="{bot_name}"/>'
    return f'<self id="{bot_id}"/>'


# ── 单条消息渲染 ─────────────────────────────────────────

def _render_message_group(msg: dict, context_messages: list[dict]) -> list[str]:
    """群聊模式：完整 sender + role + quote + content type。"""
    lines: list[str] = []
    rel_time = _format_relative_time(msg["timestamp"])
    msg_id = html.escape(str(msg["message_id"]))
    content_type = html.escape(msg.get("content_type", "text"))

    lines.append(f'  <message id="{msg_id}" timestamp="{rel_time}">')

    # <sender>
    sender_id = html.escape(str(msg.get("sender_id", "")))
    nickname = html.escape(msg.get("sender_name", ""))
    role = html.escape(msg.get("sender_role", ""))
    sender_attrs = f'id="{sender_id}" nickname="{nickname}"'
    if role:
        sender_attrs += f' role="{role}"'
    lines.append(f"    <sender {sender_attrs}/>")

    # <quote>（如果有引用）
    reply_to = msg.get("reply_to")
    if reply_to:
        quote_xml = _build_quote_xml(reply_to, context_messages, "    ")
        if quote_xml:
            lines.append(quote_xml)

    # <content>
    inner = _render_content(msg)
    lines.append(f'    <content type="{content_type}">{inner}</content>')

    lines.append("  </message>")
    return lines


def _render_message_private(msg: dict, conv_meta: dict, context_messages: list[dict]) -> list[str]:
    """私聊模式：精简，无 sender 块，bot 消息用 from="self"。"""
    lines: list[str] = []
    rel_time = _format_relative_time(msg["timestamp"])
    msg_id = html.escape(str(msg["message_id"]))
    content_type = html.escape(msg.get("content_type", "text"))

    # bot 自己的消息加 from="self"
    is_self = msg.get("role") == "bot"
    from_attr = ' from="self"' if is_self else ""
    lines.append(f'  <message id="{msg_id}" timestamp="{rel_time}"{from_attr}>')

    # <quote>（私聊也可以回复）
    reply_to = msg.get("reply_to")
    if reply_to:
        quote_xml = _build_quote_xml(reply_to, context_messages, "    ")
        if quote_xml:
            lines.append(quote_xml)

    inner = _render_content(msg)
    lines.append(f'    <content type="{content_type}">{inner}</content>')

    lines.append("  </message>")
    return lines


def _render_message_generic(msg: dict) -> list[str]:
    """Web / 通用模式：简单的 sender_name 属性 + content type。"""
    lines: list[str] = []
    rel_time = _format_relative_time(msg["timestamp"])
    msg_id = html.escape(str(msg["message_id"]))
    safe_name = html.escape(msg.get("sender_name", ""))
    content_type = html.escape(msg.get("content_type", "text"))

    lines.append(f'  <message id="{msg_id}" sender_name="{safe_name}" timestamp="{rel_time}">')
    inner = _render_content(msg)
    lines.append(f'    <content type="{content_type}">{inner}</content>')
    lines.append("  </message>")
    return lines


# ── 核心公共 API ─────────────────────────────────────────

_EMPTY_META: dict = {}


def build_chat_log_xml(
    context_messages: list[dict],
    conv_meta: dict | None = None,
) -> str:
    """将上下文消息列表转为结构化 XML 字符串（纯文本，不含图片 base64）。"""
    meta = conv_meta or _EMPTY_META
    conv_type = meta.get("type", "")

    if not context_messages:
        tag = _conv_open_tag(meta)
        self_line = _self_tag(meta)
        header = f"\n{self_line}" if self_line else ""
        return f"{tag}{header}\n<chat_logs>\n</chat_logs>\n</conversation>"

    lines: list[str] = [_conv_open_tag(meta)]
    self_line = _self_tag(meta)
    if self_line:
        lines.append(self_line)
    lines.append("<chat_logs>")

    for msg in context_messages:
        if conv_type == "group":
            lines.extend(_render_message_group(msg, context_messages))
        elif conv_type == "private":
            lines.extend(_render_message_private(msg, meta, context_messages))
        else:
            lines.extend(_render_message_generic(msg))

    lines.append("</chat_logs>")
    lines.append("</conversation>")
    return "\n".join(lines)


def build_multimodal_content(
    context_messages: list[dict],
    conv_meta: dict | None = None,
    max_images: int = 5,
) -> "str | list":
    """将上下文消息列表转为 LLM 可用的内容（纯 XML 或多模态 parts）。

    图片精准内嵌：每条携带图片的消息，其图片 content part 紧跟该消息的 XML 之后，
    模型注意力自然落在"消息文字 → 对应图片"的顺序上。

    只嵌入最新的 max_images 张图片；更早的图片消息保留 [图片] 文字占位符。
    无图片时退回纯字符串，与原有逻辑完全兼容。
    """
    meta = conv_meta or _EMPTY_META
    conv_type = meta.get("type", "")

    if not context_messages:
        return build_chat_log_xml(context_messages, conv_meta)

    image_indices = [
        i for i, m in enumerate(context_messages) if m.get("images")
    ]
    eligible: set[int] = set(image_indices[-max_images:]) if image_indices else set()

    if not eligible:
        return build_chat_log_xml(context_messages, conv_meta)

    parts: list[dict] = []
    text_buf: list[str] = [_conv_open_tag(meta)]
    _st = _self_tag(meta)
    if _st:
        text_buf.append(_st)
    text_buf.append("<chat_logs>")

    for i, msg in enumerate(context_messages):
        if conv_type == "group":
            text_buf.extend(_render_message_group(msg, context_messages))
        elif conv_type == "private":
            text_buf.extend(_render_message_private(msg, meta, context_messages))
        else:
            text_buf.extend(_render_message_generic(msg))

        if i in eligible:
            parts.append({"type": "text", "text": "\n".join(text_buf)})
            text_buf = []
            for img in msg["images"]:
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img['mime']};base64,{img['base64']}"
                        },
                    }
                )

    text_buf.append("</chat_logs>")
    text_buf.append("</conversation>")
    parts.append({"type": "text", "text": "\n".join(text_buf)})
    return parts


def format_chat_log_for_display(
    context_messages: list[dict],
    conv_meta: dict | None = None,
) -> str:
    """将上下文消息格式化为可读 XML，用于前端/日志展示。

    与 build_chat_log_xml 结构一致，但图片字段不嵌入 base64，只显示数量提示。
    """
    meta = conv_meta or _EMPTY_META
    conv_type = meta.get("type", "")

    if not context_messages:
        tag = _conv_open_tag(meta)
        self_line = _self_tag(meta)
        header = f"\n{self_line}" if self_line else ""
        return f"{tag}{header}\n<chat_logs>\n</chat_logs>\n</conversation>"

    lines: list[str] = [_conv_open_tag(meta)]
    self_line = _self_tag(meta)
    if self_line:
        lines.append(self_line)
    lines.append("<chat_logs>")

    for msg in context_messages:
        if conv_type == "group":
            msg_lines = _render_message_group(msg, context_messages)
        elif conv_type == "private":
            msg_lines = _render_message_private(msg, meta, context_messages)
        else:
            msg_lines = _render_message_generic(msg)

        # 在 </message> 关闭前插入图片数量提示
        if msg.get("images"):
            hint = f"    <!-- {len(msg['images'])}张图片（base64已省略） -->"
            msg_lines.insert(-1, hint)

        lines.extend(msg_lines)

    lines.append("</chat_logs>")
    lines.append("</conversation>")
    return "\n".join(lines)
