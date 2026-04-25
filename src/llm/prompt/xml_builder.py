"""xml_builder.py — 聊天记录 → XML 转化

将内部上下文消息列表转为结构化 XML，供 LLM 上下文使用。

输出格式根据会话类型自动适配：
  - 群聊：完整 sender / role / at / quote
  - 私聊：精简，去掉每条消息的 sender 块，bot 消息用 from="self" 标记
  - Web：兜底的通用格式

日志/前端展示使用 format_chat_log_for_display() 获取可读版本。
"""

import html
import re
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
    if minutes < 10:
        m = int(minutes)
        s = int(delta) % 60
        return f"{m}分{s}秒前"
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
    return f"{int(months)}个月前" if months < 12 else f"{int(days / 365)}年前"

# 图片位置哨兵：格式 \x00{12位hex_ref}:{label}\x00，用户输入不含 \x00，天然防注入
_IMG_SENTINEL_RE = re.compile(r'\x00([a-f0-9]{12}):([^\x00]+)\x00')

# ── 内容段渲染 ────────────────────────────────────────────

def _render_content_chunks(segments: list[dict]) -> list[tuple[str, str]]:
    """将结构化 content_segments 渲染为 (content_type, inner_xml) 列表。

    text / at / emoji 视为内联文本，合并为同一个 "text" 块；
    image / sticker / file / forward 各自独立为单独块。
    这样调用方可以为每块生成独立的 <content type="..."> 标签，彻底消除歧义。
    """
    chunks: list[tuple[str, str]] = []
    text_buf: list[str] = []

    def _flush_text() -> None:
        if text_buf:
            chunks.append(("text", "".join(text_buf)))
            text_buf.clear()

    for seg in segments:
        seg_type = seg.get("type", "")
        if seg_type == "text":
            text_buf.append(html.escape(seg.get("text", ""), quote=False))
        elif seg_type == "mention":
            uid = html.escape(str(seg.get("uid", "")))
            display = html.escape(seg.get("display", ""))
            text_buf.append(f'<at uid="{uid}">{display}</at>')
        elif seg_type == "emoji":
            eid = html.escape(str(seg.get("id", "")))
            name = html.escape(seg.get("name", ""))
            text_buf.append(f'<emoji id="{eid}" name="{name}"/>')
        elif seg_type == "image":
            _flush_text()
            ref = seg.get("ref", "")
            chunks.append(("image", f"\x00{ref}:图片\x00" if ref else "[图片]"))
        elif seg_type == "sticker":
            _flush_text()
            ref = seg.get("ref", "")
            sticker_id = seg.get("sticker_id", "")
            if ref:
                chunks.append(("sticker", f"\x00{ref}:动画表情\x00"))
            elif sticker_id:
                chunks.append(("sticker", f'[动画表情 id="{html.escape(sticker_id)}"]'))
            else:
                chunks.append(("sticker", "[动画表情]"))
        elif seg_type == "file":
            _flush_text()
            fn = html.escape(seg.get("filename", "未知"))
            chunks.append(("file", f"[文件:{fn}]"))
        elif seg_type == "forward":
            _flush_text()
            title = html.escape(seg.get("title", "合并转发"))
            preview_items = seg.get("preview") or []
            total = seg.get("total", 0)
            sub: list[str] = [f"<title>{title}</title><preview>"]
            for item in preview_items:
                sender_e = html.escape(item.get("sender", ""))
                ct = html.escape(item.get("content_type", "text"))
                text = item.get("content_text", "")
                if text:
                    sub.append(
                        f'<message sender="{sender_e}">'
                        f'<content type="{ct}">{html.escape(text)}</content>'
                        f'</message>'
                    )
                else:
                    sub.append(f'<message sender="{sender_e}"><content type="{ct}"/></message>')
            sub.append(f'</preview><footer total="{total}"/>')
            chunks.append(("forward", "".join(sub)))
        else:
            _flush_text()
            label = seg.get("label", seg_type)
            chunks.append(("text", f"[{html.escape(label)}]"))

    _flush_text()
    return chunks if chunks else [("text", "")]


def _render_content_segments(segments: list[dict]) -> str:
    """将结构化 content_segments 渲染为平坦字符串（供 unread_builder 等纯文本场景使用）。"""
    return "".join(inner for _, inner in _render_content_chunks(segments))


def _render_content_text(content: str) -> str:
    """兜底：没有 content_segments 时用纯文本渲染。"""
    return html.escape(content, quote=False)


def _render_content_xml(msg: dict) -> str:
    """将消息内容渲染为一行完整的 <content> 标签字符串（含 4 空格缩进）。

    混合消息（如文字 + 图片 + 表情包）会生成多个紧邻的 <content> 标签，
    每个标签携带正确的 type 属性，彻底消除 type="text" 内混入媒体占位符的歧义。
    """
    segments = msg.get("content_segments")
    if segments:
        chunks = _render_content_chunks(segments)
    else:
        ct = html.escape(msg.get("content_type", "text"))
        inner = _render_content_text(msg.get("content", ""))
        chunks = [(ct, inner)]
    return "    " + "".join(f'<content type="{ct}">{inner}</content>' for ct, inner in chunks)


def _build_description_block(
    description: "str | None",
    examinations: list,
) -> str:
    """将描述和精查结果渲染为 <description> XML 块。无内容时返回空字符串。"""
    if not description and not examinations:
        return ""
    lines = ["\n<description>"]
    if description:
        lines.append(f"  <auto>{html.escape(description)}</auto>")
    for exam in examinations:
        focus_e = html.escape(exam.get("focus", ""))
        result_e = html.escape(exam.get("result", ""))
        lines.append(f'  <examine focus="{focus_e}">{result_e}</examine>')
    lines.append("</description>")
    return "\n".join(lines)


def _resolve_sentinels(
    text: str,
    images: "dict[str, dict] | None" = None,
) -> str:
    """将文本中的图片哨兵（\x00ref:label\x00）替换为可读标签 [label]。

    当 images 字典提供时，若对应图片有描述/精查结果，会在标签后追加
    <description> 块，供不支持视觉的模型理解图片内容。
    """
    if not images:
        return _IMG_SENTINEL_RE.sub(
            lambda m: f'[{html.escape(m.group(2))} ref="{m.group(1)}"]',
            text,
        )

    def _replace(m: re.Match) -> str:
        ref = m.group(1)
        label = m.group(2)
        img = images.get(ref)
        if not img:
            return f'[{html.escape(label)} ref="{ref}"]'
        if img.get("failed"):
            return f'[{html.escape(label)}（加载失败） ref="{ref}"]'
        label_tag = f'[{html.escape(label)} ref="{ref}"]'
        desc_block = _build_description_block(
            img.get("description"),
            img.get("examinations") or [],
        )
        return label_tag + desc_block

    return _IMG_SENTINEL_RE.sub(_replace, text)


def _inject_images_by_ref(text: str, images: dict[str, dict]) -> list[dict]:
    """按哨兵位置将图片 part 精准嵌入文本，生成多模态 parts 列表。

    哨兵格式：\x00{ref}:{label}\x00，由 _render_content_segments 写入。
    用户输入不含 \x00，完全消除注入风险。
    图片未成功下载（ref 不在 images dict 中）时退化为可读标签。
    """
    parts: list[dict] = []
    last_end = 0
    for m in _IMG_SENTINEL_RE.finditer(text):
        ref = m.group(1)
        label = m.group(2)
        img = images.get(ref)
        before = text[last_end:m.start()]
        if img and not img.get("failed"):
            # 描述块追加在闭合括号后：vision=false 时 _strip_images 移除 image_url
            # 但保留文本 parts，模型仍能读到描述
            desc_block = _build_description_block(
                img.get("description"),
                img.get("examinations") or [],
            )
            parts.extend([
                {"type": "text", "text": f'{before}[{label} ref="{ref}"'},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime']};base64,{img['base64']}"},
                },
                {"type": "text", "text": f"]{desc_block}"},
            ])
        else:
            fail_hint = "（加载失败）" if img and img.get("failed") else ""
            parts.append({"type": "text", "text": f'{before}[{label}{fail_hint} ref="{ref}"]'})
        last_end = m.end()
    if tail := text[last_end:]:
        parts.append({"type": "text", "text": tail})
    return parts


# ── 回复引用 ─────────────────────────────────────────────

def _build_quote_xml(
    ref_id: str,
    context_messages: list[dict],
    indent: str,
    quoted_extra: dict | None = None,
) -> str | None:
    """根据 ref_id 在上下文中查找被引用的消息，构建 <quote> 标签。

    查找顺序：
      1. 当前上下文窗口（context_messages）
      2. 预取缓存（quoted_extra，由 prefetch_quoted_messages 填充）
      3. 都找不到 → [ERROR: Message_lost]
    """
    # 1. 先在当前窗口里找
    for m in context_messages:
        if str(m.get("message_id", "")) == ref_id:
            # 被引用的消息已被撤回
            if m.get("content_type") == "recall":
                return (
                    f"{indent}<quote>\n"
                    f"{indent}  <preview><recall>[原消息已撤回]</recall></preview>\n"
                    f"{indent}</quote>"
                )
            name = html.escape(m.get("sender_name", ""))
            raw = m.get("content", "")
            preview_text = raw[:50] + ("..." if len(raw) > 50 else "")
            preview = html.escape(preview_text, quote=False)
            return (
                f'{indent}<quote ref_id="{html.escape(ref_id)}">\n'
                f"{indent}  <preview>{name}: {preview}</preview>\n"
                f"{indent}</quote>"
            )

    # 2. 在预取缓存里找（窗口外但已恢复的消息）
    if quoted_extra:
        m = quoted_extra.get(ref_id)
        if m:
            name = html.escape(m.get("sender_name", ""))
            raw = m.get("content", "")
            preview_text = raw[:50] + ("..." if len(raw) > 50 else "")
            preview = html.escape(preview_text, quote=False)
            return (
                f'{indent}<quote ref_id="{html.escape(ref_id)}">\n'
                f"{indent}  <preview>{name}: {preview}</preview>\n"
                f"{indent}</quote>"
            )

    # 3. 彻底找不到（DB 和 NapCat 都没有）
    return (
        f'{indent}<quote ref_id="{html.escape(ref_id)}">\n'
        f"{indent}  <preview>[ERROR: Message_lost]</preview>\n"
        f"{indent}</quote>"
    )


# ── conversation 开闭标签 ─────────────────────────────────

def _conv_open_tag(conv_meta: dict) -> str:
    """生成 <conversation ...> 开标签。"""
    conv_type = conv_meta.get("type", "")
    conv_id = html.escape(str(conv_meta.get("id", "")))
    conv_name = html.escape(conv_meta.get("name", ""))

    if conv_type == "group":
        member_count = conv_meta.get("member_count", 0)
        attrs = f'type="group" id="{conv_id}"'
        if conv_name:
            attrs += f' name="{conv_name}"'
        if member_count:
            attrs += f' member_count="{member_count}"'
        return f"<conversation {attrs}>"
    elif conv_type == "private":
        return '<conversation type="private">'
    else:
        return "<conversation>"


def _self_tag(meta: dict) -> str | None:
    """当有 bot_id 时生成 <self> 标签，供 LLM 交叉验证 from="self" 身份。"""
    bot_id = html.escape(str(meta.get("bot_id", "")))
    bot_name = html.escape(meta.get("bot_name", ""))
    bot_card = html.escape(meta.get("bot_card", ""))
    if not bot_id:
        return None
    attrs = f'id="{bot_id}"'
    if bot_name:
        attrs += f' name="{bot_name}"'
    if bot_card:
        attrs += f' card="{bot_card}"'
    return f'<self {attrs}/>'


def _other_tag(meta: dict) -> str | None:
    """私聊时生成 <other> 标签，声明对方身份，与 <self> 对称。"""
    if meta.get("type") != "private":
        return None
    other_id = html.escape(str(meta.get("id", "")))
    other_name = html.escape(meta.get("name", ""))
    if not other_id:
        return None
    attrs = f'id="{other_id}"'
    if other_name:
        attrs += f' name="{other_name}"'
    return f'<other {attrs}/>'


# ── 单条消息渲染 ─────────────────────────────────────────

def _render_note(msg: dict) -> list[str]:
    """渲染系统通知条目（如撤回提示），输出 <note> 标签。"""
    rel_time = _format_relative_time(msg["timestamp"])
    content_type = html.escape(msg.get("content_type", "note"))
    inner = html.escape(msg.get("content", ""), quote=False)
    return [
        f'  <note timestamp="{rel_time}">',
        f'    <content type="{content_type}">{inner}</content>',
        "  </note>",
    ]


def _render_message_group(
    msg: dict,
    context_messages: list[dict],
    quoted_extra: dict | None = None,
) -> list[str]:
    """群聊模式：完整 sender + role + quote + content type。"""
    rel_time = _format_relative_time(msg["timestamp"])
    msg_id = html.escape(str(msg["message_id"]))
    lines: list[str] = [f'  <message id="{msg_id}" timestamp="{rel_time}">']

    # <sender>（bot 自身消息用 id="self"，避免重复声明已在 <self> 中给出的 id）
    if msg.get("role") == "bot":
        lines.append('    <sender id="self"/>')
    else:
        sender_id = html.escape(str(msg.get("sender_id", "")))
        nickname = html.escape(msg.get("sender_name", ""))
        group_role = html.escape(msg.get("sender_role", ""))
        sender_attrs = f'id="{sender_id}" nickname="{nickname}"'
        if group_role:
            sender_attrs += f' role="{group_role}"'
        lines.append(f"    <sender {sender_attrs}/>")

    # <quote>（如果有引用）
    if reply_to := msg.get("reply_to"):
        if quote_xml := _build_quote_xml(reply_to, context_messages, "    ", quoted_extra):
            lines.append(quote_xml)

    # <content>
    lines.extend([
        _render_content_xml(msg),
        "  </message>",
    ])
    return lines


def _render_message_private(
    msg: dict,
    conv_meta: dict,
    context_messages: list[dict],
    quoted_extra: dict | None = None,
) -> list[str]:
    """私聊模式：精简，无 sender 块，bot 消息用 from="self"。"""
    rel_time = _format_relative_time(msg["timestamp"])
    msg_id = html.escape(str(msg["message_id"]))

    # bot 自己的消息加 from="self"，对方消息加 from="other"
    is_self = msg.get("role") == "bot"
    from_attr = ' from="self"' if is_self else ' from="other"'
    lines: list[str] = [f'  <message id="{msg_id}" timestamp="{rel_time}"{from_attr}>']

    # <quote>（私聊也可以回复）
    if reply_to := msg.get("reply_to"):
        if quote_xml := _build_quote_xml(reply_to, context_messages, "    ", quoted_extra):
            lines.append(quote_xml)

    lines.extend([
        _render_content_xml(msg),
        "  </message>",
    ])
    return lines


def _render_message_generic(msg: dict) -> list[str]:
    """Web / 通用模式：简单的 sender_name 属性 + content type。"""
    rel_time = _format_relative_time(msg["timestamp"])
    msg_id = html.escape(str(msg["message_id"]))
    safe_name = html.escape(msg.get("sender_name", ""))
    lines: list[str] = [
        f'  <message id="{msg_id}" sender_name="{safe_name}" timestamp="{rel_time}">',
        _render_content_xml(msg),
        "  </message>",
    ]
    return lines


# ── 核心公共 API ─────────────────────────────────────────

_EMPTY_META: dict = {}


def build_chat_log_xml(
    context_messages: list[dict],
    conv_meta: dict | None = None,
    quoted_extra: dict | None = None,
) -> str:
    """将上下文消息列表转为结构化 XML 字符串（纯文本，不含图片 base64）。"""
    meta = conv_meta or _EMPTY_META
    conv_type = meta.get("type", "")

    if not context_messages:
        tag = _conv_open_tag(meta)
        header_parts = []
        if self_line := _self_tag(meta):
            header_parts.append(self_line)
        if other_line := _other_tag(meta):
            header_parts.append(other_line)
        header = ("\n" + "\n".join(header_parts)) if header_parts else ""
        return f"{tag}{header}\n<chat_logs>\n</chat_logs>\n</conversation>"

    lines: list[str] = [_conv_open_tag(meta)]
    if self_line := _self_tag(meta):
        lines.append(self_line)
    if other_line := _other_tag(meta):
        lines.append(other_line)
    lines.append("<chat_logs>")

    for msg in context_messages:
        if msg.get("role") == "note":
            lines.extend(_render_note(msg))
        elif conv_type == "group":
            lines.extend(_render_message_group(msg, context_messages, quoted_extra))
        elif conv_type == "private":
            lines.extend(_render_message_private(msg, meta, context_messages, quoted_extra))
        else:
            lines.extend(_render_message_generic(msg))

    lines.extend(["</chat_logs>", "</conversation>"])
    # 收集所有消息中的 images，供 _resolve_sentinels 渲染描述块
    all_images: dict[str, dict] = {}
    for msg in context_messages:
        all_images.update(msg.get("images") or {})
    return _resolve_sentinels("\n".join(lines), all_images)


def build_multimodal_content(
    context_messages: list[dict],
    conv_meta: dict | None = None,
    max_images: int = 5,
    quoted_extra: dict | None = None,
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
        return build_chat_log_xml(context_messages, conv_meta, quoted_extra)

    image_indices = [
        i for i, m in enumerate(context_messages) if m.get("images")
    ]
    eligible: set[int] = set(image_indices[-max_images:]) if image_indices else set()

    if not eligible:
        return build_chat_log_xml(context_messages, conv_meta, quoted_extra)

    parts: list[dict] = []
    text_buf: list[str] = [_conv_open_tag(meta)]
    if _st := _self_tag(meta):
        text_buf.append(_st)
    if _ot := _other_tag(meta):
        text_buf.append(_ot)
    text_buf.append("<chat_logs>")

    for i, msg in enumerate(context_messages):
        if msg.get("role") == "note":
            text_buf.extend(_render_note(msg))
        elif conv_type == "group":
            text_buf.extend(_render_message_group(msg, context_messages, quoted_extra))
        elif conv_type == "private":
            text_buf.extend(_render_message_private(msg, meta, context_messages, quoted_extra))
        else:
            text_buf.extend(_render_message_generic(msg))

        if i in eligible and msg.get("images"):
            # 按哨兵精准嵌入图片，格式：[图片"<图片内容>"]
            full_text = "\n".join(text_buf)
            text_buf = []
            parts.extend(_inject_images_by_ref(full_text, msg["images"]))

    text_buf.append("</chat_logs>")
    text_buf.append("</conversation>")
    parts.append({"type": "text", "text": "\n".join(text_buf)})
    return parts


def format_chat_log_for_display(
    context_messages: list[dict],
    conv_meta: dict | None = None,
    quoted_extra: dict | None = None,
) -> str:
    """将上下文消息格式化为可读 XML，用于前端/日志展示。

    与 build_chat_log_xml 结构一致，但图片字段不嵌入 base64，只显示数量提示。
    """
    meta = conv_meta or _EMPTY_META
    conv_type = meta.get("type", "")

    if not context_messages:
        tag = _conv_open_tag(meta)
        header_parts = []
        if self_line := _self_tag(meta):
            header_parts.append(self_line)
        if other_line := _other_tag(meta):
            header_parts.append(other_line)
        header = ("\n" + "\n".join(header_parts)) if header_parts else ""
        return f"{tag}{header}\n<chat_logs>\n</chat_logs>\n</conversation>"

    lines: list[str] = [_conv_open_tag(meta)]
    if self_line := _self_tag(meta):
        lines.append(self_line)
    if other_line := _other_tag(meta):
        lines.append(other_line)
    lines.append("<chat_logs>")

    for msg in context_messages:
        if msg.get("role") == "note":
            lines.extend(_render_note(msg))
            continue
        if conv_type == "group":
            msg_lines = _render_message_group(msg, context_messages, quoted_extra)
        elif conv_type == "private":
            msg_lines = _render_message_private(msg, meta, context_messages, quoted_extra)
        else:
            msg_lines = _render_message_generic(msg)

        # 在 </message> 关闭前插入图片数量提示
        if msg.get("images"):
            hint = f"    <!-- {len(msg['images'])}张图片（base64已省略） -->"
            msg_lines.insert(-1, hint)

        lines.extend(msg_lines)

    lines.extend(["</chat_logs>", "</conversation>"])
    # 收集所有消息中的 images，供 _resolve_sentinels 渲染描述块
    all_images: dict[str, dict] = {}
    for msg in context_messages:
        all_images.update(msg.get("images") or {})
    return _resolve_sentinels("\n".join(lines), all_images)
