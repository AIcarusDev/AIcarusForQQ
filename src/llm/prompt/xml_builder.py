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
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone

from llm.media.outbound_image import make_data_url


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
_CARD_RAW_RENDER_LIMIT = 2000


def _normalize_at_display(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text if text.startswith("@") else f"@{text}"


def _collect_mention_uids(messages: list[dict]) -> set[str]:
    uids: set[str] = set()
    for msg in messages:
        for seg in msg.get("content_segments") or []:
            if seg.get("type") != "mention":
                continue
            uid = str(seg.get("uid", "") or "").strip()
            if uid and uid not in {"all", "self"}:
                uids.add(uid)
    return uids


def _collect_group_sender_uids(messages: list[dict]) -> set[str]:
    return {
        str(msg.get("sender_id", "") or "").strip()
        for msg in messages
        if msg.get("role") != "bot" and str(msg.get("sender_id", "") or "").strip()
    }


def _load_group_display_names(group_id: str, uids: set[str]) -> dict[str, str]:
    if not group_id or not uids:
        return {}

    try:
        from database import DB_PATH
    except Exception:
        return {}

    placeholders = ",".join("?" for _ in uids)
    group_uid = f"grp_qq_{group_id}"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                f"""SELECT a.platform_id, m.cardname, a.nickname
                    FROM entities a
                    LEFT JOIN memberships m
                      ON m.account_uid=a.account_uid AND m.group_uid=?
                    WHERE a.platform='qq' AND a.platform_id IN ({placeholders})""",
                [group_uid, *sorted(uids)],
            ).fetchall()
    except Exception:
        return {}

    result: dict[str, str] = {}
    for platform_id, cardname, nickname in rows:
        display = str(cardname or nickname or platform_id or "").strip()
        if display:
            result[str(platform_id)] = display
    return result


def _load_group_member_facts(group_id: str, uids: set[str]) -> dict[str, dict]:
    if not group_id or not uids:
        return {}

    try:
        from database import DB_PATH
    except Exception:
        return {}

    placeholders = ",".join("?" for _ in uids)
    group_uid = f"grp_qq_{group_id}"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                f"""SELECT a.platform_id, m.title, m.level
                    FROM entities a
                    LEFT JOIN memberships m
                      ON m.account_uid=a.account_uid AND m.group_uid=?
                    WHERE a.platform='qq' AND a.platform_id IN ({placeholders})""",
                [group_uid, *sorted(uids)],
            ).fetchall()
    except Exception:
        return {}

    result: dict[str, dict] = {}
    for platform_id, title, level in rows:
        result[str(platform_id)] = {
            "title": str(title or ""),
            "level": str(level or ""),
        }
    return result


def _hydrate_dynamic_group_display_names(
    context_messages: list[dict],
    conv_meta: dict,
) -> list[dict]:
    """Refresh mention display labels from current group identity state.

    Persisted chat rows intentionally keep the original message payload. Mention
    display names are presentation data, so they are resolved at render time from
    the current membership table instead of rewriting old message rows.
    """
    if conv_meta.get("type") != "group" or not context_messages:
        return context_messages

    group_id = str(conv_meta.get("id", "") or "").strip()
    mention_names = _load_group_display_names(group_id, _collect_mention_uids(context_messages))
    member_facts = _load_group_member_facts(group_id, _collect_group_sender_uids(context_messages))
    self_display = (
        str(conv_meta.get("bot_card", "") or "")
        or str(conv_meta.get("bot_name", "") or "")
        or str(conv_meta.get("bot_id", "") or "")
    )
    self_id = str(conv_meta.get("bot_id", "") or "").strip()

    if not mention_names and not self_display and not member_facts:
        return context_messages

    hydrated: list[dict] = []
    changed_any = False
    for msg in context_messages:
        segments = msg.get("content_segments") or []
        if not segments:
            hydrated.append(msg)
            continue

        new_segments: list[dict] | None = None
        for idx, seg in enumerate(segments):
            if seg.get("type") != "mention":
                continue
            uid = str(seg.get("uid", "") or "").strip()
            if uid == "self" or (self_id and uid == self_id):
                resolved = self_display
            elif uid in {"", "all"}:
                resolved = ""
            else:
                resolved = mention_names.get(uid, "")
            if not resolved:
                continue

            display = _normalize_at_display(resolved)
            if display == seg.get("display"):
                continue
            if new_segments is None:
                new_segments = deepcopy(segments)
            new_segments[idx]["display"] = display

        if new_segments is None:
            hydrated.append(msg)
        else:
            new_msg = dict(msg)
            new_msg["content_segments"] = new_segments
            hydrated.append(new_msg)
            changed_any = True

    if member_facts:
        enriched: list[dict] = []
        for msg in hydrated:
            if msg.get("role") == "bot":
                enriched.append(msg)
                continue
            facts = member_facts.get(str(msg.get("sender_id", "") or "").strip())
            if not facts:
                enriched.append(msg)
                continue
            updates = {}
            if facts.get("title") and not msg.get("sender_title"):
                updates["sender_title"] = facts["title"]
            if facts.get("level") and not msg.get("sender_level"):
                updates["sender_level"] = facts["level"]
            if updates:
                new_msg = dict(msg)
                new_msg.update(updates)
                enriched.append(new_msg)
                changed_any = True
            else:
                enriched.append(msg)
        hydrated = enriched

    return hydrated if changed_any else context_messages

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

    def _voice_label(seg: dict) -> str:
        try:
            duration = float(seg.get("duration", 0) or 0)
        except (TypeError, ValueError):
            duration = 0.0
        seconds = max(0, int(duration + 0.5))
        if seconds <= 0:
            return "[语音]"
        minutes, remain = divmod(seconds, 60)
        if minutes:
            return f"[语音 {minutes}'{remain:02d}'']"
        return f"[语音 {remain}'']"

    def _truncate_text(value: str, limit: int = _CARD_RAW_RENDER_LIMIT) -> tuple[str, bool]:
        if len(value) <= limit:
            return value, False
        return value[:limit], True

    def _tag(name: str, value: object, *, quote: bool = False) -> str:
        text = str(value or "")
        if not text:
            return ""
        return f"<{name}>{html.escape(text, quote=quote)}</{name}>"

    def _render_card(seg: dict) -> str:
        sub: list[str] = []
        for field in ("title", "summary", "app", "platform", "url"):
            rendered = _tag(field, seg.get(field))
            if rendered:
                sub.append(rendered)

        if seg.get("kind") == "contact":
            attrs: list[str] = []
            if contact_type := str(seg.get("contact_type", "") or ""):
                attrs.append(f'type="{html.escape(contact_type)}"')
            if contact_id := str(seg.get("contact_id", "") or ""):
                attrs.append(f'id="{html.escape(contact_id)}"')
            if attrs:
                sub.append(f'<contact {" ".join(attrs)}/>')

        if seg.get("kind") == "location":
            lat = str(seg.get("lat", "") or "")
            lon = str(seg.get("lon", "") or "")
            if lat or lon:
                sub.append(f'<geo lat="{html.escape(lat)}" lon="{html.escape(lon)}"/>')

        if markdown := str(seg.get("markdown", "") or ""):
            rendered, truncated = _truncate_text(markdown)
            attr = ' truncated="true"' if truncated else ""
            sub.append(f"<markdown{attr}>{html.escape(rendered, quote=False)}</markdown>")

        if raw := str(seg.get("raw", "") or ""):
            rendered, truncated = _truncate_text(raw)
            attr = ' truncated="true"' if truncated else ""
            sub.append(f"<raw{attr}>{html.escape(rendered, quote=False)}</raw>")

        if not sub:
            label = seg.get("label") or f"{seg.get('kind', 'unknown')} card"
            sub.append(_tag("summary", label))
        return "".join(sub)

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
        elif seg_type == "voice":
            _flush_text()
            chunks.append(("voice", _voice_label(seg)))
        elif seg_type == "forward":
            _flush_text()
            title = html.escape(seg.get("title", "合并转发"))
            preview_items = seg.get("preview") or []
            total = seg.get("total", 0)
            sub: list[str] = [f"<title>{title}</title><preview>"]
            if error := seg.get("error"):
                sub.append(f'<error>{html.escape(str(error))}</error>')
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
        elif seg_type == "card":
            _flush_text()
            kind = html.escape(str(seg.get("kind", "unknown") or "unknown"))
            chunks.append((f"card:{kind}", _render_card(seg)))
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
    rendered: list[str] = []
    for ct, inner in chunks:
        if ct == "forward":
            rendered.append(f'<content type="{ct}" openable="true">{inner}</content>')
        elif ct.startswith("card:"):
            kind = html.escape(ct.split(":", 1)[1] or "unknown")
            rendered.append(f'<content type="card" kind="{kind}">{inner}</content>')
        else:
            rendered.append(f'<content type="{ct}">{inner}</content>')
    return "    " + "".join(rendered)


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
        if img.get("expired"):
            return f'[{html.escape(label)}（图片已过期） ref="{ref}"]'
        if img.get("failed"):
            return f'[{html.escape(label)}（加载失败） ref="{ref}"]'
        if img.get("pending"):
            return f'[{html.escape(label)}（加载中） ref="{ref}"]'
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

    本函数对 before / tail 文本会再做一次 _resolve_sentinels 清扫，
    确保 caller 传入的 images dict 不完整时，残留哨兵不会原样泄漏到
    下游请求体（\x00 进 OpenAI JSON 会触发部分服务端解析器崩溃）。
    """
    parts: list[dict] = []
    last_end = 0
    for m in _IMG_SENTINEL_RE.finditer(text):
        ref = m.group(1)
        label = m.group(2)
        img = images.get(ref)
        before = _resolve_sentinels(text[last_end:m.start()], images)
        data_url = None
        if img and not img.get("failed") and not img.get("pending") and img.get("base64"):
            data_url = img.get("_llm_data_url")
            if data_url is None and not img.get("_llm_image_failed"):
                data_url = make_data_url(
                    str(img.get("base64") or ""),
                    str(img.get("mime") or "image/jpeg"),
                )
                if data_url:
                    img["_llm_data_url"] = data_url
                else:
                    img["_llm_image_failed"] = True
        if data_url and img:
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
                    "image_url": {"url": data_url},
                },
                {"type": "text", "text": f"]{desc_block}"},
            ])
        else:
            if img and img.get("expired"):
                hint = "（图片已过期）"
            elif img and img.get("failed"):
                hint = "（加载失败）"
            elif img and img.get("pending"):
                hint = "（加载中）"
            elif img and img.get("_llm_image_failed"):
                hint = "（格式无法发送）"
            else:
                hint = ""
            parts.append({"type": "text", "text": f'{before}[{label}{hint} ref="{ref}"]'})
        last_end = m.end()
    if tail := text[last_end:]:
        parts.append({"type": "text", "text": _resolve_sentinels(tail, images)})
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

    # 3. 彻底找不到（DB 和 QQ adapter 都没有）
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
    elif conv_type == "temp":
        attrs = f'type="temp" id="{conv_id}" user_id="{conv_id}"'
        if conv_name:
            attrs += f' nickname="{conv_name}"'
        source_group_id = html.escape(str(conv_meta.get("temp_source_group_id", "") or ""))
        source_group_name = html.escape(str(conv_meta.get("temp_source_group_name", "") or ""))
        if source_group_id:
            attrs += f' source_group_id="{source_group_id}"'
        if source_group_name:
            attrs += f' source_group_name="{source_group_name}"'
        return f"<conversation {attrs}>"
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
    """私聊/临时会话时生成 <other> 标签，声明对方身份，与 <self> 对称。"""
    if meta.get("type") not in {"private", "temp"}:
        return None
    other_id = html.escape(str(meta.get("id", "")))
    other_name = html.escape(meta.get("name", ""))
    if not other_id:
        return None
    attrs = f'id="{other_id}"'
    if other_name:
        attrs += f' name="{other_name}"'
    return f'<other {attrs}/>'


def _chat_logs_open_tag(mode: str, has_previous: bool) -> str:
    """生成 <chat_logs> 开标签，统一承载窗口状态。"""
    safe_mode = html.escape(mode or "current")
    previous_attr = "true" if has_previous else "false"
    return f'<chat_logs mode="{safe_mode}" has_previous="{previous_attr}">'


def _bubble_line(bubble_text: str) -> str | None:
    """可选的窗口气泡提示。"""
    if not bubble_text:
        return None
    return f'  <bubble>{html.escape(bubble_text, quote=False)}</bubble>'


# ── 单条消息渲染 ─────────────────────────────────────────

def _render_note(msg: dict) -> list[str]:
    """渲染系统通知条目（如撤回提示），输出 <note> 标签。"""
    rel_time = _format_relative_time(msg["timestamp"])
    content_type = html.escape(msg.get("content_type", "note"))
    segments = msg.get("content_segments") or []
    if segments and segments[0].get("type") == "recall_notice":
        seg = segments[0]
        lines = [f'  <note timestamp="{rel_time}">']
        actor = seg.get("operator") or {}
        actor_id = html.escape(str(actor.get("id", "")))
        if actor_id:
            actor_attrs = f'id="{actor_id}"'
            card = html.escape(str(actor.get("card", "")))
            if card:
                actor_attrs += f' card="{card}"'
            nickname = html.escape(str(actor.get("nickname", "")))
            if nickname:
                actor_attrs += f' nickname="{nickname}"'
            lines.append(f"    <operator {actor_attrs}/>")
        inner = html.escape(msg.get("content", ""), quote=False)
        lines.extend([
            f'    <content type="recall">{inner}</content>',
            "  </note>",
        ])
        return lines
    if segments and segments[0].get("type") == "group_notice":
        seg = segments[0]
        notice_type = html.escape(str(seg.get("notice_type") or msg.get("content_type") or "notice"))
        sub_type = html.escape(str(seg.get("sub_type") or ""))
        attrs = f'type="{notice_type}"'
        if sub_type:
            attrs += f' sub_type="{sub_type}"'
        lines = [f'  <note {attrs} timestamp="{rel_time}">']
        for tag_name in ("operator", "target"):
            actor = seg.get(tag_name) or {}
            actor_id = html.escape(str(actor.get("id", "")))
            if not actor_id:
                continue
            actor_attrs = f'id="{actor_id}"'
            card = html.escape(str(actor.get("card", "")))
            if card:
                actor_attrs += f' card="{card}"'
            nickname = html.escape(str(actor.get("nickname", "")))
            if nickname:
                actor_attrs += f' nickname="{nickname}"'
            lines.append(f"    <{tag_name} {actor_attrs}/>")
        if seg.get("duration_seconds"):
            seconds = html.escape(str(seg.get("duration_seconds", "")))
            duration_text = html.escape(str(seg.get("duration_text", "")))
            lines.append(f'    <duration seconds="{seconds}">{duration_text}</duration>')
        inner = html.escape(msg.get("content", ""), quote=False)
        lines.extend([
            f"    <content>{inner}</content>",
            "  </note>",
        ])
        return lines

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
        sender_attrs += _sender_group_state_attrs(msg)
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


def _sender_group_state_attrs(msg: dict) -> str:
    attrs: list[str] = []
    title = html.escape(str(msg.get("sender_title", "") or ""))
    level = html.escape(str(msg.get("sender_level", "") or ""))

    if title:
        attrs.append(f'title="{title}"')
    if level:
        attrs.append(f'level="{level}"')
    return (" " + " ".join(attrs)) if attrs else ""


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


# ── 合并转发浏览视图 ─────────────────────────────────────

def _find_forward_id(msg: dict) -> str:
    for seg in msg.get("content_segments") or []:
        if seg.get("type") == "forward" and seg.get("forward_id"):
            return str(seg.get("forward_id"))
    return ""


def _virtual_forward_message_id(root_message_id: str, path: list[int]) -> str:
    path_text = ".".join(str(p) for p in path) if path else "0"
    return f"fwd:{root_message_id}:{path_text}"


def _forward_path_xml(root_message_id: str, path: list[int]) -> list[str]:
    lines = ["  <path>"]
    lines.append(f'    <from_chat message_id="{html.escape(root_message_id)}"/>')
    for depth, node_index in enumerate(path, start=1):
        lines.append(f'    <from_node depth="{depth}" node_index="{node_index}"/>')
    lines.append("  </path>")
    return lines


def _render_forward_node_message(msg: dict) -> list[str]:
    rel_time = _format_relative_time(msg.get("timestamp", ""))
    msg_id = html.escape(str(msg.get("message_id", "")).strip())
    sender_id = html.escape(str(msg.get("sender_id", "")))
    sender_name = html.escape(str(msg.get("sender_name", "")))
    sender_role = html.escape(str(msg.get("sender_role", "")))
    sender_attrs = f'id="{sender_id}" nickname="{sender_name}"'
    if sender_role:
        sender_attrs += f' role="{sender_role}"'
    sender_attrs += _sender_group_state_attrs(msg)

    message_attrs = ['virtual="true"', f'timestamp="{html.escape(rel_time)}"']
    if msg_id:
        message_attrs.insert(0, f'id="{msg_id}"')

    return [
        f'  <message {" ".join(message_attrs)}>',
        f"    <sender {sender_attrs}/>",
        _render_content_xml(msg),
        "  </message>",
    ]


def _build_forward_browser_raw(session) -> tuple[str, list[dict]]:
    """渲染当前合并转发浏览浮层原始 XML，并刷新虚拟 id 注册表。"""
    stack = getattr(session, "forward_browser_stack", None) or []
    if not stack:
        if hasattr(session, "forward_virtual_registry"):
            session.forward_virtual_registry.clear()
        return "", []

    frame = stack[-1]
    root_message_id = str(frame.get("root_message_id", ""))
    current_path = list(frame.get("path") or [])
    nodes = list(frame.get("nodes") or [])
    page_size = int(frame.get("page_size") or 8)
    page_offset = max(0, int(frame.get("page_offset") or 0))
    visible_nodes = nodes[page_offset:page_offset + page_size]
    visible_nodes = _hydrate_dynamic_group_display_names(visible_nodes, session._get_conv_meta())
    has_previous = page_offset > 0
    has_next = page_offset + page_size < len(nodes)

    registry: dict[str, dict] = {}
    lines = ['<forward_browser active="true">']
    lines.extend(_forward_path_xml(root_message_id, current_path))

    title = html.escape(str(frame.get("title") or "合并转发"))
    total = int(frame.get("total") or len(nodes))
    depth = len(stack)
    lines.append(
        f'<forward_view depth="{depth}" title="{title}" total="{total}" '
        f'page_offset="{page_offset}" page_size="{page_size}" '
        f'has_previous="{str(has_previous).lower()}" has_next="{str(has_next).lower()}">'
    )

    for local_index, node in enumerate(visible_nodes, start=page_offset + 1):
        node_path = current_path + [local_index]
        rendered_node = dict(node)
        forward_id = _find_forward_id(rendered_node)
        if forward_id:
            virtual_id = _virtual_forward_message_id(root_message_id, node_path)
            rendered_node["message_id"] = virtual_id
            registry_entry = {
                "forward_id": forward_id,
                "root_message_id": root_message_id,
                "path": node_path,
                "title": "合并转发",
            }
            for seg in rendered_node.get("content_segments") or []:
                if seg.get("type") == "forward" and str(seg.get("forward_id", "")) == forward_id:
                    if isinstance(seg.get("content"), list):
                        registry_entry["content"] = seg["content"]
                    break
            registry[virtual_id] = registry_entry
        else:
            rendered_node.pop("message_id", None)

        lines.extend(_render_forward_node_message(rendered_node))

    lines.extend(["</forward_view>", "</forward_browser>"])
    session.forward_virtual_registry = registry
    return "\n".join(lines), visible_nodes


def _collect_images(nodes: list[dict]) -> dict[str, dict]:
    all_images: dict[str, dict] = {}
    for node in nodes:
        all_images.update(node.get("images") or {})
    return all_images


def build_forward_browser_xml(session) -> str:
    """渲染当前合并转发浏览浮层，并刷新虚拟 id 注册表。"""
    raw_xml, visible_nodes = _build_forward_browser_raw(session)
    if not raw_xml:
        return ""
    all_images = _collect_images(visible_nodes)
    return _resolve_sentinels(raw_xml, all_images)


def build_forward_browser_content(session) -> "str | list":
    """渲染合并转发浏览浮层；有图片时输出多模态 parts。"""
    raw_xml, visible_nodes = _build_forward_browser_raw(session)
    if not raw_xml:
        return ""
    all_images = _collect_images(visible_nodes)
    if not all_images:
        return _resolve_sentinels(raw_xml, all_images)
    parts = _inject_images_by_ref(raw_xml, all_images)
    return parts or _resolve_sentinels(raw_xml, all_images)


# ── 核心公共 API ─────────────────────────────────────────

_EMPTY_META: dict = {}


def build_chat_log_xml(
    context_messages: list[dict],
    conv_meta: dict | None = None,
    quoted_extra: dict | None = None,
    *,
    chat_logs_mode: str = "current",
    has_previous: bool = False,
    bubble_text: str = "",
) -> str:
    """将上下文消息列表转为结构化 XML 字符串（纯文本，不含图片 base64）。"""
    meta = conv_meta or _EMPTY_META
    context_messages = _hydrate_dynamic_group_display_names(context_messages, meta)
    conv_type = meta.get("type", "")
    chat_logs_tag = _chat_logs_open_tag(chat_logs_mode, has_previous)
    bubble_line = _bubble_line(bubble_text)

    if not context_messages:
        tag = _conv_open_tag(meta)
        header_parts = []
        if self_line := _self_tag(meta):
            header_parts.append(self_line)
        if other_line := _other_tag(meta):
            header_parts.append(other_line)
        lines = [tag]
        lines.extend(header_parts)
        lines.append(chat_logs_tag)
        if bubble_line:
            lines.append(bubble_line)
        lines.extend(["</chat_logs>", "</conversation>"])
        return "\n".join(lines)

    lines: list[str] = [_conv_open_tag(meta)]
    if self_line := _self_tag(meta):
        lines.append(self_line)
    if other_line := _other_tag(meta):
        lines.append(other_line)
    lines.append(chat_logs_tag)

    for msg in context_messages:
        if msg.get("role") == "note":
            lines.extend(_render_note(msg))
        elif conv_type == "group":
            lines.extend(_render_message_group(msg, context_messages, quoted_extra))
        elif conv_type in {"private", "temp"}:
            lines.extend(_render_message_private(msg, meta, context_messages, quoted_extra))
        else:
            lines.extend(_render_message_generic(msg))

    if bubble_line:
        lines.append(bubble_line)
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
    *,
    chat_logs_mode: str = "current",
    has_previous: bool = False,
    bubble_text: str = "",
) -> "str | list":
    """将上下文消息列表转为 LLM 可用的内容（纯 XML 或多模态 parts）。

    图片精准内嵌：每条携带图片的消息，其图片 content part 紧跟该消息的 XML 之后，
    模型注意力自然落在"消息文字 → 对应图片"的顺序上。

    max_images 是聊天记录渲染阶段的含图片消息 hint；-1 表示本阶段不限制。
    无图片时退回纯字符串，与原有逻辑完全兼容。
    """
    meta = conv_meta or _EMPTY_META
    context_messages = _hydrate_dynamic_group_display_names(context_messages, meta)
    conv_type = meta.get("type", "")
    chat_logs_tag = _chat_logs_open_tag(chat_logs_mode, has_previous)
    bubble_line = _bubble_line(bubble_text)

    if not context_messages:
        return build_chat_log_xml(
            context_messages,
            conv_meta,
            quoted_extra,
            chat_logs_mode=chat_logs_mode,
            has_previous=has_previous,
            bubble_text=bubble_text,
        )

    image_indices = [
        i for i, m in enumerate(context_messages) if m.get("images")
    ]
    if max_images < 0:
        eligible: set[int] = set(image_indices)
    elif max_images == 0:
        eligible = set()
    else:
        eligible = set(image_indices[-max_images:]) if image_indices else set()

    if not eligible:
        return build_chat_log_xml(
            context_messages,
            conv_meta,
            quoted_extra,
            chat_logs_mode=chat_logs_mode,
            has_previous=has_previous,
            bubble_text=bubble_text,
        )

    # 汇总所有消息的 images，供未及时填充 entry["images"] 的消息（典型场景：
    # URL 图片下载与 prompt 构造的竞态）兜底解析其残留哨兵的描述块。
    all_images: dict[str, dict] = {}
    for msg in context_messages:
        all_images.update(msg.get("images") or {})

    parts: list[dict] = []
    text_buf: list[str] = [_conv_open_tag(meta)]
    if _st := _self_tag(meta):
        text_buf.append(_st)
    if _ot := _other_tag(meta):
        text_buf.append(_ot)
    text_buf.append(chat_logs_tag)

    for i, msg in enumerate(context_messages):
        if msg.get("role") == "note":
            text_buf.extend(_render_note(msg))
        elif conv_type == "group":
            text_buf.extend(_render_message_group(msg, context_messages, quoted_extra))
        elif conv_type in {"private", "temp"}:
            text_buf.extend(_render_message_private(msg, meta, context_messages, quoted_extra))
        else:
            text_buf.extend(_render_message_generic(msg))

        if i in eligible and msg.get("images"):
            # 按哨兵精准嵌入图片，格式：[图片"<图片内容>"]
            full_text = "\n".join(text_buf)
            text_buf = []
            parts.extend(_inject_images_by_ref(full_text, msg["images"]))

    if bubble_line:
        text_buf.append(bubble_line)
    text_buf.append("</chat_logs>")
    text_buf.append("</conversation>")
    parts.append({"type": "text", "text": "\n".join(text_buf)})

    # 出口兜底：所有 text part 必须不含 \x00 哨兵，否则会污染下游 JSON 请求体。
    for idx, part in enumerate(parts):
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text", "")
            if "\x00" in text:
                parts[idx] = {**part, "text": _resolve_sentinels(text, all_images)}
    return parts


def format_chat_log_for_display(
    context_messages: list[dict],
    conv_meta: dict | None = None,
    quoted_extra: dict | None = None,
    *,
    chat_logs_mode: str = "current",
    has_previous: bool = False,
    bubble_text: str = "",
) -> str:
    """将上下文消息格式化为可读 XML，用于前端/日志展示。

    与 build_chat_log_xml 结构一致，但图片字段不嵌入 base64，只显示数量提示。
    """
    meta = conv_meta or _EMPTY_META
    context_messages = _hydrate_dynamic_group_display_names(context_messages, meta)
    conv_type = meta.get("type", "")
    chat_logs_tag = _chat_logs_open_tag(chat_logs_mode, has_previous)
    bubble_line = _bubble_line(bubble_text)

    if not context_messages:
        tag = _conv_open_tag(meta)
        header_parts = []
        if self_line := _self_tag(meta):
            header_parts.append(self_line)
        if other_line := _other_tag(meta):
            header_parts.append(other_line)
        lines = [tag]
        lines.extend(header_parts)
        lines.append(chat_logs_tag)
        if bubble_line:
            lines.append(bubble_line)
        lines.extend(["</chat_logs>", "</conversation>"])
        return "\n".join(lines)

    lines: list[str] = [_conv_open_tag(meta)]
    if self_line := _self_tag(meta):
        lines.append(self_line)
    if other_line := _other_tag(meta):
        lines.append(other_line)
    lines.append(chat_logs_tag)

    for msg in context_messages:
        if msg.get("role") == "note":
            lines.extend(_render_note(msg))
            continue
        if conv_type == "group":
            msg_lines = _render_message_group(msg, context_messages, quoted_extra)
        elif conv_type in {"private", "temp"}:
            msg_lines = _render_message_private(msg, meta, context_messages, quoted_extra)
        else:
            msg_lines = _render_message_generic(msg)

        # 在 </message> 关闭前插入图片数量提示
        if msg.get("images"):
            hint = f"    <!-- {len(msg['images'])}张图片（base64已省略） -->"
            msg_lines.insert(-1, hint)

        lines.extend(msg_lines)

    if bubble_line:
        lines.append(bubble_line)
    lines.extend(["</chat_logs>", "</conversation>"])
    # 收集所有消息中的 images，供 _resolve_sentinels 渲染描述块
    all_images: dict[str, dict] = {}
    for msg in context_messages:
        all_images.update(msg.get("images") or {})
    return _resolve_sentinels("\n".join(lines), all_images)
