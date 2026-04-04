"""chat_log_builder.py — IS 哨兵聊天记录构建

将 context_messages 切片渲染成带有 <sent_messages>、<trigger_message>、
<plan_messages> 标记的 XML 文本，供哨兵模型快速读取上下文。
"""

import html
from datetime import datetime, timezone

from llm.prompt.xml_builder import _render_content_chunks


_WINDOW_SIZE = 10  # 从 trigger_message 向前取的消息数（含 trigger）


def _fmt_ts(iso_ts: str) -> str:
    """ISO 时间戳 → 相对时间短字符串。"""
    try:
        past = datetime.fromisoformat(iso_ts)
        now = datetime.now(timezone.utc)
        if past.tzinfo is None:
            past = past.replace(tzinfo=timezone.utc)
        delta = (now - past).total_seconds()
    except (ValueError, TypeError):
        return iso_ts
    if delta < 10:
        return "刚刚"
    if delta < 60:
        return f"{int(delta)}秒前"
    mins = delta / 60
    if mins < 60:
        return f"{int(mins)}分钟前"
    hrs = mins / 60
    if hrs < 24:
        return f"{int(hrs)}小时前"
    return f"{int(hrs / 24)}天前"


def _render_msg(msg: dict) -> list[str]:
    """将单条 context_messages 条目渲染为 XML 行列表（不含换行符）。

    优先使用 content_segments（展开合并转发预览、嵌入图片哨兵标记），
    无 segments 时回退到 content 纯文本兜底。
    """
    msg_id = html.escape(str(msg.get("message_id", "")))
    ts = _fmt_ts(msg.get("timestamp", ""))
    sender_id = html.escape(str(msg.get("sender_id", "")))
    sender_name = html.escape(msg.get("sender_name", ""))
    lines = [
        f'<message id="{msg_id}" timestamp="{ts}">',
        f'  <sender id="{sender_id}" nickname="{sender_name}"/>',
    ]
    segments = msg.get("content_segments")
    if segments:
        chunks = _render_content_chunks(segments)
    else:
        ct = html.escape(msg.get("content_type", "text"))
        inner = html.escape(msg.get("content", ""), quote=False)
        chunks = [(ct, inner)]
    for ct, inner in chunks:
        lines.append(f'  <content type="{ct}">{inner}</content>')
    lines.append("</message>")
    return lines


def _plan_msg_to_text(msg: dict) -> str:
    """从 LLM plan message（send_messages 条目）提取纯文本。"""
    texts: list[str] = []
    for seg in msg.get("segments", []):
        cmd = seg.get("command", "")
        params = seg.get("params", {})
        if cmd == "text":
            texts.append(params.get("content", ""))
        elif cmd == "sticker":
            texts.append("[动画表情]")
        elif cmd == "at":
            texts.append(f"@{params.get('user_id', '')}")
    return "".join(texts)


def _render_plan_msg(order: int, msg: dict) -> list[str]:
    """将 LLM plan message 渲染为 <message order="N"> XML 行列表。"""
    sticker_segs = [s for s in msg.get("segments", []) if s.get("command") == "sticker"]
    text_segs = [s for s in msg.get("segments", []) if s.get("command") == "text"]
    if sticker_segs and not text_segs:
        content_type = "sticker"
        inner = html.escape("[动画表情]")
    else:
        content_type = "text"
        inner = html.escape(_plan_msg_to_text(msg))
    return [
        f'<message order="{order}">',
        f'  <content type="{content_type}">{inner}</content>',
        "</message>",
    ]


def build_sentinel_chat_log(
    context_messages: list[dict],
    trigger_id: str,
    sent_this_round_ids: set[str],
    remaining_plan_msgs: list[dict],
    conv_meta: dict,
) -> tuple[str, dict]:
    """构建哨兵聊天记录 XML 及图片数据。

    参数：
        context_messages:    会话上下文消息列表（当前完整 context）。
        trigger_id:          触发 IS 的消息的 message_id（已在 context 中）。
        sent_this_round_ids: 本轮已发送的 bot 消息 ID 集合，用于加 <sent_messages> 标记。
        remaining_plan_msgs: 还未发送的计划消息（send_messages 后半段）。
        conv_meta:           会话元信息 dict（type/id/name），用于 <conversation> 标签。

    返回：
        (xml_str, images_dict):
          xml_str      — 带图片哨兵标记的 XML 字符串。
          images_dict  — {ref: img_info} 仅含窗口内真实图片（排除动画表情）的数据，
                         供调用方按 IS 模型是否支持视觉选择不同的注入方式。
    """
    # 找到 trigger 消息的位置
    trigger_idx = -1
    for i, m in enumerate(context_messages):
        if str(m.get("message_id", "")) == trigger_id:
            trigger_idx = i
            break
    if trigger_idx < 0:
        # 找不到时取最后一条
        trigger_idx = len(context_messages) - 1

    # 取窗口切片
    start = max(0, trigger_idx - _WINDOW_SIZE + 1)
    window = context_messages[start: trigger_idx + 1]

    # 收集窗口内真实图片数据（排除动画表情 ref），供多模态 / vision bridge 描述注入
    collected_images: dict = {}
    for _msg in window:
        _img_data = _msg.get("images")
        if not _img_data:
            continue
        _segs = _msg.get("content_segments") or []
        _image_refs = {s["ref"] for s in _segs if s.get("type") == "image" and "ref" in s}
        for _ref, _info in _img_data.items():
            if _ref in _image_refs:
                collected_images[_ref] = _info

    # 构建 <conversation> 头部
    conv_type = conv_meta.get("type", "")
    conv_id = html.escape(str(conv_meta.get("id", "")))
    conv_name = html.escape(conv_meta.get("name", ""))
    bot_id = html.escape(str(conv_meta.get("bot_id", "")))
    bot_name = html.escape(conv_meta.get("bot_name", ""))

    if conv_type == "group":
        open_tag = f'<conversation type="group" id="{conv_id}" name="{conv_name}">'
    elif conv_type == "private":
        open_tag = f'<conversation type="private" id="{conv_id}" name="{conv_name}">'
    else:
        open_tag = '<conversation>'

    lines: list[str] = [
        open_tag,
        f'<self id="{bot_id}" name="{bot_name}"/>',
        "<chat_logs>",
    ]

    # 逐条渲染，对本轮已发送的 bot 消息加 <sent_messages> 包裹
    in_sent_block = False
    for msg in window:
        msg_id = str(msg.get("message_id", ""))
        is_sent_this_round = (msg.get("role") == "bot" and msg_id in sent_this_round_ids)
        is_trigger = (msg_id == trigger_id)

        if is_sent_this_round:
            if not in_sent_block:
                lines.append("<sent_messages>")
                in_sent_block = True
            lines.extend(_render_msg(msg))
        else:
            if in_sent_block:
                lines.append("</sent_messages>")
                in_sent_block = False
            if is_trigger:
                lines.append("<trigger_message>")
                lines.extend(_render_msg(msg))
                lines.append("</trigger_message>")
            else:
                lines.extend(_render_msg(msg))

    if in_sent_block:
        lines.append("</sent_messages>")

    lines.extend(["</chat_logs>", "</conversation>"])

    # 追加 <plan_messages>
    if remaining_plan_msgs:
        lines.append("<plan_messages>")
        for i, pm in enumerate(remaining_plan_msgs, start=1):
            lines.extend(_render_plan_msg(i, pm))
        lines.append("</plan_messages>")

    return "\n".join(lines), collected_images
