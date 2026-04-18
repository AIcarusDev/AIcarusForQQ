
"""unread_builder.py — 未读消息列表 XML 构建

仅负责生成 <unread_info> 块，不承担主模型 user prompt 的总装。
"""

import html
import re

from .xml_builder import _format_relative_time, _render_content_segments




def _render_preview_text(msg: dict, max_len: int = 30) -> str:
    """从消息条目中提取用于预览的纯文本（截断），供 build_unread_info_xml 使用。"""
    content_type = msg.get("content_type", "text")
    if content_type == "image":
        return "[图片]"
    if content_type == "sticker":
        return "[动画表情]"
    if content_type == "file":
        return "[文件]"
    if content_type == "recall":
        return "[撤回了一条消息]"
    segments = msg.get("content_segments")
    if segments:
        text = _render_content_segments(segments)
    else:
        text = html.escape(msg.get("content", ""), quote=False)
    # 去掉内联 XML 标签（如 <mention>）只保留纯文本
    text = re.sub(r"<[^>]+>", "", text)
    if len(text) > max_len:
        text = text[:max_len] + "..."
    return text


def build_unread_info_xml(sessions_dict: dict, current_key: str) -> str:
    """生成 <unread_info> 块，列出除当前会话外所有有未读的会话预览。

    sessions_dict  — 全局 sessions 字典（key → ChatSession）
    current_key    — 当前 bot 正在处理的会话 key（格式 "type_id"），排除在外

    无未读时返回空字符串。
    """
    lines: list[str] = []
    for key, s in sessions_dict.items():
        if key == current_key:
            continue
        if s.unread_count <= 0:
            continue

        # 取最后一条真实用户消息（跳过 bot 和 note）作为 preview
        last_msg = None
        for m in reversed(s.context_messages):
            if m.get("role") not in ("bot", "note"):
                last_msg = m
                break
        if last_msg is None:
            continue

        unread_display = "99+" if s.unread_count > 99 else str(s.unread_count)
        rel_time = _format_relative_time(last_msg.get("timestamp", ""))
        content_type_attr = html.escape(last_msg.get("content_type", "text"))
        preview_text = _render_preview_text(last_msg)

        if s.conv_type == "group":
            conv_name = html.escape(s.conv_name or s.conv_id)
            conv_id_e = html.escape(str(s.conv_id))
            sender = html.escape(last_msg.get("sender_name", ""))
            lines.append(f'  <session type="group" id="{conv_id_e}" name="{conv_name}" unread="{unread_display}">')
            lines.append(f'    <preview timestamp="{rel_time}" type="{content_type_attr}" sender="{sender}">{preview_text}</preview>')
            lines.append("  </session>")
        elif s.conv_type == "private":
            nickname = html.escape(s.conv_name or s.conv_id)
            conv_id_e = html.escape(str(s.conv_id))
            lines.append(f'  <session type="private" id="{conv_id_e}" nickname="{nickname}" unread="{unread_display}">')
            lines.append(f'    <preview timestamp="{rel_time}" type="{content_type_attr}">{preview_text}</preview>')
            lines.append("  </session>")

    if not lines:
        return ""
    return "<unread_info>\n" + "\n".join(lines) + "\n</unread_info>"
