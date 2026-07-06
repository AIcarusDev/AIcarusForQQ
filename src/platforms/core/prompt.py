"""Prompt rendering for the local Core private dialogue."""

from __future__ import annotations

import html
from typing import Any


def _attr(name: str, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return f' {name}="{html.escape(text, quote=True)}"'


def _segment_text(seg: dict[str, Any]) -> str:
    kind = str(seg.get("type") or seg.get("command") or "").strip()
    if kind == "text":
        return str(seg.get("text") or seg.get("content") or "")
    if kind == "image":
        image_ref = str(seg.get("image_ref") or seg.get("ref") or "").strip()
        return f'[image image_ref="{image_ref}"]' if image_ref else "[image]"
    if kind == "sticker":
        sticker_id = str(seg.get("sticker_id") or "").strip()
        return f'[sticker sticker_id="{sticker_id}"]' if sticker_id else "[sticker]"
    return str(seg.get("label") or seg.get("text") or "")


def _message_text(message: dict[str, Any]) -> str:
    segments = message.get("content_segments")
    if isinstance(segments, list) and segments:
        return "".join(_segment_text(seg) for seg in segments if isinstance(seg, dict))
    return str(message.get("content") or "")


def render_dialogue(session: Any) -> str:
    """Render the fixed Core 1v1 session as compact model-visible XML."""

    mode = "history" if getattr(session, "is_browsing_history", lambda: False)() else "current"
    lines = [
        "<des></des>",
        f'<dialogue mode="{mode}" has_previous="false">',
    ]
    for message in getattr(session, "context_messages", []) or []:
        if not isinstance(message, dict):
            continue
        tag = "self" if message.get("role") == "bot" else "guardian"
        attrs = (
            _attr("id", message.get("message_id"))
            + _attr("time", message.get("timestamp"))
            + _attr("reply_to", message.get("reply_to"))
        )
        text = html.escape(_message_text(message), quote=False)
        lines.append(f"  <{tag}{attrs}>{text}</{tag}>")
    lines.append("</dialogue>")
    return "\n".join(lines)
