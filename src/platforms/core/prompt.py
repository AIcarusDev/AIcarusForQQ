"""Prompt rendering for the local Core private dialogue."""

from __future__ import annotations

import html
from typing import Any

from platforms.chat.history_window import has_previous_messages, load_history_window


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


def _dialogue_window(session: Any) -> tuple[str, bool, list[dict[str, Any]]]:
    browsing = bool(getattr(session, "is_browsing_history", lambda: False)())
    if not browsing:
        return (
            "current",
            has_previous_messages(session, browsing=False),
            list(getattr(session, "context_messages", []) or []),
        )

    view = getattr(session, "chat_window_view", {}) or {}
    top_db_id = view.get("top_db_id")
    page_size = int(view.get("page_size", 10) or 10)
    if not top_db_id:
        return (
            "current",
            has_previous_messages(session, browsing=False),
            list(getattr(session, "context_messages", []) or []),
        )

    messages = load_history_window(session, int(top_db_id), page_size)
    if not messages:
        return (
            "current",
            has_previous_messages(session, browsing=False),
            list(getattr(session, "context_messages", []) or []),
        )
    return (
        "history",
        has_previous_messages(session, browsing=True, top_db_id=int(top_db_id)),
        messages,
    )


def render_dialogue(session: Any) -> str:
    """Render the fixed Core 1v1 session as compact model-visible XML."""

    mode, has_previous, messages = _dialogue_window(session)
    lines = [
        "<des></des>",
        f'<dialogue mode="{mode}" has_previous="{str(has_previous).lower()}">',
    ]
    for message in messages:
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
