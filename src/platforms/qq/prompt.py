"""QQ platform prompt rendering."""

from __future__ import annotations

import html
from typing import Any

from .unread import build_unread_info_xml


def platform_open_tag(
    *,
    account_id: str = "",
    account_name: str = "",
) -> str:
    safe_account_id = html.escape(str(account_id or ""), quote=True)
    safe_account_name = html.escape(str(account_name or ""), quote=True)
    return (
        f'<platform name="qq" '
        f'account_id="{safe_account_id}" '
        f'account_name="{safe_account_name}">'
    )


def platform_description() -> str:
    return (
        "<des>"
        "These are the messages visible to the currently logged-in account on the QQ platform."
        "- If a message in the `unread_info` preview originates from a group chat, it represents a public message posted by another user."
        "- Each message has its own `type` to distinguish whether it is text or another format."
        "- The chat window displays only 10 messages, but earlier messages can be viewed by scrolling."
        "</des>"
    )


def append_text_part(parts: list, text: str) -> None:
    if not text:
        return
    if parts and isinstance(parts[-1], dict) and parts[-1].get("type") == "text":
        parts[-1] = {**parts[-1], "text": parts[-1].get("text", "") + text}
    else:
        parts.append({"type": "text", "text": text})


def render_platform_block(
    *,
    session: Any,
    sessions: dict[str, Any],
    current_key: str,
    current_time: str,
    chat_log: str | list,
    forward_content: str | list = "",
    account_id: str = "",
    account_name: str = "",
) -> str | list:
    unread_xml = build_unread_info_xml(sessions, current_key)
    unread_block = unread_xml if unread_xml else "<unread_info/>"
    current_time_block = f"<current_time>{current_time}</current_time>"
    platform_open = platform_open_tag(account_id=account_id, account_name=account_name)
    des = platform_description()

    if isinstance(chat_log, str) and not isinstance(forward_content, list):
        forward_block = f"\n{forward_content}" if forward_content else ""
        return (
            f"{current_time_block}\n{platform_open}\n"
            f"{des}\n{unread_block}\n{chat_log}{forward_block}\n"
            "</platform>"
        )

    parts: list = [
        {
            "type": "text",
            "text": f"{current_time_block}\n{platform_open}\n{des}\n{unread_block}\n",
        }
    ]
    if isinstance(chat_log, str):
        append_text_part(parts, chat_log)
    else:
        parts.extend(chat_log)
    if forward_content:
        append_text_part(parts, "\n")
        if isinstance(forward_content, str):
            append_text_part(parts, forward_content)
        else:
            parts.extend(forward_content)
    append_text_part(parts, "\n</platform>")
    return parts
