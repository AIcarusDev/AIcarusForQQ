"""QQ platform prompt rendering."""

from __future__ import annotations

import html
from typing import Any

from .unread import build_recent_active_sessions_xml, build_unread_info_xml


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


def platform_description(*, home_view: bool = False) -> str:
    if home_view:
        body = (
            "This is the QQ platform home view for the currently logged-in account.\n"
            "- `unread_info` lists other sessions with unread messages and previews the newest unread message.\n"
            "- `recent_active_sessions` lists recently active sessions without unread messages; it is an index for awareness, not an opened chat window.\n"
            "- An empty `current_session` element means no specific chat window is open."
        )
    else:
        body = (
            "These are the messages visible in the currently opened QQ chat window.\n"
            "- `unread_info` lists other sessions with unread messages; group previews are public messages posted by other users.\n"
            "- Each current-session message has its own `type` to distinguish whether it is text or another format.\n"
            "- The chat window displays only 10 messages, but earlier messages can be viewed by scrolling."
        )
    return f"<des>{body}</des>"


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
    is_home_view = isinstance(chat_log, str) and chat_log.strip() == "<current_session/>"
    recent_xml = build_recent_active_sessions_xml(sessions, current_key) if is_home_view else ""
    recent_block = f"\n{recent_xml or '<recent_active_sessions/>'}" if is_home_view else ""
    current_time_block = f"<current_time>{current_time}</current_time>"
    platform_open = platform_open_tag(account_id=account_id, account_name=account_name)
    des = platform_description(home_view=is_home_view)

    if isinstance(chat_log, str) and not isinstance(forward_content, list):
        forward_block = f"\n{forward_content}" if forward_content else ""
        return (
            f"{current_time_block}\n{platform_open}\n"
            f"{des}\n{unread_block}{recent_block}\n{chat_log}{forward_block}\n"
            "</platform>"
        )

    parts: list = [
        {
            "type": "text",
            "text": f"{current_time_block}\n{platform_open}\n{des}\n{unread_block}{recent_block}\n",
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
