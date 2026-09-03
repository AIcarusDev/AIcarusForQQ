"""QQ conversation identity helpers.

Temp private sessions are keyed by peer user only.  The source group is stored
as metadata because it is required by adapters for opening/sending temp chats,
but it is not part of the local session identity.
"""

from __future__ import annotations

from typing import Any

TEMP_CONV_TYPE = "temp"


def make_session_key(conv_type: str, conv_id: str | int) -> str:
    return f"qq:{conv_type}:{conv_id}"


def make_temp_session_key(user_id: str | int) -> str:
    return make_session_key(TEMP_CONV_TYPE, user_id)


def parse_session_key(session_key: str | None) -> tuple[str, str]:
    raw = str(session_key or "")
    parts = raw.split(":", 2)
    if len(parts) == 3 and parts[0] == "qq":
        conv_type, conv_id = parts[1], parts[2]
    else:
        conv_type, sep, conv_id = raw.partition("_")
        if not sep:
            return "", ""
    if conv_type not in {"group", "private", TEMP_CONV_TYPE}:
        return "", ""
    return conv_type, conv_id


def event_sender_id(event: dict[str, Any]) -> str:
    return str((event.get("sender") or {}).get("user_id", "") or "").strip()


def event_group_id(event: dict[str, Any]) -> str:
    return str(event.get("group_id", "") or "").strip()


def is_temp_private_event(event: dict[str, Any]) -> bool:
    """Return whether an event is a group-origin private temp message."""

    if str(event.get("message_type", "")) != "private":
        return False
    sub_type = str(event.get("sub_type", "") or "").lower()
    if sub_type == "group":
        return True
    return bool(event_group_id(event))


def get_temp_source_group_id(session: Any) -> str:
    return str(getattr(session, "temp_source_group_id", "") or "").strip()


def get_temp_source_group_name(session: Any) -> str:
    return str(getattr(session, "temp_source_group_name", "") or "").strip()


def set_temp_source(
    session: Any,
    group_id: str | int | None,
    group_name: str | None = None,
) -> None:
    gid = str(group_id or "").strip()
    if gid:
        session.temp_source_group_id = gid
    if group_name:
        session.temp_source_group_name = str(group_name)


def format_adapter_error(api_error: dict[str, Any] | None, fallback: str = "QQ adapter 调用失败") -> str:
    """Expose only bounded adapter metadata, never adapter-provided prose."""

    if not api_error:
        return fallback
    raw_action = str(api_error.get("action") or "").strip()
    action = raw_action if raw_action.replace("_", "").isalnum() and len(raw_action) <= 64 else ""
    raw_status = str(api_error.get("status") or "failed").strip().lower()
    status = raw_status if raw_status in {"failed", "error", "timeout", "disconnected"} else "failed"
    parts = []
    if action:
        parts.append(action)
    parts.append(status)
    retcode = api_error.get("retcode")
    if (isinstance(retcode, int) and not isinstance(retcode, bool)) or (
        isinstance(retcode, str)
        and retcode.removeprefix("-").isdigit()
        and len(retcode) <= 12
    ):
        parts.append(f"retcode={retcode}")
    return "QQ adapter 返回错误: " + " / ".join(parts) if parts else fallback
