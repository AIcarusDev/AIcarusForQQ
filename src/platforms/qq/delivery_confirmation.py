"""QQ 发送投递确认。

adapter 在超时或断线时可能同步返回空 message_id。工具侧此时先写入一条 pending
占位消息，由本模块按 adapter 历史回查对齐真实 message_id；四次回查仍无法确认时
把该消息标记为 failed。
"""

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger("AICQ.tools")

_PENDING_RECHECK_DELAYS = (0.2, 2.0, 5.0, 10.0)


def _normalize_delivery_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def delivery_match_text(
    message: list[dict],
    *,
    bot_sender_id: str = "",
    bot_sender_name: str = "",
) -> str:
    """Compare face IDs instead of display names that can vary in QQ echoes."""
    from platforms.qq.adapter import segments as qq_segments

    comparable = [
        {"type": "text", "data": {"text": f"[QQ-face:{(seg.get('data') or {}).get('id', '')}]"}}
        if seg.get("type") == "face" else seg
        for seg in message
    ]
    return qq_segments.qq_adapter_segments_to_text(
        comparable,
        bot_id=bot_sender_id,
        bot_display_name=bot_sender_name,
    )


def _patch_session_message(session: Any, message_id: str, updates: dict[str, Any]) -> dict | None:
    messages = list(getattr(session, "context_messages", []) or [])
    for index, entry in enumerate(messages):
        if str(entry.get("message_id", "")) != str(message_id):
            continue
        updated = dict(entry)
        for key, value in updates.items():
            if value is None:
                updated.pop(key, None)
            else:
                updated[key] = value
        messages[index] = updated
        try:
            session.context_messages = messages
        except Exception:
            return None
        return updated
    return None


def history_message_matches_pending_send(
    event: dict,
    *,
    bot_sender_id: str,
    bot_sender_name: str,
    expected_text: str,
    reply_id: str | None,
    sent_started_at: float,
    known_bot_message_ids: set[str],
) -> bool:
    from platforms.qq.adapter import segments as qq_segments

    message_id = str(event.get("message_id", "") or "").strip()
    if not message_id or message_id in known_bot_message_ids:
        return False
    sender = event.get("sender") or {}
    sender_id = str(sender.get("user_id") or event.get("user_id") or "").strip()
    if sender_id != str(bot_sender_id):
        return False
    try:
        event_time = float(event.get("time") or 0)
    except (TypeError, ValueError):
        event_time = 0.0
    if event_time and event_time < sent_started_at - 5.0:
        return False

    message = event.get("message")
    if not isinstance(message, list):
        return False
    if str(qq_segments.get_reply_message_id(message) or "") != str(reply_id or ""):
        return False
    actual_text = delivery_match_text(
        message,
        bot_sender_id=str(bot_sender_id),
        bot_sender_name=str(bot_sender_name or bot_sender_id),
    )
    return _normalize_delivery_match_text(actual_text) == _normalize_delivery_match_text(expected_text)


async def _fetch_recent_history_for_confirmation(
    qq_client: Any,
    *,
    conv_type: str,
    group_id: int | None,
    user_id: int | None,
    temp_source_group_id: int | None,
) -> list[dict]:
    if not qq_client or not getattr(qq_client, "connected", False):
        return []
    if conv_type == "group":
        if group_id is None:
            return []
        action = "get_group_msg_history"
        params: dict[str, Any] = {"group_id": int(group_id), "count": 50}
    elif conv_type in {"private", "temp"}:
        if user_id is None:
            return []
        action = "get_friend_msg_history"
        params = {"user_id": int(user_id), "count": 50}
        if temp_source_group_id is not None:
            params["group_id"] = int(temp_source_group_id)
    else:
        return []

    if hasattr(qq_client, "send_api_raw"):
        resp = await qq_client.send_api_raw(action, params, timeout=20.0)
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return []
        data = resp.get("data") or {}
    else:
        data = await qq_client.send_api(action, params, timeout=20.0)
    messages = (data or {}).get("messages", [])
    return messages if isinstance(messages, list) else []


async def reconcile_pending_send(
    *,
    session: Any,
    conversation_id: str,
    internal_message_id: str,
    qq_client: Any,
    conv_type: str,
    group_id: int | None,
    user_id: int | None,
    temp_source_group_id: int | None,
    bot_sender_id: str,
    bot_sender_name: str,
    expected_text: str,
    reply_id: str | None,
    sent_started_at: float,
    known_bot_message_ids: set[str],
    pending_error: str,
) -> None:
    from database import update_chat_message_delivery_state, update_chat_message_id

    for delay in _PENDING_RECHECK_DELAYS:
        await asyncio.sleep(delay)
        try:
            history = await _fetch_recent_history_for_confirmation(
                qq_client,
                conv_type=conv_type,
                group_id=group_id,
                user_id=user_id,
                temp_source_group_id=temp_source_group_id,
            )
        except Exception:
            logger.debug("[send_message] 投递回查异常 conv=%s id=%s", conversation_id, internal_message_id, exc_info=True)
            continue

        matches = [
            event for event in history
            if history_message_matches_pending_send(
                event,
                bot_sender_id=bot_sender_id,
                bot_sender_name=bot_sender_name,
                expected_text=expected_text,
                reply_id=reply_id,
                sent_started_at=sent_started_at,
                known_bot_message_ids=known_bot_message_ids,
            )
        ]
        if not matches:
            continue
        matches.sort(key=lambda event: float(event.get("time") or 0))
        real_id = str(matches[0].get("message_id", "") or "").strip()
        if not real_id:
            continue
        _patch_session_message(
            session,
            internal_message_id,
            {
                "message_id": real_id,
                "delivery_state": None,
                "delivery_error": None,
            },
        )
        await update_chat_message_id(conversation_id, internal_message_id, real_id)
        logger.info(
            "[send_message] pending 投递已回查确认 conv=%s internal=%s real=%s",
            conversation_id,
            internal_message_id,
            real_id,
        )
        return

    failed_error = pending_error or "消息投递状态未能通过 adapter 历史回查确认。"
    _patch_session_message(
        session,
        internal_message_id,
        {
            "delivery_state": "failed",
            "delivery_error": failed_error,
        },
    )
    await update_chat_message_delivery_state(
        conversation_id,
        internal_message_id,
        "failed",
        failed_error,
    )
    logger.warning("[send_message] pending 投递确认失败 conv=%s id=%s", conversation_id, internal_message_id)
