"""QQ 发送流程。

负责把一批消息真正送出去：统一的结果信封、发送前图片预检、adapter 调用结果
分类，以及成功/待确认消息的入库、上下文写入和 debug 前端广播。
"""

import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from platforms.qq.adapter.conversation import format_adapter_error
from platforms.qq.delivery_confirmation import delivery_match_text, reconcile_pending_send
from platforms.qq.image_pipeline import (
    materialize_browser_resources,
    materialize_local_image_paths,
    unconfirmed_high_risk_image_error,
)

logger = logging.getLogger("AICQ.tools")


def build_result(
    target: str,
    *,
    sent_count: int = 0,
    failed_count: int = 0,
    total_count: int = 0,
    interrupted: bool = False,
    **extra: Any,
) -> dict:
    """Assemble the fixed send_message result envelope, then append extra fields."""
    result: dict[str, Any] = {
        "to": target,
        "sent_count": sent_count,
        "failed_count": failed_count,
        "total_count": total_count,
        "interrupted": interrupted,
    }
    result.update(extra)
    return result


def failure_result(
    target: str,
    error: str,
    total_count: int,
    *,
    sent_count: int = 0,
    failed_count: int | None = None,
    interrupted: bool = False,
) -> dict:
    """Failure envelope; by default every unsent message counts as failed."""
    return build_result(
        target,
        sent_count=sent_count,
        failed_count=(total_count - sent_count if failed_count is None else failed_count),
        total_count=total_count,
        interrupted=interrupted,
        error=error,
    )


def adapter_response_is_definitive_failure(api_error: dict[str, Any] | None) -> bool:
    """Return True when the adapter has synchronously rejected the send."""

    if not api_error:
        return False
    status = str(api_error.get("status") or "").strip().lower()
    if status in {"failed", "failure", "error"}:
        return True
    if status in {"timeout", "disconnected"}:
        return False
    retcode = api_error.get("retcode")
    return retcode not in (None, "", 0, "0")


@dataclass(frozen=True)
class SendTarget:
    """Addressing info shared by every message in one send batch."""

    conversation_id: str
    conv_type: str
    conv_id: str
    group_id: int | None
    user_id: int | None
    temp_source_group_id: int | None
    bot_sender_id: str
    bot_sender_name: str


@dataclass
class SendOutcome:
    """Adapter verdict for one message: sent, failed, offline, or pending."""

    kind: str
    real_id: str = ""
    delivery_state: str = ""
    delivery_error: str = ""


def load_face_catalog(
    config: dict | None,
    send_messages: list,
) -> tuple[dict[int, str], set[int]]:
    """Load face display text and super-only ids, but only for batches using faces."""
    has_face = any(
        isinstance(msg, dict)
        and isinstance(msg.get("segments"), list)
        and any(isinstance(seg, dict) and seg.get("command") == "face" for seg in msg["segments"])
        for msg in send_messages
    )
    if not has_face:
        return {}, set()
    from platforms.qq.tools.qq_social.list_faces import load_face_entries

    face_entries = load_face_entries(config or {})
    return (
        {face_id: face.des for face_id, face in face_entries.items()},
        {face_id for face_id, face in face_entries.items() if face.super and not face.normal},
    )


def preflight_images(
    send_messages: list[dict],
    *,
    session: Any,
    config: dict | None,
    loop: asyncio.AbstractEventLoop,
    target: str,
    round_inbound_revision: int | None,
    simulate_typing: bool,
) -> tuple[list[dict], dict | None]:
    """Materialize every image source before sending.

    Returns the prepared messages and, when the batch must stop, a ready-to-return
    response (image failure, invalidated intent, or high-risk confirmation).
    """
    import app_state
    from tools._async_bridge import run_coroutine_sync

    prepared_messages, local_image_error = run_coroutine_sync(
        materialize_local_image_paths(
            send_messages,
            getattr(app_state, "workspace_service", None),
        ),
        loop,
    )
    if local_image_error or prepared_messages is None:
        return send_messages, failure_result(
            target,
            local_image_error or "Agent Linux 图片读取失败。",
            len(send_messages),
        )
    send_messages = prepared_messages
    if (
        any(
            isinstance(segment, dict) and segment.get("_local_image_base64")
            for message in send_messages
            if isinstance(message, dict)
            for segment in (message.get("segments") or [])
        )
        and round_inbound_revision is not None
        and int(getattr(session, "inbound_revision", 0) or 0) != int(round_inbound_revision)
    ):
        return send_messages, failure_result(
            target,
            "读取图片期间收到新消息，本次图片发送意图已失效，未发送。",
            len(send_messages),
            interrupted=True,
        )
    confirmation_bypass_error = unconfirmed_high_risk_image_error(send_messages, session, config)
    if confirmation_bypass_error:
        return send_messages, failure_result(target, confirmation_bypass_error, len(send_messages))
    prepared_messages, browser_artifacts, browser_error = materialize_browser_resources(send_messages)
    if browser_error or prepared_messages is None:
        return send_messages, failure_result(
            target,
            browser_error or "浏览器原图固化失败。",
            len(send_messages),
        )
    send_messages = prepared_messages
    if not browser_artifacts:
        return send_messages, None
    if (
        round_inbound_revision is not None
        and int(getattr(session, "inbound_revision", 0) or 0) != int(round_inbound_revision)
    ):
        return send_messages, failure_result(
            target,
            "固化期间收到新消息，本次图片发送意图已失效，未发送。",
            len(send_messages),
            interrupted=True,
        )
    from browser.config import browser_image_send_confirmation

    confirmation_reasons = list(dict.fromkeys(
        str(reason)
        for artifact in browser_artifacts
        for reason in artifact.get("confirmation_reasons") or []
    ))
    if browser_image_send_confirmation(config) != "high_risk" or not confirmation_reasons:
        return send_messages, None
    from browser.image_confirmation import stage_pending

    pending = stage_pending(
        session,
        messages=send_messages,
        artifacts=browser_artifacts,
        simulate_typing=bool(simulate_typing),
    )
    return send_messages, build_result(
        target,
        total_count=len(send_messages),
        confirmation_required=True,
        batch_id=pending.batch_id,
        image_count=len(browser_artifacts),
        confirmation_reasons=confirmation_reasons,
        note="最终原图将在下一正常复合轮展示；该轮未确认或取消会自动失效。",
    )


def send_one(
    target: SendTarget,
    *,
    qq_client: Any,
    loop: asyncio.AbstractEventLoop,
    message_index: int,
    qq_adapter_segs: list[dict],
    simulate_typing: bool,
    offline_mode: bool,
) -> SendOutcome:
    """Send one converted message and turn the adapter response into a verdict."""
    if offline_mode:
        return SendOutcome(
            kind="offline",
            real_id=f"offline_{uuid.uuid4().hex[:8]}",
            delivery_state="failed",
            delivery_error="QQ adapter 未连接；消息只保存在本地，未投递。",
        )

    from tools._async_bridge import run_coroutine_sync

    send_exception = None
    try:
        send_result = run_coroutine_sync(
            qq_client.send_message(
                group_id=target.group_id,
                user_id=target.user_id,
                temp_source_group_id=target.temp_source_group_id,
                message=qq_adapter_segs,
                llm_elapsed=0.0,
                simulate_typing=bool(simulate_typing),
            ),
            loop,
            timeout=30,
        )
    except Exception as e:
        logger.warning("[send_message] 发送第 %d 条消息失败: %s", message_index + 1, e)
        send_exception = e
        send_result = None

    if send_result and send_result.get("message_id") is not None:
        return SendOutcome(kind="sent", real_id=str(send_result["message_id"]))

    last_api_error = getattr(qq_client, "last_api_error", None)
    if send_exception is None and adapter_response_is_definitive_failure(last_api_error):
        return SendOutcome(
            kind="failed",
            delivery_error=format_adapter_error(last_api_error, "QQ adapter send_msg failed"),
        )
    return SendOutcome(
        kind="pending",
        real_id=f"pending_{uuid.uuid4().hex[:8]}",
        delivery_state="pending",
        delivery_error=format_adapter_error(
            last_api_error,
            "QQ adapter send_msg failed or returned no message_id",
        ),
    )


def record_bot_message(
    target: SendTarget,
    *,
    session: Any,
    loop: asyncio.AbstractEventLoop,
    qq_client: Any,
    segments: list[dict],
    text: str,
    content_segments: list[dict],
    content_type: str,
    reply_id: str | None,
    outcome: SendOutcome,
    qq_adapter_segs: list[dict],
    send_started_at: float,
    known_bot_message_ids: set[str],
) -> None:
    """Store the bot entry in context/DB and fan it out to the debug UI."""
    import app_state
    from database import save_chat_message
    from llm.core.round_context import get_current_inner_state
    from web.debug_server import broadcast_chat_event

    entry: dict = {
        "role": "bot",
        "message_id": outcome.real_id,
        "sender_id": target.bot_sender_id,
        "sender_name": target.bot_sender_name,
        "sender_role": "",
        "timestamp": datetime.now(app_state.TIMEZONE).isoformat(),
        "content": text,
        "content_type": content_type,
        "content_segments": content_segments,
    }
    # Ordinary message image metadata keeps uncollected sends visible in history.
    # Collection images remain addressable by ref without duplicating their bytes in the DB.
    sent_images = {
        seg["image_ref"]: {
            "base64": base64.b64encode(seg["_image_bytes"]).decode("ascii"),
            "mime": seg["_image_mime"],
            "label": "动画表情",
        }
        for seg in segments
        if seg.get("command") == "sticker" and seg.get("_image_source") != "sticker"
    }
    if sent_images:
        entry["images"] = sent_images
    if outcome.delivery_state:
        entry["delivery_state"] = outcome.delivery_state
    if outcome.delivery_error:
        entry["delivery_error"] = outcome.delivery_error
    if reply_id:
        entry["reply_to"] = str(reply_id)
    session.add_to_context(entry)

    # 持久化（fire-and-forget，不阻塞发送循环）
    asyncio.run_coroutine_threadsafe(save_chat_message(target.conversation_id, entry), loop)
    if outcome.delivery_state == "pending":
        asyncio.run_coroutine_threadsafe(
            reconcile_pending_send(
                session=session,
                conversation_id=target.conversation_id,
                internal_message_id=outcome.real_id,
                qq_client=qq_client,
                conv_type=target.conv_type,
                group_id=target.group_id,
                user_id=target.user_id,
                temp_source_group_id=target.temp_source_group_id,
                bot_sender_id=str(target.bot_sender_id),
                bot_sender_name=str(target.bot_sender_name),
                expected_text=delivery_match_text(
                    qq_adapter_segs,
                    bot_sender_id=str(target.bot_sender_id),
                    bot_sender_name=str(target.bot_sender_name),
                ),
                reply_id=str(reply_id) if reply_id else None,
                send_started_at=send_started_at,
                known_bot_message_ids=set(known_bot_message_ids),
                pending_error=outcome.delivery_error,
            ),
            loop,
        )
    known_bot_message_ids.add(outcome.real_id)

    # 广播到 debug 前端
    asyncio.run_coroutine_threadsafe(
        broadcast_chat_event({
            "type": "bot_turn",
            "conv_id": target.conversation_id,
            "conv_name": session.conv_name or target.conversation_id,
            "conv_type": target.conv_type or "unknown",
            "entries": [entry],
            "inner_state": get_current_inner_state(),
        }),
        loop,
    )
