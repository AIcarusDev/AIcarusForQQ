"""send_message.py — 发送消息工具实现

Handler 运行在 asyncio.to_thread 派生的线程中，
所有 async 操作通过 asyncio.run_coroutine_threadsafe + app_state.main_loop 执行。
"""

import asyncio
import base64
import copy
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync
from qq_adapter.conversation import format_adapter_error

from .prompt import get_description

logger = logging.getLogger("AICQ.tools")


DECLARATION: dict = {
    "name": "send_message",
}

EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}

REQUIRES_CONTEXT: list[str] = ["session", "qq_adapter_client"]

_SEND_MESSAGE_TAIL_LEAK_RE = re.compile(
    r'^(?P<body>.*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?P<key>messages|segments|quote|command|content|user_id|image_ref|sticker_id)"?)\s*:.*)$',
    re.DOTALL,
)

_STICKER_REF_FALLBACK_WARNING = (
    'The provided "sticker_id" was invalid; however, the system still sent a '
    "sticker based on a hash match—serving as a fallback mechanism. Whenever "
    'possible, please use "list_stickers" to check your sticker collection '
    "first before initiating a send action. If the sticker sent by the system "
    "in this instance does not meet your expectations, you may retract it."
)


_MESSAGE_SHAPE_ARRAY = "array"
_MESSAGE_SHAPE_SINGLE = "single"
_SINGLE_SHAPE_ALIASES = {"single", "single_message", "message", "segments"}
_ARRAY_SHAPE_ALIASES = {"array", "messages", "multi", "multi_message", "batch"}
_PENDING_RECHECK_DELAYS = (0.2, 2.0, 5.0, 10.0)


def get_send_message_shape(config: dict | None = None) -> str:
    """Return the configured model-facing send_message shape."""
    tools_cfg = (config or {}).get("tools")
    send_cfg: Any = {}
    if isinstance(tools_cfg, dict):
        send_cfg = tools_cfg.get("send_message", {})
    raw_shape: Any = None
    if isinstance(send_cfg, dict):
        raw_shape = (
            send_cfg.get("message_shape")
            or send_cfg.get("shape")
            or send_cfg.get("mode")
        )
    elif isinstance(send_cfg, str):
        raw_shape = send_cfg
    shape = str(raw_shape or "").strip().lower().replace("-", "_")
    if shape in _SINGLE_SHAPE_ALIASES:
        return _MESSAGE_SHAPE_SINGLE
    if shape in _ARRAY_SHAPE_ALIASES:
        return _MESSAGE_SHAPE_ARRAY
    return _MESSAGE_SHAPE_ARRAY


_QUOTE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "x-coerce-integer": True,
    "description": "要引用/回复的目标消息 ID（可选）。",
}


_SEGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["text", "at", "image", "sticker"],
        },
        "content": {"type": "string"},
        "user_id": {"type": "string"},
        "image_ref": {
            "type": "string",
            "description": "<world> 中的 image ref，例如 3a686ed196bf。",
        },
        "sticker_id": {"type": "string"},
    },
    "required": ["command"],
    "allOf": [
        {
            "if": {"properties": {"command": {"const": "text"}}},
            "then": {"required": ["content"]},
        },
        {
            "if": {"properties": {"command": {"const": "at"}}},
            "then": {"required": ["user_id"]},
        },
        {
            "if": {"properties": {"command": {"const": "image"}}},
            "then": {"required": ["image_ref"]},
        },
        {
            "if": {"properties": {"command": {"const": "sticker"}}},
            "then": {"required": ["sticker_id"]},
        },
    ],
}


_SEGMENTS_SCHEMA: dict[str, Any] = {
    "type": "array",
    "description": "该条消息的内容片段。",
    "items": _SEGMENT_SCHEMA,
    "minItems": 1,
}


def _single_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "quote": copy.deepcopy(_QUOTE_SCHEMA),
            "segments": copy.deepcopy(_SEGMENTS_SCHEMA),
        },
        "required": ["segments"],
    }


def _array_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "description": "要发送的消息列表，每个元素作为一条消息独立发送。",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": copy.deepcopy(_QUOTE_SCHEMA),
                        "segments": copy.deepcopy(_SEGMENTS_SCHEMA),
                    },
                    "required": ["segments"],
                },
                "minItems": 1,
            },
        },
        "required": ["messages"],
    }


def get_declaration(session: Any | None = None, config: dict | None = None, **_: Any) -> dict:
    message_shape = get_send_message_shape(config)
    return {
        "name": "send_message",
        "description": get_description(message_shape),
        "parameters": (
            _single_parameters_schema()
            if message_shape == _MESSAGE_SHAPE_SINGLE
            else _array_parameters_schema()
        ),
    }


def get_prompt_signature(config: dict | None = None, **_: Any) -> str:
    message_shape = get_send_message_shape(config)
    if message_shape == _MESSAGE_SHAPE_SINGLE:
        return """
// 向当前打开的会话窗口发送一条消息。
// 内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包、图片等不同类型片段拼合为单条消息发送。
// 如需发送多条消息，按顺序多次调用该工具即可。
// 注意：
// - 私聊和临时会话无法发送 @某人（at）片段。当前会话是私聊/临时会话时，如果消息包含 at，会发送失败。
// - 消息会发送到当前会话，如果你想回应的是其它会话的未读消息，需先 shift 到指定会话。
send_message(args: {
  quote?: string; // 要引用/回复的目标消息 ID（可选）。
  segments: (
    | { command: "text"; content: string }
    | { command: "at"; user_id: string }
    | { command: "image"; image_ref: string } // <world> 中的 image ref，例如 3a686ed196bf。
    | { command: "sticker"; sticker_id: string }
  )[]; // 该条消息的内容片段。
})
"""
    return """
// 向当前打开的会话窗口发送一条或多条消息。
// "messages" 参数是一个列表，每个列表项都是一条独立消息，会按顺序依次发送。
// 每条消息内部的 "segments" 字段是内容片段列表，用于将文字、@某人、表情包、图片等不同类型片段拼合为单条消息发送。
// 注意：
// - 同一条消息内的多个 segment 只会被拼接为一条消息，并不会变成多条。若要发送多条独立消息，请在 messages 数组中添加多个元素。
// - 私聊和临时会话无法发送 @某人（at）片段。当前会话是私聊/临时会话时，如果某条消息包含 at，该条消息会发送失败。
// - 消息会发送到当前会话，如果你想回应的是其它会话的未读消息，需先 shift 到指定会话。
send_message(args: {
  messages: {
    quote?: string; // 要引用/回复的目标消息 ID（可选）。
    segments: (
      | { command: "text"; content: string }
      | { command: "at"; user_id: string }
      | { command: "image"; image_ref: string } // <world> 中的 image ref，例如 3a686ed196bf。
      | { command: "sticker"; sticker_id: string }
    )[]; // 该条消息的内容片段。
  }[]; // 要发送的消息列表，每个元素作为一条消息独立发送。
})
"""


def _repair_schema_args_for_shape(
    args: dict[str, Any],
    message_shape: str,
) -> tuple[dict[str, Any], list[str]]:
    """修复 send_message 的 messages 容器结构性字段错误。"""
    repair_notes: list[str] = []
    messages = args.get("messages")
    root_segments = args.get("segments")
    if (
        message_shape == _MESSAGE_SHAPE_ARRAY
        and not isinstance(messages, list)
        and isinstance(root_segments, list)
    ):
        message: dict[str, Any] = {"segments": root_segments}
        if "quote" in args and args.get("quote") not in (None, ""):
            quote = args.get("quote")
            if isinstance(quote, int) and not isinstance(quote, bool):
                quote = str(quote)
            message["quote"] = quote
        repaired_args = {
            key: value
            for key, value in args.items()
            if key not in {"segments", "quote"}
        }
        repaired_args["messages"] = [message]
        return repaired_args, ["wrapped root single-message fields into messages[0]"]

    if not isinstance(messages, list):
        return args, repair_notes

    normalized_messages: list[Any] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            normalized_messages.append(item)
            continue

        segments = item.get("segments")
        if not isinstance(segments, list):
            normalized_messages.append(item)
            continue

        current_segments: list[Any] = []
        leaked_messages: list[dict[str, Any]] = []
        for segment in segments:
            if (
                isinstance(segment, dict)
                and "segments" in segment
                and "command" not in segment
            ):
                leaked_messages.append(dict(segment))
                continue
            current_segments.append(segment)

        if not leaked_messages:
            normalized_messages.append(item)
            continue

        repaired_item = dict(item)
        repaired_item["segments"] = current_segments
        if current_segments:
            normalized_messages.append(repaired_item)
        normalized_messages.extend(leaked_messages)
        repair_notes.append(f"split leaked message objects from messages[{index}].segments")

    if normalized_messages == messages:
        return args, repair_notes

    repaired_args = dict(args)
    repaired_args["messages"] = normalized_messages
    return repaired_args, repair_notes


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Compatibility wrapper for array-shaped send_message schema repair."""
    return _repair_schema_args_for_shape(args, _MESSAGE_SHAPE_ARRAY)


def make_schema_repairer(
    config: dict | None = None,
) -> Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]:
    message_shape = get_send_message_shape(config)

    def _repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        return _repair_schema_args_for_shape(args, message_shape)

    return _repair_schema_args


def _strip_tool_arg_tail_leak(text: str) -> tuple[str, bool]:
    """截断被错误吞进字符串里的后续 JSON 尾巴。"""
    match = _SEND_MESSAGE_TAIL_LEAK_RE.match(text)
    if not match:
        return text, False
    cleaned = match.group("body").rstrip()
    if not cleaned:
        return text, False
    return cleaned, True


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    """去除文本污染，并按消息语义拆分连续 text segments。"""
    changes: list[str] = []

    def _walk(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {
                key: _walk(nested, f"{path}.{key}" if path else str(key))
                for key, nested in value.items()
            }
        if isinstance(value, list):
            return [_walk(nested, f"{path}[{index}]") for index, nested in enumerate(value)]
        if isinstance(value, str):
            cleaned, changed = _strip_tool_arg_tail_leak(value)
            if changed:
                changes.append(f"trimmed leaked tail in {path or '<root>'}")
                return cleaned
        return value

    sanitized = _walk(args, "")
    repaired_args = sanitized if isinstance(sanitized, dict) else args
    messages = repaired_args.get("messages")
    if isinstance(messages, list):
        expanded = _expand_messages(messages)
        if expanded != messages:
            if repaired_args is args:
                repaired_args = dict(args)
            repaired_args["messages"] = expanded
            changes.append(
                f"expanded messages by splitting consecutive text segments ({len(messages)} -> {len(expanded)})"
            )
    return repaired_args, changes, None


def _extract_message_text(segments: list[dict]) -> tuple[str, list[dict], str]:
    """从 segments 提取纯文本和结构化 content_segments。"""
    text_parts: list[str] = []
    content_segments: list[dict] = []
    for seg in segments:
        cmd = seg.get("command", "")
        if cmd == "text":
            t = seg.get("content", "")
            text_parts.append(t)
            if t:
                content_segments.append({"type": "text", "text": t})
        elif cmd == "at":
            uid = str(seg.get("user_id", ""))
            text_parts.append(f"@{uid}")
            content_segments.append({"type": "mention", "uid": uid, "display": f"@{uid}"})
        elif cmd == "sticker":
            sticker_id = seg.get("sticker_id", "")
            text_parts.append("[动画表情]")
            content_segments.append({"type": "sticker", "sticker_id": sticker_id})
        elif cmd == "image":
            image_ref = seg.get("image_ref", "")
            text_parts.append("[图片]")
            content_segments.append({"type": "image", "image_ref": image_ref})
    text = "".join(text_parts)
    has_sticker = any(s.get("type") == "sticker" for s in content_segments)
    has_image = any(s.get("type") == "image" for s in content_segments)
    has_text = any(s.get("type") == "text" for s in content_segments)
    content_type = "sticker" if has_sticker and not has_text else "image" if has_image and not has_text else "text"
    return text, content_segments, content_type


def _normalize_delivery_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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


def _history_message_matches_pending_send(
    event: dict,
    *,
    bot_sender_id: str,
    bot_sender_name: str,
    expected_text: str,
    reply_id: str | None,
    sent_started_at: float,
    known_bot_message_ids: set[str],
) -> bool:
    from qq_adapter.segments import get_reply_message_id, qq_adapter_segments_to_text

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
    if str(get_reply_message_id(message) or "") != str(reply_id or ""):
        return False
    actual_text = qq_adapter_segments_to_text(
        message,
        bot_id=str(bot_sender_id),
        bot_display_name=str(bot_sender_name or bot_sender_id),
    )
    return _normalize_delivery_match_text(actual_text) == _normalize_delivery_match_text(expected_text)


async def _fetch_recent_history_for_confirmation(
    qq_adapter_client: Any,
    *,
    conv_type: str,
    group_id: int | None,
    user_id: int | None,
    temp_source_group_id: int | None,
) -> list[dict]:
    if not qq_adapter_client or not getattr(qq_adapter_client, "connected", False):
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

    if hasattr(qq_adapter_client, "send_api_raw"):
        resp = await qq_adapter_client.send_api_raw(action, params, timeout=20.0)
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return []
        data = resp.get("data") or {}
    else:
        data = await qq_adapter_client.send_api(action, params, timeout=20.0)
    messages = (data or {}).get("messages", [])
    return messages if isinstance(messages, list) else []


async def _reconcile_pending_send(
    *,
    session: Any,
    conversation_id: str,
    internal_message_id: str,
    qq_adapter_client: Any,
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
                qq_adapter_client,
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
            if _history_message_matches_pending_send(
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


def _message_has_at_segment(segments: list[dict]) -> bool:
    return any(isinstance(seg, dict) and seg.get("command") == "at" for seg in segments)


def _format_result_target(session: Any, temp_source_group_id: int | None = None) -> str:
    conv_type = str(getattr(session, "conv_type", "") or "")
    conv_id = str(getattr(session, "conv_id", "") or "")
    if conv_type == "temp":
        source_group_id = str(temp_source_group_id or getattr(session, "temp_source_group_id", "") or "").strip()
        if source_group_id:
            return f"temp_{conv_id}@group_{source_group_id}"
        return f"temp_{conv_id}"
    if conv_type:
        return f"{conv_type}_{conv_id}" if conv_id else conv_type
    return conv_id or "unknown"


def _resolve_send_target(session: Any) -> tuple[int | None, int | None, int | None, str | None]:
    conv_type = getattr(session, "conv_type", "")
    conv_id = getattr(session, "conv_id", "")
    try:
        if conv_type == "group":
            return int(conv_id), None, None, None
        if conv_type == "private":
            return None, int(conv_id), None, None
        if conv_type == "temp":
            source_group_id = str(getattr(session, "temp_source_group_id", "") or "").strip()
            if not source_group_id:
                return None, None, None, "临时会话缺少来源群，无法发送。请先从可用群聊打开该临时会话。"
            return None, int(conv_id), int(source_group_id), None
    except (ValueError, TypeError):
        return None, None, None, f"会话 ID 无效: {conv_id}"
    return None, None, None, f"当前会话类型不支持发送 QQ 消息: {conv_type or 'unknown'}"


def _load_context_sticker_ref(session: Any, image_ref: str) -> tuple[bytes, str] | None:
    for entry in reversed(getattr(session, "context_messages", []) or []):
        images = entry.get("images") or {}
        if not isinstance(images, dict) or image_ref not in images:
            continue

        target_img = images[image_ref] or {}
        b64: str = target_img.get("base64", "")
        mime: str = target_img.get("mime", "image/jpeg")
        raw_bytes: bytes | None = None
        if b64:
            try:
                raw_bytes = base64.b64decode(b64)
            except Exception as exc:
                logger.warning("[send_message] 表情 ref base64 解码失败 ref=%s: %s", image_ref, exc)

        if raw_bytes is None and (phash := target_img.get("phash")):
            try:
                from llm.media.image_cache import read_image_bytes
                raw_bytes = read_image_bytes(str(phash))
            except Exception as exc:
                logger.warning("[send_message] 表情 ref 缓存读取失败 ref=%s phash=%s: %s", image_ref, phash, exc)

        if raw_bytes is not None:
            return raw_bytes, mime
    return None


def _prepare_sendable_segments(
    segments: list[dict],
    session: Any,
) -> tuple[list[dict] | None, str | None, list[str]]:
    has_sendable = False
    prepared_segments: list[dict] = []
    warnings: list[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            prepared_segments.append(seg)
            continue
        prepared_seg = copy.deepcopy(seg)
        cmd = seg.get("command", "")

        if cmd == "text" and str(seg.get("content", "") or ""):
            has_sendable = True
        elif cmd == "at" and str(seg.get("user_id", "") or ""):
            has_sendable = True
        elif cmd == "sticker":
            sticker_id = str(seg.get("sticker_id", "") or "")
            if not sticker_id:
                return None, "sticker segment 缺少 sticker_id。发送表情包前请先调用 list_stickers 获取自己的表情包 ID。", warnings
            try:
                from llm.media.sticker_collection import load_sticker_bytes
                sticker_data = load_sticker_bytes(sticker_id)
            except Exception as exc:
                logger.warning("[send_message] 校验表情包失败 id=%s: %s", sticker_id, exc)
                sticker_data = None
            if sticker_data is None:
                fallback = _load_context_sticker_ref(session, sticker_id)
                if fallback is None:
                    return None, (
                        f"表情包 sticker_id \"{sticker_id}\" 不存在。"
                    ), warnings
                raw_bytes, mime = fallback
                prepared_seg["_fallback_base64"] = base64.b64encode(raw_bytes).decode("ascii")
                prepared_seg["_fallback_mime"] = mime
                prepared_seg["_fallback_ref"] = sticker_id
                warnings.append(_STICKER_REF_FALLBACK_WARNING)
            has_sendable = True
        elif cmd == "image":
            image_ref = str(seg.get("image_ref", "") or "")
            if not image_ref:
                return None, "image segment 缺少 image_ref。", warnings
            has_sendable = True
        prepared_segments.append(prepared_seg)

    if not has_sendable:
        return None, "消息没有可发送的内容，未发送。", warnings
    return prepared_segments, None, warnings


def _split_consecutive_texts(segments: list[dict]) -> list[list[dict]]:
    """将含连续 text segments 的消息拆分为多组。"""
    if not segments:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_was_text = False
    for seg in segments:
        is_text = isinstance(seg, dict) and seg.get("command") == "text"
        if is_text and prev_was_text:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
        prev_was_text = is_text
    if current:
        groups.append(current)
    return groups


def _expand_messages(messages: list) -> list:
    """将 messages 列表中每条消息的连续 text segments 拆分为多条独立消息。"""
    result: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        segs = msg.get("segments", [])
        if not isinstance(segs, list):
            result.append(msg)
            continue
        groups = _split_consecutive_texts(segs)
        if len(groups) <= 1:
            result.append(msg)
            continue
        result.append({**msg, "segments": groups[0]})
        for group in groups[1:]:
            result.append({"segments": group})
    return result


def _coerce_execute_messages(
    messages: list | None,
    segments: list | None,
    quote: str | None,
) -> tuple[list[dict] | None, str | None]:
    if isinstance(messages, list):
        if not messages:
            return None, "messages must be a non-empty array."
        return messages, None
    if isinstance(segments, list):
        if not segments:
            return None, "segments must be a non-empty array."
        message: dict[str, Any] = {"segments": segments}
        if quote:
            message["quote"] = quote
        return [message], None
    return None, "messages must be a non-empty array, or segments must be a non-empty array."


def make_handler(session: Any, qq_adapter_client: Any) -> Callable:
    def execute(
        messages: list | None = None,
        segments: list | None = None,
        quote: str | None = None,
        **kwargs,
    ) -> dict:
        import app_state
        from qq_adapter import llm_segments_to_qq_adapter, ImageLoadError
        from database import save_chat_message
        from llm.core.round_context import get_current_inner_state
        from web.debug_server import broadcast_chat_event

        send_messages, message_error = _coerce_execute_messages(messages, segments, quote)
        if message_error or send_messages is None:
            return {
                "to": _format_result_target(session),
                "error": message_error or "messages must be a non-empty array.",
                "sent_count": 0,
                "failed_count": 1,
                "total_count": 1,
                "interrupted": False,
            }

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        target = _format_result_target(session)
        if loop is None or not loop.is_running():
            return {
                "to": target,
                "error": "主事件循环不可用",
                "sent_count": 0,
                "failed_count": len(send_messages),
                "total_count": len(send_messages),
                "interrupted": False,
            }

        qq_adapter_available = bool(qq_adapter_client and qq_adapter_client.connected)

        # 确定发送目标
        conv_type = session.conv_type
        conv_id = session.conv_id
        group_id, user_id, temp_source_group_id, target_error = _resolve_send_target(session)
        target = _format_result_target(session, temp_source_group_id)
        if target_error:
            return {
                "to": target,
                "error": target_error,
                "sent_count": 0,
                "failed_count": len(send_messages),
                "total_count": len(send_messages),
                "interrupted": False,
            }

        # QQ adapter 不可用时降级运行：仅入库/入上下文，不实际发送。
        offline_mode = not qq_adapter_available
        if offline_mode and conv_type == "temp":
            return {
                "to": target,
                "error": "QQ adapter 未连接，无法发送临时会话消息",
                "sent_count": 0,
                "failed_count": len(send_messages),
                "total_count": len(send_messages),
                "interrupted": False,
            }
        if offline_mode and not str(conv_id).replace("_", "").replace("-", "").replace(".", "").isalnum():
            return {
                "to": target,
                "error": "QQ adapter 未连接",
                "sent_count": 0,
                "failed_count": len(send_messages),
                "total_count": len(send_messages),
                "interrupted": False,
            }

        conversation_id = f"{conv_type}_{conv_id}"
        bot_sender_id = session._qq_id or "bot"
        bot_sender_name = session._qq_name or ""

        # 发送前快照现有非 bot 消息 ID，用于统计发送期间新增消息数。
        pre_send_ids: set[str] = {
            str(m["message_id"])
            for m in session.context_messages
            if m.get("message_id") is not None and m.get("role") != "bot"
        }
        known_bot_message_ids: set[str] = {
            str(m["message_id"])
            for m in session.context_messages
            if m.get("message_id") is not None and m.get("role") == "bot"
        }
        sent_count: int = 0
        failed_count: int = 0
        failed_messages: list[dict] = []
        warnings: list[str] = []

        for i, msg in enumerate(send_messages):
            if not isinstance(msg, dict):
                failed_count += 1
                failed_messages.append({
                    "index": i,
                    "reason": "message item must be an object.",
                })
                continue
            segments = msg.get("segments", [])
            if not isinstance(segments, list):
                segments = []
            if conv_type in {"private", "temp"} and _message_has_at_segment(segments):
                failed_count += 1
                failed_messages.append({
                    "index": i,
                    "reason": "私聊不支持 at；临时会话同样不支持 at；包含 at 的消息已发送失败。",
                })
                logger.warning(
                    "[send_message] 私聊/临时会话消息包含 at segment，拒绝发送 conv=%s idx=%d",
                    conversation_id,
                    i,
                )
                continue
            prepared_segments, validation_error, segment_warnings = _prepare_sendable_segments(segments, session)
            if validation_error:
                failed_count += 1
                failed_messages.append({
                    "index": i,
                    "reason": validation_error,
                })
                logger.warning(
                    "[send_message] 消息段校验失败 conv=%s idx=%d reason=%s",
                    conversation_id,
                    i,
                    validation_error,
                )
                continue
            if segment_warnings:
                warnings.extend(segment_warnings)
            segments = prepared_segments or []
            reply_id = msg.get("quote") or None
            try:
                qq_adapter_segs = llm_segments_to_qq_adapter(
                    segments,
                    reply_message_id=reply_id,
                    adapter=getattr(qq_adapter_client, "adapter", ""),
                )
            except ImageLoadError as img_err:
                logger.warning("[send_message] 图片加载失败，终止本次发送 conv=%s — %s", conversation_id, img_err)
                return {
                    "to": target,
                    "error": str(img_err),
                    "sent_count": sent_count,
                    "failed_count": failed_count + 1,
                    "total_count": len(send_messages),
                    "interrupted": False,
                }
            if not qq_adapter_segs:
                failed_count += 1
                failed_messages.append({
                    "index": i,
                    "reason": "message converted to empty QQ adapter segments",
                })
                logger.warning(
                    "[send_message] 消息转换后为空 conv=%s idx=%d",
                    conversation_id,
                    i,
                )
                continue

            text, content_segments, content_type = _extract_message_text(segments)
            send_started_at = time.time()

            # 发送消息（异步→同步）
            if offline_mode:
                send_result = None
            else:
                try:
                    send_result = run_coroutine_sync(
                        qq_adapter_client.send_message(
                            group_id=group_id,
                            user_id=user_id,
                            temp_source_group_id=temp_source_group_id,
                            message=qq_adapter_segs,
                            llm_elapsed=0.0,
                        ),
                        loop,
                        timeout=30,
                    )
                except Exception as e:
                    logger.warning("[send_message] 发送第 %d 条消息失败: %s", i + 1, e)
                    send_result = None

            now_ts = datetime.now(app_state.TIMEZONE).isoformat()

            if offline_mode:
                real_id = f"offline_{uuid.uuid4().hex[:8]}"
                delivery_state = "failed"
                delivery_error = "QQ adapter 未连接；消息只保存在本地，未投递。"
                failed_count += 1
                failed_messages.append({
                    "index": i,
                    "reason": delivery_error,
                })
            elif send_result and send_result.get("message_id") is not None:
                real_id = str(send_result["message_id"])
                delivery_state = ""
                delivery_error = ""
                sent_count += 1
            else:
                real_id = f"pending_{uuid.uuid4().hex[:8]}"
                delivery_state = "pending"
                delivery_error = format_adapter_error(
                    getattr(qq_adapter_client, "last_api_error", None),
                    "QQ adapter send_msg failed or returned no message_id",
                )
                logger.warning("[send_message] 消息投递状态待确认 conv=%s idx=%d", conversation_id, i)

            entry: dict = {
                "role": "bot",
                "message_id": real_id,
                "sender_id": bot_sender_id,
                "sender_name": bot_sender_name,
                "sender_role": "",
                "timestamp": now_ts,
                "content": text,
                "content_type": content_type,
                "content_segments": content_segments,
            }
            if delivery_state:
                entry["delivery_state"] = delivery_state
            if delivery_error:
                entry["delivery_error"] = delivery_error
            if reply_id:
                entry["reply_to"] = str(reply_id)
            session.add_to_context(entry)

            # 持久化（fire-and-forget，不阻塞发送循环）
            asyncio.run_coroutine_threadsafe(
                save_chat_message(conversation_id, entry), loop
            )
            if delivery_state == "pending":
                asyncio.run_coroutine_threadsafe(
                    _reconcile_pending_send(
                        session=session,
                        conversation_id=conversation_id,
                        internal_message_id=real_id,
                        qq_adapter_client=qq_adapter_client,
                        conv_type=conv_type,
                        group_id=group_id,
                        user_id=user_id,
                        temp_source_group_id=temp_source_group_id,
                        bot_sender_id=str(bot_sender_id),
                        bot_sender_name=str(bot_sender_name),
                        expected_text=text,
                        reply_id=str(reply_id) if reply_id else None,
                        sent_started_at=send_started_at,
                        known_bot_message_ids=set(known_bot_message_ids),
                        pending_error=delivery_error,
                    ),
                    loop,
                )
            known_bot_message_ids.add(real_id)

            # 广播到 debug 前端
            asyncio.run_coroutine_threadsafe(
                broadcast_chat_event({
                    "type": "bot_turn",
                    "conv_id": conversation_id,
                    "conv_name": session.conv_name or conversation_id,
                    "conv_type": conv_type or "unknown",
                    "entries": [entry],
                    "inner_state": get_current_inner_state(),
                }),
                loop,
            )

        new_msgs_count = len([
            m for m in session.context_messages
            if m.get("role") != "bot"
            and m.get("message_id") is not None
            and str(m["message_id"]) not in pre_send_ids
        ])

        result: dict = {
            "to": target,
            "sent_count": sent_count,
            "failed_count": failed_count,
            "total_count": len(send_messages),
            "interrupted": False,
            "new_messages_count": new_msgs_count,
        }
        if failed_messages:
            result["failed_messages"] = failed_messages
        if warnings:
            result["warnings"] = warnings
            result["warning"] = warnings[0]
        if failed_count:
            result["error"] = "部分消息发送失败；请查看 failed_messages。"
        if failed_count and sent_count == 0:
            result["error"] = failed_messages[0].get("reason") or "消息发送失败。"
        return result

    return execute
