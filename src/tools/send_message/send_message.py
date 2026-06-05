"""send_message.py — 发送消息工具实现

Handler 运行在 asyncio.to_thread 派生的线程中，
所有 async 操作通过 asyncio.run_coroutine_threadsafe + app_state.main_loop 执行。
"""

import asyncio
import base64
import copy
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync
from qq_adapter.conversation import format_adapter_error

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools")

_AT_SEGMENT_SCHEMA: dict = {
    "type": "object",
    "description": "@某人，params 需含 user_id。",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["at"],
        },
        "params": {
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
            },
            "required": ["user_id"],
        },
    },
    "required": ["command", "params"],
}

_TEXT_SEGMENT_SCHEMA: dict = {
    "type": "object",
    "description": "文字，params 需含 content，建议控制单条消息长度。",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["text"],
        },
        "params": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
            },
            "required": ["content"],
        },
    },
    "required": ["command", "params"],
}

_STICKER_SEGMENT_SCHEMA: dict = {
    "type": "object",
    "description": "表情包，params 需含 sticker_id，可通过 list_stickers 工具查询。",
    "properties": {
        "command": {
            "type": "string",
            "enum": ["sticker"],
        },
        "params": {
            "type": "object",
            "properties": {
                "sticker_id": {"type": "string"},
            },
            "required": ["sticker_id"],
        },
    },
    "required": ["command", "params"],
}

_IMAGE_SEGMENT_SCHEMA: dict = {
    "type": "object",
    "description": (
        "发送一张图片，params 需含 url（图片 HTTP/HTTPS 直链）"
        "或 image_ref（browser_control 缓存图片 ref）。"
    ),
    "properties": {
        "command": {
            "type": "string",
            "enum": ["image"],
        },
        "params": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "图片的 HTTP 或 HTTPS 直链地址。",
                },
                "image_ref": {
                    "type": "string",
                    "description": "browser_control 返回的缓存图片 ref，例如 brimg_xxx。",
                },
            },
            "anyOf": [
                {"required": ["url"]},
                {"required": ["image_ref"]},
            ],
        },
    },
    "required": ["command", "params"],
}


DECLARATION: dict = {
    "name": "send_message",
}

REQUIRES_CONTEXT: list[str] = ["session", "qq_adapter_client"]

_SEND_MESSAGE_TAIL_LEAK_RE = re.compile(
    r'^(?P<body>.*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?P<key>messages|segments|quote|command|params|content)"?)\s*:.*)$',
    re.DOTALL,
)

_STICKER_REF_FALLBACK_WARNING = (
    'The provided "sticker_id" was invalid; however, the system still sent a '
    "sticker based on a hash match—serving as a fallback mechanism. Whenever "
    'possible, please use "list_stickers" to check your sticker collection '
    "first before initiating a send action. If the sticker sent by the system "
    "in this instance does not meet your expectations, you may retract it."
)


def get_declaration(session: Any | None = None, **_: Any) -> dict:
    return {
        "name": "send_message",
        "description": DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "要发送的消息列表，每个元素独立发送。",
                    "items": {
                        "type": "object",
                        "description": "单条消息的结构",
                        "properties": {
                            "quote": {
                                "type": "string",
                                "x-coerce-integer": True,
                                "description": "要引用/回复的目标消息 ID（可选）。",
                            },
                            "segments": {
                                "type": "array",
                                "description": "该条消息的内容片段。",
                                "items": {
                                    "oneOf": [
                                        _AT_SEGMENT_SCHEMA,
                                        _TEXT_SEGMENT_SCHEMA,
                                        _STICKER_SEGMENT_SCHEMA,
                                        _IMAGE_SEGMENT_SCHEMA,
                                    ],
                                },
                            },
                        },
                        "required": ["segments"],
                    },
                },
            },
            "required": ["messages"],
        },
    }


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """修复 send_message 的结构性字段错误。"""
    repair_notes: list[str] = []
    messages = args.get("messages")
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
                and "params" not in segment
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

    repaired_args = args
    if normalized_messages != messages:
        repaired_args = dict(args)
        repaired_args["messages"] = normalized_messages
        messages = normalized_messages

    rewritten_messages = messages
    removed_fields: list[str] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or "motivation" not in item:
            continue
        if rewritten_messages is messages:
            rewritten_messages = list(messages)
        updated_item = dict(item)
        updated_item.pop("motivation", None)
        rewritten_messages[index] = updated_item
        removed_fields.append(f"messages[{index}].motivation")

    if removed_fields:
        if repaired_args is args:
            repaired_args = dict(args)
        repaired_args["messages"] = rewritten_messages
        repair_notes.append(f"removed legacy {', '.join(removed_fields)}")

    return repaired_args, repair_notes


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
    if not isinstance(messages, list):
        return repaired_args, changes, None

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
        params = seg.get("params", {})
        if cmd == "text":
            t = params.get("content", "")
            text_parts.append(t)
            if t:
                content_segments.append({"type": "text", "text": t})
        elif cmd == "at":
            uid = str(params.get("user_id", ""))
            text_parts.append(f"@{uid}")
            content_segments.append({"type": "mention", "uid": uid, "display": f"@{uid}"})
        elif cmd == "sticker":
            sticker_id = params.get("sticker_id", "")
            text_parts.append("[动画表情]")
            content_segments.append({"type": "sticker", "sticker_id": sticker_id})
        elif cmd == "image":
            url = params.get("url", "")
            image_ref = params.get("image_ref", "")
            text_parts.append("[图片]")
            content_segments.append({"type": "image", "url": url, "image_ref": image_ref})
    text = "".join(text_parts)
    has_sticker = any(s.get("type") == "sticker" for s in content_segments)
    has_image = any(s.get("type") == "image" for s in content_segments)
    has_text = any(s.get("type") == "text" for s in content_segments)
    content_type = "sticker" if has_sticker and not has_text else "image" if has_image and not has_text else "text"
    return text, content_segments, content_type


def _is_plan_msg_sticker_only(msg: dict) -> bool:
    """判断计划消息是否为纯动画表情（无文字）。"""
    segments = msg.get("segments", [])
    if not segments:
        return False
    return all(seg.get("command") == "sticker" for seg in segments)


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
    return conv_id or "web_user"


def _resolve_send_target(session: Any) -> tuple[int | None, int | None, int | None, str | None]:
    conv_type = getattr(session, "conv_type", "")
    conv_id = getattr(session, "conv_id", "")
    if conv_type == "" or str(conv_id) == "web_user":
        return None, None, None, None
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
        params = seg.get("params", {})
        if not isinstance(params, dict):
            params = {}
        prepared_params = prepared_seg.setdefault("params", {})
        if not isinstance(prepared_params, dict):
            prepared_params = {}
            prepared_seg["params"] = prepared_params

        if cmd == "text" and str(params.get("content", "") or ""):
            has_sendable = True
        elif cmd == "at" and str(params.get("user_id", "") or ""):
            has_sendable = True
        elif cmd == "sticker":
            sticker_id = str(params.get("sticker_id", "") or "")
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
                        f"表情包 sticker_id \"{sticker_id}\" 不存在，未发送。"
                        "发送表情包前请先调用 list_stickers 获取自己的表情包 ID；"
                        "聊天记录中的动画表情 ref 只有在当前上下文仍能找到图片时才可作为兜底发送。"
                    ), warnings
                raw_bytes, mime = fallback
                prepared_params["_fallback_base64"] = base64.b64encode(raw_bytes).decode("ascii")
                prepared_params["_fallback_mime"] = mime
                prepared_params["_fallback_ref"] = sticker_id
                warnings.append(_STICKER_REF_FALLBACK_WARNING)
            has_sendable = True
        elif cmd == "image":
            url = str(params.get("url", "") or "")
            image_ref = str(params.get("image_ref", "") or "")
            if not url and not image_ref:
                return None, "image segment 缺少 url 或 image_ref。", warnings
            has_sendable = True
        prepared_segments.append(prepared_seg)

    if not has_sendable:
        return None, "消息没有可发送的内容，未发送。", warnings
    return prepared_segments, None, warnings


def _split_consecutive_texts(segments: list[dict]) -> list[list[dict]]:
    """将含连续 text segments 的消息拆分为多组。

    规则：每当遇到第二个连续 text，在第一个 text 之后切割。
    例：[at, text, text] → [[at, text], [text]]
        [text, text, text] → [[text], [text], [text]]
    """
    if not segments:
        return []
    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_was_text = False
    for seg in segments:
        is_text = seg.get("command") == "text"
        if is_text and prev_was_text:
            groups.append(current)
            current = [seg]
        else:
            current.append(seg)
        prev_was_text = is_text
    if current:
        groups.append(current)
    return groups


def _expand_messages(messages: list[dict]) -> list[dict]:
    """将 messages 列表中每条消息的连续 text segments 拆分为多条独立消息。"""
    result: list[dict] = []
    for msg in messages:
        segs = msg.get("segments", [])
        groups = _split_consecutive_texts(segs)
        if len(groups) <= 1:
            result.append(msg)
        else:
            # 第一组继承原消息的 quote 等字段
            result.append({**msg, "segments": groups[0]})
            for g in groups[1:]:
                # 后续组不继承 quote，避免重复引用同一条消息
                result.append({"segments": g})
    return result


def make_handler(session: Any, qq_adapter_client: Any) -> Callable:
    def execute(messages: list, **kwargs) -> dict:
        import app_state
        from qq_adapter import llm_segments_to_qq_adapter, ImageDownloadError
        from database import save_chat_message
        from llm.core.round_context import get_current_inner_state
        from web.debug_server import broadcast_chat_event

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        target = _format_result_target(session)
        if loop is None or not loop.is_running():
            return {"to": target, "error": "主事件循环不可用", "sent_count": 0, "total_count": len(messages), "interrupted": False}

        qq_adapter_available = bool(qq_adapter_client and qq_adapter_client.connected)

        # 确定发送目标
        conv_type = session.conv_type
        conv_id = session.conv_id
        group_id, user_id, temp_source_group_id, target_error = _resolve_send_target(session)
        target = _format_result_target(session, temp_source_group_id)
        if target_error:
            return {"to": target, "error": target_error, "sent_count": 0, "total_count": len(messages), "interrupted": False}

        # QQ adapter 不可用时只允许 web 会话降级运行（仅入库/入上下文，不实际发送）
        is_web_session = conv_type == "" or str(conv_id) == "web_user"
        web_mode = is_web_session or not qq_adapter_available
        if web_mode and conv_type == "temp":
            return {"to": target, "error": "QQ adapter 未连接，无法发送临时会话消息", "sent_count": 0, "total_count": len(messages), "interrupted": False}
        if web_mode and not str(conv_id).replace("_", "").replace("-", "").replace(".", "").isalnum():
            return {"to": target, "error": "QQ adapter 未连接", "sent_count": 0, "total_count": len(messages), "interrupted": False}

        conversation_id = f"{conv_type}_{conv_id}"
        bot_sender_id = session._qq_id or "bot"
        bot_sender_name = session._qq_name or ""

        # IS 相关：发送前快照现有非 bot 消息 ID
        pre_send_ids: set[str] = {
            str(m["message_id"])
            for m in session.context_messages
            if m.get("message_id") is not None and m.get("role") != "bot"
        }
        sent_ids: set[str] = set()
        sent_count: int = 0
        failed_count: int = 0
        failed_messages: list[dict] = []
        warnings: list[str] = []
        interrupted: bool = False
        interrupt_reason: str = ""
        broadcast_entries: list[dict] = []

        for i, msg in enumerate(messages):
            segments = msg.get("segments", [])
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
            except ImageDownloadError as img_err:
                logger.warning("[send_message] 图片下载失败，终止本次发送 conv=%s — %s", conversation_id, img_err)
                return {
                    "error": str(img_err),
                    "sent_count": sent_count,
                    "total_count": len(messages),
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

            # 发送消息（异步→同步）
            if web_mode:
                send_result = None  # web 模式：跳过实际发送
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

            if web_mode:
                real_id = f"web_{uuid.uuid4().hex[:8]}"
                content_ok = True
            elif send_result and send_result.get("message_id") is not None:
                real_id = str(send_result["message_id"])
                content_ok = True
            else:
                real_id = f"failed_{uuid.uuid4().hex[:8]}"
                content_ok = False
                failed_count += 1
                failed_messages.append({
                    "index": i,
                    "reason": format_adapter_error(
                        getattr(qq_adapter_client, "last_api_error", None),
                        "QQ adapter send_msg failed or returned no message_id",
                    ),
                })
                logger.warning("[send_message] 消息发送失败 conv=%s idx=%d", conversation_id, i)

            text, content_segments, content_type = _extract_message_text(segments)
            if not content_ok:
                content_type = "send_failed"

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
            if reply_id:
                entry["reply_to"] = str(reply_id)
            session.add_to_context(entry)
            broadcast_entries.append(entry)
            sent_ids.add(real_id)
            if content_ok:
                sent_count += 1

            # 持久化（fire-and-forget，不阻塞发送循环）
            asyncio.run_coroutine_threadsafe(
                save_chat_message(conversation_id, entry), loop
            )

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

            # IS 检测：只在有剩余消息 且 本次调用含 ≥2 条消息 时触发
            if not interrupted and len(messages) >= 2 and i + 1 < len(messages):
                new_user_msgs = [
                    m for m in session.context_messages
                    if m.get("role") != "bot"
                    and m.get("message_id") is not None
                    and str(m["message_id"]) not in pre_send_ids
                ]
                if new_user_msgs:
                    trigger_entry = new_user_msgs[0]
                    remaining = messages[i + 1:]

                    # 过滤：触发消息是纯表情 → 跳过 IS
                    if trigger_entry.get("content_type") == "sticker":
                        pass
                    # 过滤：剩余唯一一条纯表情 → 跳过 IS
                    elif len(remaining) == 1 and _is_plan_msg_sticker_only(remaining[0]):
                        pass
                    else:
                        try:
                            from llm.IS import check_interruption
                            should_stop, reason = run_coroutine_sync(
                                check_interruption(
                                    session=session,
                                    cognition=str(get_current_inner_state().get("cognition") or ""),
                                    all_messages=messages,
                                    sent_count=sent_count,
                                    trigger_entry=trigger_entry,
                                    remaining_plan_msgs=remaining,
                                    sent_this_round_ids=sent_ids,
                                ),
                                loop,
                                timeout=60,
                            )
                            if should_stop:
                                logger.info(
                                    "[IS] 中断发送 sent=%d/%d reason=%s conv=%s",
                                    sent_count, len(messages), reason, conversation_id,
                                )
                                interrupted = True
                                interrupt_reason = reason
                                break
                        except Exception as e:
                            logger.warning("[send_message] IS 检测异常，默认继续: %s", e)

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
            "total_count": len(messages),
            "interrupted": interrupted,
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
        if interrupted:
            result["interrupt_reason"] = interrupt_reason
        return result

    return execute
