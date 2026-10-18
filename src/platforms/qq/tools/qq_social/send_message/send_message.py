"""send_message.py — 发送消息工具实现

Handler 运行在 asyncio.to_thread 派生的线程中，
所有 async 操作通过 asyncio.run_coroutine_threadsafe + app_state.main_loop 执行。
"""

import asyncio
import copy
import logging
import re
import time
from typing import Any, Callable, Literal

from pydantic import Field, RootModel

from tools.contract import ToolArgsModel, ToolContract
from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.prompt_signatures import build_prompt_signature
from platforms.qq.image_pipeline import prepare_sticker_segment
from platforms.qq.send_pipeline import (
    SendTarget,
    build_result,
    failure_result,
    load_face_catalog,
    preflight_images,
    record_bot_message,
    send_one,
)

from .prompt import get_description

logger = logging.getLogger("AICQ.tools")


EXTERNALLY_PERCEPTIBLE: bool = True
TOOL_EFFECT: dict[str, str] = {"surface": "qq", "kind": "session_write"}

REQUIRES_CONTEXT: list[str] = [
    "qq_session_provider",
    "qq_client",
    "config",
    "round_inbound_revision",
]

_SEND_MESSAGE_TAIL_LEAK_RE = re.compile(
    r'^(?P<body>.*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?P<key>messages|segments|quote|command|content|user_id|image_ref|resource_ref|sticker_id)"?)\s*:.*)$',
    re.DOTALL,
)


class TextSegment(ToolArgsModel):
    command: Literal["text"]
    content: str = Field(min_length=1)


class AtSegment(ToolArgsModel):
    command: Literal["at"]
    user_id: str = Field(min_length=1)


class ImageSegment(ToolArgsModel):
    command: Literal["image"]
    path: str = Field(
        default="",
        pattern=r"^(?:|/home/agent/[^\r\n]+)$",
        description=(
            "Agent Linux 中已有图片的绝对路径，必须位于 /home/agent 下。"
        ),
    )
    image_ref: str = Field(
        default="",
        description="image_ref 。",
    )
    resource_ref: str = Field(
        default="",
        description=(
            "<browser><images> 中原图候选的 resource_ref。系统会在发送前按需固化原图；"
        ),
    )


class StickerSegment(ToolArgsModel):
    command: Literal["sticker"]
    image_ref: str = Field(min_length=4, description="已收藏或可见聊天图片的 image_ref。")


class FaceSegment(ToolArgsModel):
    command: Literal["face"]
    id: int = Field(description="QQ 原生表情 ID；需先用 qq_social.list_faces 查询。")


class SendMessageSegment(RootModel[TextSegment | AtSegment | ImageSegment | StickerSegment | FaceSegment]):
    pass


class SendMessageItem(ToolArgsModel):
    quote: str = Field(
        default="",
        json_schema_extra={"x-coerce-integer": True},
        description="要引用/回复的目标消息 ID（可选）。",
    )
    segments: list[SendMessageSegment] = Field(
        min_length=1,
        description="该条消息的内容片段。",
    )


class SendMessageArrayArgs(ToolArgsModel):
    messages: list[SendMessageItem] = Field(
        min_length=1,
        description="要发送的消息列表，每个元素作为一条消息独立发送。",
    )
    simulate_typing: bool = Field(
        default=True,
        description=(
            "模拟打字时间，默认 true。"
        ),
    )


SEND_MESSAGE_ARRAY_CONTRACT = ToolContract(
    name="send_message",
    description=get_description(),
    args_model=SendMessageArrayArgs,
)

TOOL_CONTRACT = SEND_MESSAGE_ARRAY_CONTRACT


def get_declaration(session: Any | None = None, config: dict | None = None, **_: Any) -> dict:
    return SEND_MESSAGE_ARRAY_CONTRACT.declaration()


def get_prompt_signature(config: dict | None = None, **_: Any) -> str:
    return build_prompt_signature(get_declaration(config=config))


def _repair_schema_args_for_shape(
    args: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """修复 send_message 的 messages 容器结构性字段错误。"""
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


def make_schema_repairer(
    config: dict | None = None,
) -> Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]:
    def _repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        return _repair_schema_args_for_shape(args)

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


def _extract_message_text(
    segments: list[dict],
    face_descriptions: dict[int, str] | None = None,
) -> tuple[str, list[dict], str]:
    """从 segments 提取纯文本和结构化 content_segments。"""
    from platforms.qq.adapter.segments import _face_plain_text, face_content_segment

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
        elif cmd == "face":
            face_id = seg.get("id")
            des = (face_descriptions or {}).get(face_id)
            face_data = {"id": face_id}
            if des:
                face_data["raw"] = {"faceText": des}
            face = face_content_segment(face_data)
            text_parts.append(_face_plain_text(face))
            content_segments.append(face)
        elif cmd == "sticker":
            image_ref = seg.get("image_ref", "")
            text_parts.append("[动画表情]")
            content_segments.append({"type": "sticker", "image_ref": image_ref})
        elif cmd == "image":
            image_ref = seg.get("image_ref") or seg.get("_local_image_ref", "")
            text_parts.append("[图片]")
            content_segments.append({"type": "image", "image_ref": image_ref})
    text = "".join(text_parts)
    has_sticker = any(s.get("type") == "sticker" for s in content_segments)
    has_image = any(s.get("type") == "image" for s in content_segments)
    has_text = any(s.get("type") == "text" for s in content_segments)
    has_face = any(s.get("type") == "face" for s in content_segments)
    content_type = (
        "sticker" if has_sticker and not has_text
        else "image" if has_image and not has_text
        else "face" if has_face and not has_text
        else "text"
    )
    return text, content_segments, content_type


def _message_has_at_segment(segments: list[dict]) -> bool:
    return any(isinstance(seg, dict) and seg.get("command") == "at" for seg in segments)


def _validate_face_composition(segments: list[dict], super_only_face_ids: set[int]) -> str | None:
    if len(segments) == 1:
        return None
    if any(
        isinstance(seg, dict)
        and seg.get("command") == "face"
        and isinstance(seg.get("id"), int)
        and not isinstance(seg.get("id"), bool)
        and seg.get("id") in super_only_face_ids
        for seg in segments
    ):
        return "超级表情只能单独作为一条消息发送；请把 face 放在独立的 messages 项中。"
    return None


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
        elif cmd == "face":
            face_id = seg.get("id")
            if isinstance(face_id, bool) or not isinstance(face_id, int) or face_id < 0:
                return None, "face segment 需要非负整数 id。", warnings
            has_sendable = True
        elif cmd == "sticker":
            sticker_fields, sticker_error = prepare_sticker_segment(seg, session)
            if sticker_error:
                return None, sticker_error, warnings
            prepared_seg.update(sticker_fields or {})
            has_sendable = True
        elif cmd == "image":
            image_ref = str(seg.get("image_ref", "") or "")
            local_image_base64 = str(seg.get("_local_image_base64", "") or "")
            if not image_ref and not local_image_base64:
                return None, "image segment 缺少可发送的图片来源。", warnings
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


def _snap_chat_window_to_latest_for_send(session: Any) -> bool:
    """Treat sending from a browsed history window as jumping back to latest."""
    try:
        if session.is_browsing_history():
            session.reset_chat_window_view()
            return True
    except AttributeError:
        return False
    return False


def make_handler(
    qq_session_provider: Callable[[], Any | None],
    qq_client: Any,
    config: dict | None = None,
    round_inbound_revision: int | None = None,
) -> Callable:
    qq_session_provider = ensure_session_provider(qq_session_provider)

    def execute(
        messages: list | None = None,
        segments: list | None = None,
        quote: str | None = None,
        simulate_typing: bool = True,
        **kwargs,
    ) -> dict:
        import app_state
        from platforms.qq import adapter as qq_adapter

        session = qq_session_provider()
        if session is None:
            return failure_result("qq:home", NO_CURRENT_SESSION_ERROR, 1)

        confirmed_batch_id = str(kwargs.pop("_confirmed_batch_id", "") or "")
        if confirmed_batch_id:
            from browser.image_confirmation import consume_pending

            pending = consume_pending(session, confirmed_batch_id)
            if pending is None:
                return failure_result(
                    _format_result_target(session),
                    "图片确认批次不存在、已过期或因新消息而失效。",
                    1,
                )
            send_messages = [copy.deepcopy(message) for message in pending.messages]
            simulate_typing = pending.simulate_typing
            message_error = None
        else:
            send_messages, message_error = _coerce_execute_messages(messages, segments, quote)
            if message_error or send_messages is None:
                return failure_result(
                    _format_result_target(session),
                    message_error or "messages must be a non-empty array.",
                    1,
                )

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        target = _format_result_target(session)
        if loop is None or not loop.is_running():
            return failure_result(target, "主事件循环不可用", len(send_messages))

        qq_adapter_available = bool(qq_client and qq_client.connected)

        # 确定发送目标
        conv_type = session.conv_type
        conv_id = session.conv_id
        group_id, user_id, temp_source_group_id, target_error = _resolve_send_target(session)
        target = _format_result_target(session, temp_source_group_id)
        if target_error:
            return failure_result(target, target_error, len(send_messages))

        if not confirmed_batch_id:
            send_messages, image_response = preflight_images(
                send_messages,
                session=session,
                config=config,
                loop=loop,
                target=target,
                round_inbound_revision=round_inbound_revision,
                simulate_typing=simulate_typing,
            )
            if image_response is not None:
                return image_response


        # QQ adapter 不可用时降级运行：仅入库/入上下文，不实际发送。
        offline_mode = not qq_adapter_available
        if offline_mode and conv_type == "temp":
            return failure_result(
                target,
                "QQ adapter 未连接，无法发送临时会话消息",
                len(send_messages),
            )
        if offline_mode and not str(conv_id).replace("_", "").replace("-", "").replace(".", "").isalnum():
            return failure_result(target, "QQ adapter 未连接", len(send_messages))

        conversation_id = session.key or f"qq:{conv_type}:{conv_id}"
        send_target = SendTarget(
            conversation_id=conversation_id,
            conv_type=conv_type,
            conv_id=conv_id,
            group_id=group_id,
            user_id=user_id,
            temp_source_group_id=temp_source_group_id,
            bot_sender_id=session._qq_id or "bot",
            bot_sender_name=session._qq_name or "",
        )

        # Keep the image-visible window for the entire batch before sending snaps UI to live.
        image_session = copy.copy(session)
        image_session.context_messages = copy.deepcopy(getattr(session, "context_messages", []))
        image_session.chat_window_view = dict(getattr(session, "chat_window_view", {}) or {})
        image_session.forward_browser_stack = copy.deepcopy(getattr(session, "forward_browser_stack", []) or [])
        snapped_to_latest = _snap_chat_window_to_latest_for_send(session)

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
        face_descriptions, super_only_face_ids = load_face_catalog(config, send_messages)


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
            validation_error = _validate_face_composition(segments, super_only_face_ids)
            if validation_error:
                prepared_segments, segment_warnings = None, []
            else:
                prepared_segments, validation_error, segment_warnings = _prepare_sendable_segments(
                    segments, image_session,
                )
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
                qq_adapter_segs = qq_adapter.llm_segments_to_qq_adapter(
                    segments,
                    reply_message_id=reply_id,
                    adapter=getattr(qq_client, "adapter", ""),
                )
            except qq_adapter.ImageLoadError:
                logger.warning(
                    "[send_message] 图片加载失败，终止本次发送 conv=%s",
                    conversation_id,
                    exc_info=True,
                )
                return failure_result(
                    target,
                    "图片加载失败，消息未发送。",
                    len(send_messages),
                    sent_count=sent_count,
                    failed_count=failed_count + 1,
                )
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

            text, content_segments, content_type = _extract_message_text(segments, face_descriptions)
            send_started_at = time.time()
            outcome = send_one(
                send_target,
                qq_client=qq_client,
                loop=loop,
                message_index=i,
                qq_adapter_segs=qq_adapter_segs,
                simulate_typing=simulate_typing,
                offline_mode=offline_mode,
            )
            if outcome.kind == "failed":
                failed_count += 1
                failed_messages.append({"index": i, "reason": outcome.delivery_error})
                logger.warning(
                    "[send_message] adapter 同步返回发送失败 conv=%s idx=%d reason=%s",
                    conversation_id,
                    i,
                    outcome.delivery_error,
                )
                continue
            if outcome.kind == "sent":
                sent_count += 1
            elif outcome.kind == "offline":
                failed_count += 1
                failed_messages.append({"index": i, "reason": outcome.delivery_error})
            elif outcome.kind == "pending":
                logger.warning("[send_message] 消息投递状态待确认 conv=%s idx=%d", conversation_id, i)

            record_bot_message(
                send_target,
                session=session,
                loop=loop,
                qq_client=qq_client,
                segments=segments,
                text=text,
                content_segments=content_segments,
                content_type=content_type,
                reply_id=reply_id,
                outcome=outcome,
                qq_adapter_segs=qq_adapter_segs,
                send_started_at=send_started_at,
                known_bot_message_ids=known_bot_message_ids,
            )

        new_msgs_count = len([
            m for m in session.context_messages
            if m.get("role") != "bot"
            and m.get("message_id") is not None
            and str(m["message_id"]) not in pre_send_ids
        ])

        result = build_result(
            target,
            sent_count=sent_count,
            failed_count=failed_count,
            total_count=len(send_messages),
            new_messages_count=new_msgs_count,
        )
        if snapped_to_latest:
            result["chat_window"] = {
                "snapped_to_latest": True,
                "reason": "send_message_from_history",
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
