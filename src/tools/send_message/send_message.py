"""send_message.py — 发送消息工具实现

Handler 运行在 asyncio.to_thread 派生的线程中，
所有 async 操作通过 asyncio.run_coroutine_threadsafe + app_state.main_loop 执行。
"""

import asyncio
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Callable

from tools._async_bridge import run_coroutine_sync

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


DECLARATION: dict = {
    "name": "send_message",
}

REQUIRES_CONTEXT: list[str] = ["session", "napcat_client"]

_SEND_MESSAGE_TAIL_LEAK_RE = re.compile(
    r'^(?P<body>.*?)(?P<tail>(?:\s*[}\]]{2,}\s*,?\s*)+(?:"?(?P<key>motivation|messages|segments|quote|command|params|content)"?)\s*:.*)$',
    re.DOTALL,
)


def _get_segment_schema_variants(conv_type: str | None) -> list[dict]:
    variants = [_TEXT_SEGMENT_SCHEMA, _STICKER_SEGMENT_SCHEMA]
    if conv_type != "private":
        variants.insert(0, _AT_SEGMENT_SCHEMA)
    return variants


def get_declaration(session: Any | None = None, **_: Any) -> dict:
    conv_type = getattr(session, "conv_type", None)
    is_private = conv_type == "private"
    segment_variants = _get_segment_schema_variants(conv_type)
    segments_description = "该条消息的内容片段"
    if is_private:
        segments_description += "（私聊中仅支持 text / sticker，不支持 at）"

    return {
        "name": "send_message",
        "description": DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "motivation": {
                    "type": "string"
                },
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
                                "description": segments_description,
                                "items": {
                                    "oneOf": segment_variants,
                                },
                            },
                        },
                        "required": ["segments"],
                    },
                },
            },
            "required": ["motivation", "messages"],
        },
    }


def _merge_motivation_texts(values: list[str]) -> tuple[str | None, bool]:
    """合并多个 motivation；纯复读时保留一个，否则按顺序拼接。"""
    unique_values: list[str] = []
    seen_markers: set[str] = set()

    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        marker = " ".join(stripped.split())
        if not marker or marker in seen_markers:
            continue
        seen_markers.add(marker)
        unique_values.append(stripped)

    if not unique_values:
        return None, False
    if len(unique_values) == 1:
        return unique_values[0], len(values) > 1
    return "\n\n".join(unique_values), True


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

    collected_motivations: list[str] = []
    root_motivation = repaired_args.get("motivation")
    if isinstance(root_motivation, str) and root_motivation.strip():
        collected_motivations.append(root_motivation)

    rewritten_messages = messages
    hoisted_fields: list[str] = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict) or "motivation" not in item:
            continue
        if rewritten_messages is messages:
            rewritten_messages = list(messages)
        updated_item = dict(item)
        nested_motivation = updated_item.pop("motivation", None)
        rewritten_messages[index] = updated_item
        hoisted_fields.append(f"messages[{index}].motivation")
        if isinstance(nested_motivation, str) and nested_motivation.strip():
            collected_motivations.append(nested_motivation)

    if hoisted_fields:
        if repaired_args is args:
            repaired_args = dict(args)
        repaired_args["messages"] = rewritten_messages
        merged_motivation, _changed = _merge_motivation_texts(collected_motivations)
        if merged_motivation is not None:
            repaired_args["motivation"] = merged_motivation
        repair_notes.append(f"hoisted {', '.join(hoisted_fields)} -> motivation")

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
    text = "".join(text_parts)
    has_sticker = any(s.get("type") == "sticker" for s in content_segments)
    has_text = any(s.get("type") == "text" for s in content_segments)
    content_type = "sticker" if has_sticker and not has_text else "text"
    return text, content_segments, content_type


def _is_plan_msg_sticker_only(msg: dict) -> bool:
    """判断计划消息是否为纯动画表情（无文字）。"""
    segments = msg.get("segments", [])
    if not segments:
        return False
    return all(seg.get("command") == "sticker" for seg in segments)


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


def make_handler(session: Any, napcat_client: Any) -> Callable:
    def execute(motivation: str, messages: list, **kwargs) -> dict:
        import app_state
        from napcat import llm_segments_to_napcat
        from database import save_chat_message
        from web.debug_server import broadcast_chat_event

        loop: asyncio.AbstractEventLoop | None = getattr(app_state, "main_loop", None)
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用", "sent_count": 0, "total_count": len(messages), "interrupted": False}

        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接", "sent_count": 0, "total_count": len(messages), "interrupted": False}

        # 确定发送目标
        conv_type = session.conv_type
        conv_id = session.conv_id
        try:
            group_id = int(conv_id) if conv_type == "group" else None
            user_id = int(conv_id) if conv_type == "private" else None
        except (ValueError, TypeError):
            return {"error": f"会话 ID 无效: {conv_id}", "sent_count": 0, "total_count": len(messages), "interrupted": False}

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
        interrupted: bool = False
        interrupt_reason: str = ""
        broadcast_entries: list[dict] = []

        for i, msg in enumerate(messages):
            segments = msg.get("segments", [])
            reply_id = msg.get("quote") or None
            napcat_segs = llm_segments_to_napcat(segments, reply_message_id=reply_id)
            if not napcat_segs:
                continue

            # 发送消息（异步→同步）
            try:
                send_result = run_coroutine_sync(
                    napcat_client.send_message(
                        group_id=group_id,
                        user_id=user_id,
                        message=napcat_segs,
                        llm_elapsed=0.0,
                    ),
                    loop,
                    timeout=30,
                )
            except Exception as e:
                logger.warning("[send_message] 发送第 %d 条消息失败: %s", i + 1, e)
                send_result = None

            now_ts = datetime.now(app_state.TIMEZONE).isoformat()

            if send_result and send_result.get("message_id") is not None:
                real_id = str(send_result["message_id"])
                content_ok = True
            else:
                real_id = f"failed_{uuid.uuid4().hex[:8]}"
                content_ok = False
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
            session.add_to_context(entry)
            broadcast_entries.append(entry)
            sent_ids.add(real_id)
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
                    "inner_state": {},
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
                                    motivation=motivation,
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
            "sent_count": sent_count,
            "total_count": len(messages),
            "interrupted": interrupted,
            "new_messages_count": new_msgs_count,
        }
        if interrupted:
            result["interrupt_reason"] = interrupt_reason
        return result

    return execute
