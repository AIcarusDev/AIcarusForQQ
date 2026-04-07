"""send_message.py — 发送消息工具实现

Handler 运行在 asyncio.to_thread 派生的线程中，
所有 async 操作通过 asyncio.run_coroutine_threadsafe + app_state.main_loop 执行。
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable
from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
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
                            "description": "要引用/回复的目标消息 ID（可选）。",
                        },
                        "segments": {
                            "type": "array",
                            "description": "该条消息的内容片段",
                            "items": {
                                "type": "object",
                                "description": (
                                    "消息片段。command 可选值："
                                    "at(@某人，params 需含 user_id)、"
                                    "text(文字，params 需含 content，建议控制单条消息长度)、"
                                    "sticker(表情包，params 需含 sticker_id，"
                                    "可通过 list_stickers 工具查询)"
                                ),
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "enum": ["at", "text", "sticker"],
                                    },
                                    "params": {"type": "object"},
                                },
                                "required": ["command", "params"],
                            },
                        },
                    },
                    "required": ["segments"],
                },
            },
        },
        "required": ["motivation", "messages"],
    },
    "max_calls_per_response": 10,
}

REQUIRES_CONTEXT: list[str] = ["session", "napcat_client"]

# result 不写入 previous_tools_used 摘要（消息本身已在上下文中）
RESULT_MAX_CHARS: int = -1


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
                send_result = asyncio.run_coroutine_threadsafe(
                    napcat_client.send_message(
                        group_id=group_id,
                        user_id=user_id,
                        message=napcat_segs,
                        llm_elapsed=0.0,
                    ),
                    loop,
                ).result(timeout=30)
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
                            should_stop, reason = asyncio.run_coroutine_threadsafe(
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
                            ).result(timeout=60)
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
