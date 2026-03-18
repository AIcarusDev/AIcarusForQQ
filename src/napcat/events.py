"""napcat/events.py — NapCat 事件解析

将 NapCat 原始消息事件转为 core 内部格式：
  - napcat_event_to_context  — 事件 → context entry dict
  - get_conversation_id      — 事件 → 会话 ID 字符串
  - should_respond           — 判断是否应当回复该消息
"""

import asyncio
import base64
import logging
import urllib.request
import uuid
from datetime import datetime
from typing import Any

from .segments import (
    napcat_segments_to_text,
    build_content_segments,
    get_reply_message_id,
    _determine_content_type,
)
from database import get_display_name

logger = logging.getLogger("AICQ.napcat")


# ── 图片下载工具 ──────────────────────────────────────────────────────────────

async def _fetch_image_b64(url: str) -> tuple[str, str] | None:
    """从 URL 下载图片，返回 (base64字符串, mime_type)，失败返回 None。"""
    loop = asyncio.get_running_loop()
    try:
        def _download():
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                content_type = resp.headers.get("Content-Type", "image/jpeg")
                mime = content_type.split(";")[0].strip() or "image/jpeg"
                return data, mime
        data, mime = await loop.run_in_executor(None, _download)
        return base64.b64encode(data).decode("ascii"), mime
    except Exception as e:
        logger.warning("图片下载失败 (url=%s...): %s", url[:60], e)
        return None


# ── NapCat 事件 → core 上下文条目 ────────────────────────────────────────────

async def napcat_event_to_context(
    event: dict,
    bot_id: str | None = None,
    bot_display_name: str = "",
    timezone: Any = None,
) -> dict | None:
    """将 NapCat 消息事件转为 core 的上下文条目格式。

    返回字段:
      role, message_id, sender_id, sender_name, timestamp, content,
      sender_role   — 群聊: "owner"/"admin"/"member"；私聊: ""
      content_type  — "text"/"image"/"file"
      content_segments — 结构化内容段列表（供 xml_builder 渲染富文本）
      reply_to      — 被回复消息的 ID（可选）
      images        — {ref: {"base64": str, "mime": str, "label": str}, ...}（可选）
    """
    if event.get("post_type") != "message":
        return None

    sender = event.get("sender", {})
    msg_type = event.get("message_type", "")
    # 群里优先用 card（群昵称），没有就用 nickname
    sender_name = (
        sender.get("card") or sender.get("nickname") or str(sender.get("user_id", "未知"))
    )
    message_segs = event.get("message", [])
    text = napcat_segments_to_text(message_segs, bot_id=bot_id, bot_display_name=bot_display_name)
    if not text:
        return None

    from zoneinfo import ZoneInfo

    tz = timezone or ZoneInfo("Asia/Shanghai")
    timestamp = datetime.fromtimestamp(event.get("time", 0), tz=tz).isoformat()

    content_segments = build_content_segments(message_segs, bot_id=bot_id, bot_display_name=bot_display_name)
    reply_to = get_reply_message_id(message_segs)
    content_type = _determine_content_type(message_segs)

    # 对 display 仍为纯 UID 的 mention，从 DB 补全显示名（优先群名片，其次昵称）
    _group_id = str(event.get("group_id", "")) if msg_type == "group" else ""
    _display_name_cache: dict[tuple, str] = {}
    for _seg in content_segments:
        if _seg.get("type") == "mention" and _seg.get("uid") not in ("all", "self"):
            _uid = _seg["uid"]
            if _seg.get("display") == f"@{_uid}":
                _cache_key = (_uid, _group_id or None)
                if _cache_key not in _display_name_cache:
                    _display_name_cache[_cache_key] = await get_display_name("qq", _uid, _group_id or None)
                _seg["display"] = "@" + _display_name_cache[_cache_key]

    sender_role = sender.get("role", "") if msg_type == "group" else ""

    # 下载图片，以 ref 为键建立 dict
    image_refs = [
        (seg["ref"], "动画表情" if seg["type"] == "sticker" else "图片")
        for seg in content_segments
        if seg.get("type") in ("image", "sticker") and "ref" in seg
    ]
    image_tasks = []
    for seg in message_segs:
        if seg.get("type") != "image":
            continue
        data = seg.get("data", {})
        raw_b64 = data.get("base64", "")
        if raw_b64:
            image_tasks.append(("b64", raw_b64, "image/jpeg"))
        elif url := data.get("url", ""):
            image_tasks.append(("url", url, ""))

    images: dict[str, dict] = {}
    for (ref, label), (kind, value, preset_mime) in zip(image_refs, image_tasks):
        if kind == "b64":
            images[ref] = {"base64": value, "mime": preset_mime, "label": label}
        else:
            result = await _fetch_image_b64(value)
            if result:
                b64, mime = result
                images[ref] = {"base64": b64, "mime": mime, "label": label}

    entry: dict = {
        "role": "user",
        "message_id": str(event.get("message_id", f"msg_{uuid.uuid4().hex[:8]}")),
        "sender_id": str(sender.get("user_id", "unknown")),
        "sender_name": sender_name,
        "sender_role": sender_role,
        "timestamp": timestamp,
        "content": text,
        "content_type": content_type,
        "content_segments": content_segments,
    }
    if reply_to:
        entry["reply_to"] = reply_to
    if images:
        entry["images"] = images
    return entry


def get_conversation_id(event: dict) -> str:
    """从 NapCat 事件中提取会话 ID。"""
    msg_type = event.get("message_type", "")
    if msg_type == "group":
        return f"group_{event.get('group_id', 'unknown')}"
    elif msg_type == "private":
        return f"private_{event.get('sender', {}).get('user_id', 'unknown')}"
    return "unknown"


def should_respond(event: dict, bot_id: str | None, bot_name: str = "") -> bool:
    """判断是否应该回复这条消息。

    私聊：始终回复
    群聊：被 @、消息中提到 bot_name、或回复了 bot 的消息时回复
    """
    msg_type = event.get("message_type", "")

    if msg_type == "private":
        return True

    message_segs = event.get("message", [])
    for seg in message_segs:
        if seg.get("type") == "at":
            if str(seg.get("data", {}).get("qq", "")) == str(bot_id):
                return True
        if seg.get("type") == "text" and bot_name:
            if bot_name in seg.get("data", {}).get("text", ""):
                return True

    return False
