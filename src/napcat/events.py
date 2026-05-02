"""napcat/events.py — NapCat 事件解析

将 NapCat 原始消息事件转为 core 内部格式：
  - napcat_event_to_context  — 事件 → context entry dict
  - get_conversation_id      — 事件 → 会话 ID 字符串
  - should_respond           — 判断是否应当回复该消息
"""

import asyncio
import base64
import logging
import time
import uuid
from datetime import datetime
from typing import Any

import httpx

from .segments import (
    napcat_segments_to_text,
    build_content_segments,
    get_reply_message_id,
    _determine_content_type,
)
from database import get_display_name

logger = logging.getLogger("AICQ.napcat.events")


# ── 图片下载工具 ──────────────────────────────────────────────────────────────

_MAX_DOWNLOAD_RETRIES = 2
_EXPIRED_SENTINEL = ("__EXPIRED__", "")

# 限制并发下载数，避免大量同时连接耗尽 Windows 套接字缓冲区（WinError 10055）
_DOWNLOAD_SEMAPHORE = asyncio.Semaphore(4)

# 复用连接池；verify=False 对应原来的 CERT_NONE
_HTTP_CLIENT = httpx.AsyncClient(
    verify=False,
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=15.0,
    limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
)

# 短时 URL 缓存：避免同一 URL 在短时间内被重复下载（转发风暴、消息重试等场景）
# 结构：url -> (result, expire_ts)；result 与 _fetch_image_b64 返回值类型相同
_URL_CACHE: dict[str, tuple[tuple | None, float]] = {}
_URL_CACHE_TTL = 300.0  # 5 分钟，QQ CDN URL 通常在此范围内有效


def _url_cache_get(url: str) -> tuple[tuple | None, bool]:
    """查询 URL 缓存。返回 (result, hit)；hit=False 表示未命中或已过期。"""
    entry = _URL_CACHE.get(url)
    if entry is None:
        return None, False
    result, expire_ts = entry
    if time.monotonic() > expire_ts:
        del _URL_CACHE[url]
        return None, False
    return result, True


def _url_cache_set(url: str, result: tuple | None) -> None:
    """写入 URL 缓存，同时淘汰已过期条目（保持内存占用可控）。"""
    now = time.monotonic()
    # 惰性清理：仅在缓存条目超过 200 时全量扫描，日常消息量下不会触发
    if len(_URL_CACHE) > 200:
        expired_keys = [k for k, (_, exp) in _URL_CACHE.items() if now > exp]
        for k in expired_keys:
            del _URL_CACHE[k]
    _URL_CACHE[url] = (result, now + _URL_CACHE_TTL)


async def _fetch_image_b64(url: str) -> tuple[str, str] | None:
    """从 URL 下载图片，返回 (base64字符串, mime_type)，失败返回 None。

    特殊情况：HTTP 404（CDN URL 已过期）时返回 _EXPIRED_SENTINEL，不重试。
    使用共享 AsyncClient 复用连接，并通过信号量限制并发，
    防止 Windows 套接字缓冲区耗尽（WinError 10055）。
    同一 URL 在 5 分钟内命中缓存时直接返回，避免重复下载。
    """
    cached, hit = _url_cache_get(url)
    if hit:
        logger.debug("图片 URL 缓存命中，跳过下载 (url=%s...)", url[:60])
        return cached

    last_err = None
    for attempt in range(_MAX_DOWNLOAD_RETRIES + 1):
        try:
            async with _DOWNLOAD_SEMAPHORE:
                resp = await _HTTP_CLIENT.get(url)
            if resp.status_code == 404:
                logger.warning("图片 URL 已过期 (url=%s...): %s", url[:60], resp.status_code)
                _url_cache_set(url, _EXPIRED_SENTINEL)
                return _EXPIRED_SENTINEL
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/jpeg")
            mime = content_type.split(";")[0].strip() or "image/jpeg"
            result = base64.b64encode(resp.content).decode("ascii"), mime
            _url_cache_set(url, result)
            return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("图片 URL 已过期 (url=%s...): %s", url[:60], e)
                _url_cache_set(url, _EXPIRED_SENTINEL)
                return _EXPIRED_SENTINEL
            last_err = e
            if attempt < _MAX_DOWNLOAD_RETRIES:
                await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as e:
            last_err = e
            if attempt < _MAX_DOWNLOAD_RETRIES:
                await asyncio.sleep(0.5 * (attempt + 1))

    logger.warning("图片下载失败 (url=%s...): %s", url[:60], last_err)
    # 失败结果不缓存，允许下次消息重新尝试
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
                _seg["display"] = f"@{_display_name_cache[_cache_key]}"

    sender_role = sender.get("role", "") if msg_type == "group" else ""

    # 收集图片引用信息（不下载），以 ref 为键建立 dict
    image_refs = [
        (seg["ref"], "动画表情" if seg["type"] == "sticker" else "图片")
        for seg in content_segments
        if seg.get("type") in ("image", "sticker") and "ref" in seg
    ]
    image_tasks = []
    for seg in message_segs:
        if seg.get("type") not in ("image", "mface"):
            continue
        data = seg.get("data", {})
        if raw_b64 := data.get("base64", ""):
            image_tasks.append(("b64", raw_b64, "image/jpeg"))
        elif url := data.get("url", ""):
            image_tasks.append(("url", url, ""))

    # 立即可用的图片（base64直传）和需要下载的图片（URL）分开处理
    # URL 类先以 {"pending": True} 预占位，避免下载未完成时模型把图片误认为已加载，
    # 同时给 xml 渲染端一个明确的「加载中」状态。
    images: dict[str, dict] = {}
    pending_downloads: list[tuple[str, str, str]] = []  # (ref, url, label)
    for (ref, label), (kind, value, preset_mime) in zip(image_refs, image_tasks):
        if kind == "b64":
            images[ref] = {"base64": value, "mime": preset_mime, "label": label}
        else:
            images[ref] = {"pending": True, "label": label}
            pending_downloads.append((ref, value, label))

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
    if pending_downloads:
        entry["_pending_images"] = pending_downloads
    return entry


async def download_pending_images(entry: dict) -> bool:
    """下载 entry 中待获取的图片，原地更新 images 字段。

    返回 True 表示有新图片被成功下载。
    调用前 entry 可能已经通过 add_to_context 加入会话上下文，
    由于 add_to_context 存引用，此处的修改会自动对上下文生效。
    """
    pending = entry.pop("_pending_images", None)
    if not pending:
        return False

    images = entry.get("images") or {}
    downloaded_any = False
    for ref, url, label in pending:
        result = await _fetch_image_b64(url)
        if result is _EXPIRED_SENTINEL:
            images[ref] = {"expired": True, "label": label}
            logger.warning("图片已过期，已标记 ref=%s", ref)
        elif result:
            b64, mime = result
            images[ref] = {"base64": b64, "mime": mime, "label": label}
            downloaded_any = True
        else:
            images[ref] = {"failed": True, "label": label}
            logger.warning("图片下载失败，已标记 ref=%s", ref)
    if images:
        entry["images"] = images
    return downloaded_any


async def expand_forward_previews(entry: dict, client) -> None:
    """展开 content_segments 中待解析的合并转发段，填入预览数据。

    原地修改 entry（引用语义，自动对已入上下文的条目生效）。
    """
    segments = entry.get("content_segments", [])
    _PREVIEW_TEXT_MAX = 20
    for seg in segments:
        if seg.get("type") != "forward" or not seg.pop("_needs_expand", False):
            continue

        fwd_id = seg.get("forward_id", "")
        if not fwd_id:
            seg.update({"title": "合并转发", "preview": [], "total": 0})
            continue

        result = await client.send_api("get_forward_msg", {"id": fwd_id}, timeout=10.0)
        if not result:
            seg.update({"title": "合并转发", "preview": [], "total": 0})
            logger.warning("get_forward_msg 失败: forward_id=%s", fwd_id)
            continue

        # ⚠️ NapCat 已知 Bug（截至 2026-03）：
        #   对于「私聊合并转发」，get_forward_msg 返回的 messages 列表中
        #   会丢失所有 sender.user_id == self_id（即 bot 自身）发送的消息，
        #   且返回的 total 也是过滤后的数量，不反映原始消息总数。
        #   例：bot 与用户的 5 条对话合并转发后，bot 发出的 3 条全部缺失，
        #   仅返回用户发出的 2 条，total=2。
        #   群聊合并转发不受此影响，消息完整。
        #   上游 issue 建议：https://github.com/NapNeko/NapCatQQ （待提交）
        #   此处代码无法修复，数据在 NapCat 层已丢失，只能原样展示残缺预览。
        messages = result.get("messages", [])
        total = len(messages)

        # 判断群/私聊，构建 title
        first = messages[0] if messages else {}
        if first.get("message_type") == "group":
            title = "群聊的聊天记录"
        else:
            # 私聊：取前两个不同 uid 的 nickname 拼 title
            seen_uids: list[str] = []
            seen_names: list[str] = []
            for node in messages:
                sender = node.get("sender", {})
                uid = str(sender.get("user_id", ""))
                name = sender.get("nickname", "") or uid
                if uid and uid not in seen_uids:
                    seen_uids.append(uid)
                    seen_names.append(name)
                if len(seen_uids) >= 2:
                    break
            if len(seen_names) >= 2:
                title = f"{seen_names[0]} 与 {seen_names[1]} 的聊天记录"
            elif seen_names:
                title = f"{seen_names[0]} 的聊天记录"
            else:
                title = "聊天记录"

        # 构建前4条预览
        preview: list[dict] = []
        for node in messages[:4]:
            sender = node.get("sender", {})
            nickname = sender.get("card") or sender.get("nickname") or str(sender.get("user_id", ""))
            sub_msgs = node.get("message", [])
            item_type = "text"
            item_text = ""
            for sub in sub_msgs:
                st = sub.get("type", "")
                sd = sub.get("data", {})
                if st == "text":
                    raw = sd.get("text", "").replace("\n", " ").strip()
                    item_text = (
                        f"{raw[:_PREVIEW_TEXT_MAX]}..."
                        if len(raw) > _PREVIEW_TEXT_MAX
                        else raw
                    )
                    item_type = "text"
                    break
                elif st == "image":
                    item_type = "sticker" if sd.get("sub_type", 0) == 1 else "image"
                    break
                elif st == "mface":
                    item_type = "sticker"
                    break
                elif st == "file":
                    item_type = "file"
                    fn = sd.get("name", "")
                    item_text = (
                        f"{fn[:_PREVIEW_TEXT_MAX]}..."
                        if len(fn) > _PREVIEW_TEXT_MAX
                        else fn
                    )
                    break
                else:
                    item_type = st or "unknown"
            preview.append({"sender": nickname, "content_type": item_type, "content_text": item_text})

        seg.update({"title": title, "preview": preview, "total": total})
        logger.debug("合并转发展开完成: forward_id=%s total=%d", fwd_id, total)


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

    私聊：始终不主动触发回复，仅静默计入未读（bot 在其他会话中能通过 unread_info 感知到）。
    群聊：被 @、消息中提到 bot_name 时回复。
    """
    msg_type = event.get("message_type", "")

    if msg_type == "private":
        return False

    message_segs = event.get("message", [])
    for seg in message_segs:
        if seg.get("type") == "at":
            if str(seg.get("data", {}).get("qq", "")) == str(bot_id):
                return True
        if seg.get("type") == "text" and bot_name:
            if bot_name in seg.get("data", {}).get("text", ""):
                return True

    return False
