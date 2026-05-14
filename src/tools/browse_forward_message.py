"""browse_forward_message.py — 打开和浏览合并转发消息。"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from napcat.segments import (
    _determine_content_type,
    build_content_segments,
    get_reply_message_id,
    napcat_segments_to_text,
)
from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

SCOPE: str = "all"
REQUIRES_CONTEXT: list[str] = ["session", "napcat_client"]

_OPEN_ACTIONS = ["open"]
_BROWSING_ACTIONS = ["open", "next_page", "prev_page", "back", "close_all"]


def _declaration_for_state(*, browsing: bool) -> dict:
    if not browsing:
        return {
            "name": "browse_forward_message",
            "description": (
                "打开合并转发浏览窗口。"
                "看到 <content type=\"forward\" openable=\"true\"> 时，"
                "用 action=open 并填写外层 <message id=\"...\"> 的 id 即可打开。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": _OPEN_ACTIONS,
                        "description": "open=打开 id 对应的合并转发。",
                    },
                    "id": {
                        "type": "string",
                        "description": "要打开的外层 <message id=\"...\">，通常是真实 QQ message_id。",
                    },
                    "motivation": {"type": "string"},
                },
                "required": ["action", "motivation"],
            },
        }

    return {
        "name": "browse_forward_message",
        "description": (
            "浏览当前已经打开的合并转发视图。"
            "可以翻页、返回、关闭，也可以继续打开视图内 "
            "<content type=\"forward\" openable=\"true\"> 对应的嵌套合并转发。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _BROWSING_ACTIONS,
                    "description": (
                        "open=打开 id 对应的合并转发；"
                        "next_page/prev_page=在当前合并转发内翻页；"
                        "back=关闭当前层并回到上一层；"
                        "close_all=关闭所有合并转发浏览视图。"
                    ),
                },
                "id": {
                    "type": "string",
                    "description": (
                        "action=open 时必填。填写要打开的 <message id=\"...\">。"
                        "可以是真实 QQ message_id，也可以是 fwd: 开头的虚拟 id。"
                    ),
                },
                "impression": {
                    "type": "string",
                    "description": "操作前你对当前合并转发视图的印象。",
                },
                "motivation": {"type": "string"},
            },
            "required": ["action", "impression", "motivation"],
        },
    }


DECLARATION: dict = _declaration_for_state(browsing=False)


def get_declaration(session: Any | None = None) -> dict:
    return _declaration_for_state(
        browsing=bool(session is not None and session.is_browsing_forward())
    )


def _session_key(session: Any) -> str:
    return f"{session.conv_type}_{session.conv_id}" if getattr(session, "conv_type", "") else ""


def _row_to_entry(row: sqlite3.Row) -> dict:
    entry = {
        "role": row["role"],
        "message_id": row["message_id"],
        "sender_id": row["sender_id"],
        "sender_name": row["sender_name"],
        "sender_role": row["sender_role"],
        "timestamp": row["timestamp"],
        "content": row["content"],
        "content_type": row["content_type"],
        "content_segments": json.loads(row["content_segments"] or "[]"),
    }
    if reply_to := str(row["reply_to"] or ""):
        entry["reply_to"] = reply_to
    images = json.loads(row["images"] or "[]")
    if images:
        entry["images"] = images
    return entry


def _find_real_message(session: Any, message_id: str) -> dict | None:
    for entry in getattr(session, "context_messages", []) or []:
        if str(entry.get("message_id", "")) == message_id:
            return entry

    session_key = _session_key(session)
    if not session_key:
        return None

    from database import DB_PATH

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT role, message_id, sender_id, sender_name, sender_role,
                          timestamp, reply_to, content, content_type, content_segments, images
                   FROM chat_messages
                   WHERE session_key=? AND message_id=?
                   ORDER BY id DESC
                   LIMIT 1""",
                (session_key, message_id),
            ).fetchone()
            return _row_to_entry(row) if row else None
    except Exception:
        logger.exception("[tools] browse_forward_message: 查询消息失败 message_id=%s", message_id)
        return None


def _extract_forward_id(entry: dict) -> str:
    segments = entry.get("content_segments") or []
    forward_ids = [
        str(seg.get("forward_id", ""))
        for seg in segments
        if seg.get("type") == "forward" and seg.get("forward_id")
    ]
    if len(forward_ids) != 1:
        return ""
    return forward_ids[0]


def _extract_forward_content(entry: dict, forward_id: str) -> list[dict] | None:
    for seg in entry.get("content_segments") or []:
        if seg.get("type") != "forward" or str(seg.get("forward_id", "")) != forward_id:
            continue
        content = seg.get("content")
        if isinstance(content, list):
            return [node for node in content if isinstance(node, dict)]
    return None


def _normalize_forward_node(
    node: dict,
    *,
    bot_id: str | None,
    bot_display_name: str,
    timezone: Any,
) -> dict:
    sender = node.get("sender", {}) or {}
    message = node.get("message", []) or []
    timestamp = node.get("time")
    if timestamp:
        try:
            ts = datetime.fromtimestamp(float(timestamp), tz=timezone).isoformat()
        except Exception:
            ts = str(timestamp)
    else:
        ts = datetime.now(timezone).isoformat()

    text = napcat_segments_to_text(message, bot_id=bot_id, bot_display_name=bot_display_name)
    content_segments = build_content_segments(message, bot_id=bot_id, bot_display_name=bot_display_name)
    entry: dict = {
        "role": "user",
        "sender_id": str(sender.get("user_id", "")),
        "sender_name": sender.get("card") or sender.get("nickname") or str(sender.get("user_id", "")),
        "sender_role": sender.get("role", ""),
        "timestamp": ts,
        "content": text,
        "content_type": _determine_content_type(message),
        "content_segments": content_segments,
    }
    if reply_to := get_reply_message_id(message):
        entry["reply_to"] = reply_to
    _attach_forward_images(entry, message)
    return entry


def _attach_forward_images(entry: dict, message: list[dict]) -> None:
    """Attach normal image state for images/stickers inside a forward node."""
    image_refs = [
        (seg["ref"], "动画表情" if seg["type"] == "sticker" else "图片")
        for seg in entry.get("content_segments", [])
        if seg.get("type") in ("image", "sticker") and "ref" in seg
    ]
    if not image_refs:
        return

    image_tasks: list[tuple[str, str, str]] = []
    for seg in message:
        if seg.get("type") not in ("image", "mface"):
            continue
        data = seg.get("data", {}) or {}
        if raw_b64 := data.get("base64", ""):
            image_tasks.append(("b64", raw_b64, "image/jpeg"))
        elif url := data.get("url", ""):
            image_tasks.append(("url", url, ""))

    images: dict[str, dict] = {}
    pending_downloads: list[tuple[str, str, str]] = []
    for (ref, label), (kind, value, preset_mime) in zip(image_refs, image_tasks):
        if kind == "b64":
            images[ref] = {"base64": value, "mime": preset_mime, "label": label}
        else:
            images[ref] = {"pending": True, "label": label}
            pending_downloads.append((ref, value, label))

    if images:
        entry["images"] = images
    if pending_downloads:
        entry["_pending_images"] = pending_downloads


async def _download_forward_node_images(nodes: list[dict]) -> None:
    from napcat.events import download_pending_images

    await asyncio.gather(*[
        download_pending_images(node)
        for node in nodes
        if node.get("_pending_images")
    ])


def _process_forward_node_images(nodes: list[dict]) -> None:
    try:
        import app_state

        vision_bridge = getattr(app_state, "vision_bridge", None)
    except Exception:
        vision_bridge = None
    if not vision_bridge:
        return

    for node in nodes:
        if node.get("images"):
            vision_bridge.process_entry(node)


def _messages_from_forward_result(result: Any) -> list[dict]:
    if isinstance(result, dict):
        raw = result.get("messages", [])
    elif isinstance(result, list):
        raw = result
    else:
        raw = []
    return [node for node in raw if isinstance(node, dict)]


def _find_forward_content_in_history_result(
    result: Any,
    *,
    root_message_id: str,
    forward_id: str,
) -> list[dict] | None:
    if not isinstance(result, dict):
        return None
    messages = result.get("messages") or result.get("message") or result.get("msgList") or []
    for msg in messages:
        if not isinstance(msg, dict) or str(msg.get("message_id", "")) != str(root_message_id):
            continue
        for seg in msg.get("message") or []:
            if not isinstance(seg, dict) or seg.get("type") != "forward":
                continue
            data = seg.get("data") or {}
            if forward_id and str(data.get("id", "")) != str(forward_id):
                continue
            content = data.get("content")
            if isinstance(content, list):
                return [node for node in content if isinstance(node, dict)]
    return None


def _fetch_forward_content_from_history(
    *,
    napcat_client: Any,
    session: Any,
    root_message_id: str,
    forward_id: str,
    loop: asyncio.AbstractEventLoop,
) -> list[dict] | None:
    conv_type = getattr(session, "conv_type", "")
    conv_id = str(getattr(session, "conv_id", "") or "")
    if conv_type != "group" or not conv_id:
        return None

    result = run_coroutine_sync(
        napcat_client.send_api(
            "get_group_msg_history",
            {
                "group_id": conv_id,
                "count": 50,
                "parse_mult_msg": True,
            },
            timeout=15.0,
        ),
        loop,
        timeout=20,
    )
    return _find_forward_content_in_history_result(
        result,
        root_message_id=root_message_id,
        forward_id=forward_id,
    )


def _fetch_forward_frame(
    *,
    napcat_client: Any,
    session: Any,
    forward_id: str,
    root_message_id: str,
    path: list[int],
    title: str,
    content_nodes_raw: list[dict] | None = None,
) -> dict | None:
    loop: asyncio.AbstractEventLoop | None = getattr(napcat_client, "_loop", None)
    if loop is None or not loop.is_running():
        raise RuntimeError("主事件循环不可用")

    if content_nodes_raw is None:
        result = run_coroutine_sync(
            napcat_client.send_api("get_forward_msg", {"id": forward_id}, timeout=15.0),
            loop,
            timeout=20,
        )
        if result is None:
            content_nodes_raw = None
            if not path:
                content_nodes_raw = _fetch_forward_content_from_history(
                    napcat_client=napcat_client,
                    session=session,
                    root_message_id=root_message_id,
                    forward_id=forward_id,
                    loop=loop,
                )
            if content_nodes_raw is None:
                api_error = getattr(napcat_client, "last_api_error", None) or {}
                message = api_error.get("message") or "NapCat 返回空结果"
                raise RuntimeError(f"NapCat 调用 get_forward_msg 失败: {message}")
        else:
            content_nodes_raw = _messages_from_forward_result(result)

    nodes_raw = [node for node in content_nodes_raw if isinstance(node, dict)]
    timezone = getattr(session, "_timezone", None) or ZoneInfo("Asia/Shanghai")
    nodes = [
        _normalize_forward_node(
            node,
            bot_id=getattr(napcat_client, "bot_id", None),
            bot_display_name=getattr(session, "_qq_card", "") or getattr(session, "_qq_name", ""),
            timezone=timezone,
        )
        for node in nodes_raw
    ]
    if any(node.get("_pending_images") for node in nodes):
        run_coroutine_sync(_download_forward_node_images(nodes), loop, timeout=60)
    _process_forward_node_images(nodes)
    return {
        "forward_id": forward_id,
        "root_message_id": root_message_id,
        "path": path,
        "title": title or "合并转发",
        "nodes": nodes,
        "total": len(nodes),
        "page_offset": 0,
        "page_size": 8,
    }


def _view_summary(session: Any) -> dict:
    stack = getattr(session, "forward_browser_stack", []) or []
    if not stack:
        return {"mode": "chat"}
    frame = stack[-1]
    nodes = frame.get("nodes") or []
    page_offset = int(frame.get("page_offset") or 0)
    page_size = int(frame.get("page_size") or 8)
    return {
        "mode": "forward",
        "depth": len(stack),
        "title": frame.get("title") or "合并转发",
        "total": int(frame.get("total") or len(nodes)),
        "page_offset": page_offset,
        "page_size": page_size,
        "has_previous": page_offset > 0,
        "has_next": page_offset + page_size < len(nodes),
        "path": [
            {"kind": "chat_message", "message_id": frame.get("root_message_id", "")},
            *[
                {"kind": "forward_node", "node_index": index}
                for index in (frame.get("path") or [])
            ],
        ],
    }


def make_handler(session: Any, napcat_client: Any) -> Callable:
    def execute(
        action: str = "",
        id: str = "",
        impression: str = "",
        motivation: str = "",
        **kwargs,
    ) -> dict:
        action = (action or "").strip()
        target_id = str(id or "").strip()

        if action not in {"open", "next_page", "prev_page", "back", "close_all"}:
            return {"ok": False, "action": action, "moved": False, "error": f"未知 action: {action!r}"}

        if action == "close_all":
            moved = bool(getattr(session, "forward_browser_stack", None))
            session.close_forward_browser()
            return {"ok": True, "action": action, "moved": moved, "view": _view_summary(session)}

        if action == "back":
            stack = getattr(session, "forward_browser_stack", []) or []
            if not stack:
                return {"ok": True, "action": action, "moved": False, "message": "当前没有打开的合并转发。"}
            stack.pop()
            session.forward_virtual_registry.clear()
            return {"ok": True, "action": action, "moved": True, "view": _view_summary(session)}

        if action in {"next_page", "prev_page"}:
            stack = getattr(session, "forward_browser_stack", []) or []
            if not stack:
                return {"ok": False, "action": action, "moved": False, "error": "当前没有打开的合并转发。"}
            frame = stack[-1]
            page_size = int(frame.get("page_size") or 8)
            nodes = frame.get("nodes") or []
            old_offset = int(frame.get("page_offset") or 0)
            if action == "next_page":
                new_offset = min(old_offset + page_size, max(0, len(nodes) - page_size))
            else:
                new_offset = max(0, old_offset - page_size)
            frame["page_offset"] = new_offset
            session.forward_virtual_registry.clear()
            return {
                "ok": True,
                "action": action,
                "moved": new_offset != old_offset,
                "view": _view_summary(session),
            }

        if not target_id:
            return {"ok": False, "action": action, "moved": False, "error": "open 需要填写 id。"}
        if not napcat_client or not napcat_client.connected:
            return {"ok": False, "action": action, "moved": False, "error": "NapCat 未连接，无法展开合并转发。"}

        is_nested_open = target_id.startswith("fwd:")
        if is_nested_open:
            registry = getattr(session, "forward_virtual_registry", {}) or {}
            openable = registry.get(target_id)
            if not openable:
                return {"ok": False, "action": action, "moved": False, "error": f"id={target_id} 不是当前可打开的合并转发。"}
            forward_id = str(openable["forward_id"])
            root_message_id = str(openable["root_message_id"])
            path = list(openable.get("path") or [])
            title = str(openable.get("title") or "合并转发")
            content_nodes_raw = openable.get("content") if isinstance(openable.get("content"), list) else None
        else:
            entry = _find_real_message(session, target_id)
            if not entry:
                return {"ok": False, "action": action, "moved": False, "error": f"当前会话找不到 message_id={target_id}。"}
            forward_id = _extract_forward_id(entry)
            if not forward_id or entry.get("content_type") != "forward":
                return {"ok": False, "action": action, "moved": False, "error": f"message_id={target_id} 不是一条合并转发消息。"}
            root_message_id = target_id
            path = []
            title = "合并转发"
            content_nodes_raw = _extract_forward_content(entry, forward_id)

        if is_nested_open and forward_id in {str(frame.get("forward_id", "")) for frame in session.forward_browser_stack}:
            return {"ok": False, "action": action, "moved": False, "error": "该合并转发已在当前路径中，已阻止循环打开。"}

        try:
            frame = _fetch_forward_frame(
                napcat_client=napcat_client,
                session=session,
                forward_id=forward_id,
                root_message_id=root_message_id,
                path=path,
                title=title,
                content_nodes_raw=content_nodes_raw,
            )
        except Exception as exc:
            logger.warning("[tools] browse_forward_message: 展开失败 id=%s forward_id=%s: %s", target_id, forward_id, exc)
            return {"ok": False, "action": action, "moved": False, "error": f"展开合并转发失败: {exc}"}

        if frame is None:
            return {"ok": False, "action": action, "moved": False, "error": "NapCat 返回空结果。"}

        if is_nested_open:
            session.forward_browser_stack.append(frame)
        else:
            session.forward_browser_stack = [frame]
        session.forward_virtual_registry.clear()
        return {"ok": True, "action": action, "moved": True, "view": _view_summary(session)}

    return execute
