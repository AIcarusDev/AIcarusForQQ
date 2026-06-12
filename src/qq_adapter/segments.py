"""QQ adapter/segments.py — 消息段格式互转

OneBot v11 消息段（QQ adapter 格式）与各种中间格式之间的互转：
  - QQ adapter 消息段 → 纯文本（qq_adapter_segments_to_text）
  - QQ adapter 消息段 → 结构化内容段（build_content_segments）
  - LLM segments → QQ adapter 消息段（llm_segments_to_qq_adapter）
"""

import base64
import json
import logging
import re
import uuid

_seg_logger = logging.getLogger("AICQ.qq_adapter.segments")


class ImageLoadError(Exception):
    """图片 ref 加载失败，由 llm_segments_to_qq_adapter 抛出，调用方应返回工具失败。"""

    def __init__(self, source: str, reason: str = "") -> None:
        msg = f"图片加载失败 source={source!r}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.source = source
        self.reason = reason


def _coerce_duration_seconds(data: dict) -> float | None:
    for key in ("duration", "file_duration", "time"):
        value = data.get(key)
        if value is None or value == "":
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds >= 0:
            return seconds
    return None


# ── QQ 表情 ID → 文字映射 ─────────────────────────────────────────────────────

QQ_FACE: dict[str, str] = {
    "0": "[惊讶]", "1": "[撇嘴]", "2": "[色]", "3": "[发呆]", "4": "[得意]",
    "5": "[流泪]", "6": "[害羞]", "7": "[闭嘴]", "8": "[睡]", "9": "[大哭]",
    "10": "[尴尬]", "11": "[发怒]", "12": "[调皮]", "13": "[呲牙]", "14": "[微笑]",
    "15": "[难过]", "16": "[酷]", "18": "[抓狂]", "19": "[吐]", "20": "[偷笑]",
    "21": "[可爱]", "22": "[白眼]", "23": "[傲慢]", "24": "[饥饿]", "25": "[困]",
    "26": "[惊恐]", "27": "[流汗]", "28": "[憨笑]", "29": "[悠闲]", "30": "[奋斗]",
    "31": "[咒骂]", "32": "[疑问]", "33": "[嘘]", "34": "[晕]", "35": "[折磨]",
    "36": "[衰]", "37": "[骷髅]", "38": "[敲打]", "39": "[再见]", "41": "[发抖]",
    "42": "[爱情]", "43": "[跳跳]", "46": "[猪头]", "49": "[拥抱]", "53": "[蛋糕]",
    "56": "[刀]", "59": "[便便]", "60": "[咖啡]", "63": "[玫瑰]", "64": "[凋谢]",
    "66": "[爱心]", "67": "[心碎]", "69": "[礼物]", "74": "[太阳]", "75": "[月亮]",
    "76": "[赞]", "77": "[踩]", "78": "[握手]", "79": "[胜利]",
    "85": "[飞吻]", "86": "[怄火]", "89": "[西瓜]", "96": "[冷汗]", "97": "[擦汗]",
    "98": "[抠鼻]", "99": "[鼓掌]", "100": "[糗大了]", "101": "[坏笑]", "102": "[左哼哼]",
    "103": "[右哼哼]", "104": "[哈欠]", "105": "[鄙视]", "106": "[委屈]", "107": "[快哭了]",
    "108": "[阴险]", "109": "[左亲亲]", "110": "[吓]", "111": "[可怜]",
    "112": "[菜刀]", "113": "[啤酒]", "114": "[篮球]", "115": "[乒乓]",
    "116": "[示爱]", "117": "[瓢虫]", "118": "[抱拳]", "119": "[勾引]",
    "120": "[拳头]", "121": "[差劲]", "122": "[爱你]", "123": "[NO]", "124": "[OK]",
    "171": "[茶]", "172": "[眨眼睛]", "173": "[泪奔]", "174": "[无奈]", "175": "[卖萌]",
    "176": "[小纠结]", "177": "[喷血]", "178": "[斜眼笑]", "179": "[doge]",
    "180": "[惊喜]", "181": "[戳一戳]", "182": "[笑哭]",
    "277": "[汪汪]", "305": "[右亲亲]", "306": "[牛气冲天]",
    "307": "[喵喵]", "308": "[无语]",
    "323": "[嫌弃]", "324": "[吃糖]", "326": "[生气]",
}

_SEG_LABEL: dict[str, str] = {
    "record": "[语音]",
    "video": "[视频]",
    "forward": "[合并转发]",
    "json": "[卡片消息]",
    "xml": "[XML消息]",
    "markdown": "[Markdown卡片]",
    "music": "[音乐卡片]",
    "contact": "[联系人卡片]",
    "location": "[位置卡片]",
    "miniapp": "[小程序卡片]",
    "poke": "[戳一戳]",
}

_CARD_SEG_TYPES = {"json", "xml", "markdown", "music", "contact", "location", "miniapp"}
_RAW_CARD_LIMIT = 12000
_TEXT_DATA_KEYS = ("text", "content", "message", "msg")


def get_forward_node_message_segments(node: dict) -> list[dict]:
    """Return message segments from a merged-forward node.

    OneBot-style adapters usually put node body segments in ``message``.
    LLOneBot may return the same segment list in ``content``.
    """
    if not isinstance(node, dict):
        return []
    for key in ("message", "content"):
        value = node.get(key)
        if isinstance(value, list):
            return [seg for seg in value if isinstance(seg, dict)]
    return []


def get_text_segment_text(data: dict) -> str:
    """Return text from OneBot-style text segment variants."""
    if not isinstance(data, dict):
        return ""
    for key in _TEXT_DATA_KEYS:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            text = str(value)
            if text:
                return text
    return ""


def get_image_sub_type(data: dict) -> int:
    """Return QQ image sub type across snake_case/camelCase variants."""
    if not isinstance(data, dict):
        return 0
    value = data.get("sub_type", data.get("subType", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _stringify_raw(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _trim_raw(value: str, limit: int = _RAW_CARD_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"...[truncated {len(value) - limit} chars]"


def _parse_jsonish(value) -> dict | list | None:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _walk_values(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)


def _first_field(value, keys: tuple[str, ...]) -> str:
    objects = list(_walk_values(value))
    for key in keys:
        for obj in objects:
            found = obj.get(key)
            if isinstance(found, (str, int, float)) and str(found).strip():
                return str(found).strip()
    return ""


def _infer_json_card_kind(payload) -> str:
    app = _first_field(payload, ("app", "appName"))
    if "music" in app.lower():
        return "music"
    if "miniapp" in app.lower() or _first_field(payload, ("miniappShareOrigin", "appId")):
        return "miniapp"
    if "contact" in app.lower():
        return "contact"
    prompt = _first_field(payload, ("prompt", "desc")).lower()
    if "音乐" in prompt or "music" in prompt:
        return "music"
    if "小程序" in prompt or "miniapp" in prompt:
        return "miniapp"
    return "json"


def _build_json_card(data: dict) -> dict:
    raw_value = data.get("data")
    payload = _parse_jsonish(raw_value)
    raw = _trim_raw(_stringify_raw(raw_value))
    card: dict = {
        "type": "card",
        "kind": _infer_json_card_kind(payload) if payload is not None else "json",
        "label": "卡片消息",
    }
    if payload is not None:
        if title := _first_field(payload, ("title", "name", "musicName")):
            card["title"] = title
        if summary := _first_field(payload, ("desc", "summary", "content", "singer", "tag", "prompt")):
            card["summary"] = summary
        if app := _first_field(payload, ("app", "appName", "source", "sourceName")):
            card["app"] = app
        if url := _first_field(payload, ("jumpUrl", "url", "webUrl", "qqdocurl", "preview")):
            card["url"] = url
    if raw:
        card["raw"] = raw
    return card


def _xml_attr_or_tag(raw: str, names: tuple[str, ...]) -> str:
    for name in names:
        attr_match = re.search(
            rf'\b{name}\s*=\s*["\']([^"\']+)["\']',
            raw,
            flags=re.IGNORECASE,
        )
        if attr_match:
            return attr_match.group(1).strip()
        tag_match = re.search(
            rf"<{name}\b[^>]*>(.*?)</{name}>",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tag_match:
            return re.sub(r"\s+", " ", tag_match.group(1)).strip()
    return ""


def _build_xml_card(data: dict) -> dict:
    raw = _trim_raw(str(data.get("data", "") or ""))
    card: dict = {"type": "card", "kind": "xml", "label": "XML消息"}
    if title := _xml_attr_or_tag(raw, ("title", "name")):
        card["title"] = title
    if summary := _xml_attr_or_tag(raw, ("brief", "desc", "summary", "content")):
        card["summary"] = summary
    if url := _xml_attr_or_tag(raw, ("url", "jumpUrl", "actionData")):
        card["url"] = url
    if raw:
        card["raw"] = raw
    return card


def _build_markdown_card(data: dict) -> dict:
    content = str(data.get("content", "") or "")
    summary = re.sub(r"\s+", " ", content).strip()
    card: dict = {"type": "card", "kind": "markdown", "label": "Markdown卡片"}
    if summary:
        card["summary"] = summary[:120] + ("..." if len(summary) > 120 else "")
        card["markdown"] = content
    return card


def _build_music_card(data: dict) -> dict:
    card: dict = {"type": "card", "kind": "music", "label": "音乐卡片"}
    if platform := str(data.get("type", "") or ""):
        card["platform"] = platform
    if title := str(data.get("title", "") or ""):
        card["title"] = title
    if summary := str(data.get("content", "") or ""):
        card["summary"] = summary
    if url := str(data.get("url", "") or ""):
        card["url"] = url
    if music_id := data.get("id"):
        card["music_id"] = str(music_id)
    card["raw"] = _trim_raw(_stringify_raw(data))
    return card


def _build_contact_card(data: dict) -> dict:
    contact_type = str(data.get("type", "") or "qq")
    contact_id = str(data.get("id", "") or "")
    card: dict = {
        "type": "card",
        "kind": "contact",
        "label": "联系人卡片",
        "contact_type": contact_type,
        "contact_id": contact_id,
    }
    card["summary"] = "群聊推荐" if contact_type == "group" else "QQ联系人分享"
    card["raw"] = _trim_raw(_stringify_raw(data))
    return card


def _build_location_card(data: dict) -> dict:
    card: dict = {"type": "card", "kind": "location", "label": "位置卡片"}
    if title := str(data.get("title", "") or ""):
        card["title"] = title
    if summary := str(data.get("content", "") or ""):
        card["summary"] = summary
    for key in ("lat", "lon"):
        if data.get(key) is not None:
            card[key] = str(data.get(key))
    card["raw"] = _trim_raw(_stringify_raw(data))
    return card


def _build_miniapp_card(data: dict) -> dict:
    raw_value = data.get("data")
    payload = _parse_jsonish(raw_value)
    card = _build_json_card({"data": raw_value})
    card["kind"] = "miniapp"
    card["label"] = "小程序卡片"
    if payload is None and raw_value:
        card["raw"] = _trim_raw(str(raw_value))
    return card


def _build_card_segment(seg_type: str, data: dict) -> dict:
    if seg_type == "json":
        return _build_json_card(data)
    if seg_type == "xml":
        return _build_xml_card(data)
    if seg_type == "markdown":
        return _build_markdown_card(data)
    if seg_type == "music":
        return _build_music_card(data)
    if seg_type == "contact":
        return _build_contact_card(data)
    if seg_type == "location":
        return _build_location_card(data)
    if seg_type == "miniapp":
        return _build_miniapp_card(data)
    return {"type": "card", "kind": seg_type, "label": seg_type, "raw": _trim_raw(_stringify_raw(data))}


# ── QQ adapter 消息段 → 纯文本 ────────────────────────────────────────────────────

def qq_adapter_segments_to_text(
    message: list[dict],
    bot_id: str | None = None,
    bot_display_name: str = "",
) -> str:
    """将 QQ adapter 消息段列表转为人类可读的纯文本。

    用于填充 context_messages 的 content 字段。
    """
    parts: list[str] = []
    for seg in message:
        seg_type = seg.get("type", "")
        data = seg.get("data", {})

        if seg_type == "text":
            parts.append(get_text_segment_text(data))
        elif seg_type == "face":
            face_id = str(data.get("id", ""))
            parts.append(QQ_FACE.get(face_id, f"[表情:{face_id}]"))
        elif seg_type == "at":
            qq = str(data.get("qq", ""))
            if qq == "all":
                parts.append("@全体成员")
            elif qq == bot_id:
                parts.append(f"@{bot_display_name or qq}")
            else:
                parts.append(f"@{qq}")
        elif seg_type == "mface":
            parts.append("[动画表情]")
        elif seg_type == "image":
            parts.append("[动画表情]" if get_image_sub_type(data) == 1 else "[图片]")
        elif seg_type == "file":
            parts.append(f"[文件:{data.get('name', '未知')}]")
        elif seg_type == "reply":
            pass  # 回复引用不显示在正文里
        elif label := _SEG_LABEL.get(seg_type):
            parts.append(label)
        else:
            parts.append(f"[{seg_type}]")

    return "".join(parts).strip()


def get_reply_message_id(message: list[dict]) -> str | None:
    """从 QQ adapter 消息段中提取被回复的消息 ID。"""
    for seg in message:
        if seg.get("type") == "reply":
            return str(seg["data"].get("id", ""))
    return None


def build_content_segments(
    message: list[dict],
    bot_id: str | None = None,
    bot_display_name: str = "",
) -> list[dict]:
    """将 QQ adapter 消息段列表转为结构化内容段列表。

    返回列表元素格式:
      {"type": "text",    "text": "..."}
      {"type": "mention", "uid": "...", "display": "@..."}
      {"type": "emoji",   "id": "...", "name": "..."}
      {"type": "image"}
      {"type": "file",    "filename": "..."}
      其他: {"type": "..."}
    """
    parts: list[dict] = []
    for seg in message:
        seg_type = seg.get("type", "")
        data = seg.get("data", {})

        if seg_type == "text":
            text = get_text_segment_text(data)
            if text:
                parts.append({"type": "text", "text": text})
        elif seg_type == "face":
            face_id = str(data.get("id", ""))
            name = QQ_FACE.get(face_id, f"表情{face_id}")
            clean_name = name.strip("[]")
            parts.append({"type": "emoji", "id": face_id, "name": clean_name})
        elif seg_type == "at":
            qq = str(data.get("qq", ""))
            if qq == "all":
                parts.append({"type": "mention", "uid": "all", "display": "@全体成员"})
            elif qq == bot_id:
                display = bot_display_name or qq
                parts.append({"type": "mention", "uid": "self", "display": f"@{display}"})
            else:
                name = data.get("name", "").strip()
                display_name = name if name else qq
                parts.append({"type": "mention", "uid": qq, "display": f"@{display_name}"})
        elif seg_type == "mface":
            ref = uuid.uuid4().hex[:12]
            parts.append({"type": "sticker", "ref": ref})
        elif seg_type == "image":
            sub_type = get_image_sub_type(data)
            ref = uuid.uuid4().hex[:12]
            if sub_type == 1:
                parts.append({"type": "sticker", "ref": ref})
            else:
                parts.append({"type": "image", "ref": ref})
        elif seg_type == "file":
            parts.append({"type": "file", "filename": data.get("name", "未知")})
        elif seg_type == "reply":
            pass  # 回复引用单独处理，不放入 content_segments
        elif seg_type == "forward":
            part = {"type": "forward", "forward_id": str(data.get("id", "")), "_needs_expand": True}
            if isinstance(data.get("content"), list):
                part["content"] = data["content"]
            parts.append(part)
        elif seg_type == "record":
            voice_seg: dict[str, str | float] = {"type": "voice", "label": "语音"}
            duration = _coerce_duration_seconds(data)
            if duration is not None:
                voice_seg["duration"] = duration
            parts.append(voice_seg)
        elif seg_type in _CARD_SEG_TYPES:
            parts.append(_build_card_segment(seg_type, data if isinstance(data, dict) else {}))
        elif seg_type in ("video", "poke"):
            label_map = {
                "video": "视频",
                "poke": "戳一戳",
            }
            parts.append({"type": seg_type, "label": label_map.get(seg_type, seg_type)})
        else:
            parts.append({"type": seg_type, "label": seg_type})
    return parts


def _determine_content_type(message_segs: list[dict]) -> str:
    """根据消息段列表判断消息的主要内容类型。"""
    types = {seg.get("type") for seg in message_segs if seg.get("type") != "reply"}
    has_text = any(
        seg.get("type") == "text" and get_text_segment_text(seg.get("data", {})).strip()
        for seg in message_segs
    )
    if "forward" in types:
        return "forward"
    if "file" in types:
        return "file"
    if "record" in types and not has_text:
        return "voice"
    if "mface" in types and not has_text:
        return "sticker"
    if "video" in types and not has_text:
        return "video"
    if "image" in types and not has_text:
        has_real_image = any(
            seg.get("type") == "image" and get_image_sub_type(seg.get("data", {})) != 1
            for seg in message_segs
        )
        return "image" if has_real_image else "sticker"
    return "text"


# ── LLM 输出 → QQ adapter 消息段 ─────────────────────────────────────────────────

def _sticker_image_data(file_value: str, adapter: str | None = None) -> dict:
    data: dict = {"file": file_value}
    if str(adapter or "").lower() == "llonebot":
        data["subType"] = 1
    else:
        data["sub_type"] = 1
    return data


def llm_segments_to_qq_adapter(
    segments: list[dict],
    reply_message_id: str | None = None,
    *,
    adapter: str | None = None,
) -> list[dict]:
    """将 LLM 输出的 segments 转为 QQ adapter 消息段数组。

    LLM 输出格式: [{"command": "text", "params": {"content": "..."}}, ...]
    QQ adapter 输入格式: [{"type": "text", "data": {"text": "..."}}, ...]
    """
    qq_adapter_segs: list[dict] = []

    if reply_message_id:
        qq_adapter_segs.append({"type": "reply", "data": {"id": str(reply_message_id)}})

    for seg in segments:
        cmd = seg.get("command", "")
        params = seg.get("params", {})

        if cmd == "text":
            content = params.get("content", "")
            if content:
                qq_adapter_segs.append({"type": "text", "data": {"text": content}})
        elif cmd == "at":
            user_id = params.get("user_id", "")
            if user_id:
                qq_adapter_segs.append({"type": "at", "data": {"qq": str(user_id)}})
        elif cmd == "sticker":
            sticker_id = params.get("sticker_id", "")
            if sticker_id:
                _data = _load_sticker_for_send(sticker_id)
                if _data is not None:
                    _raw, _mime = _data
                    _b64 = base64.b64encode(_raw).decode("ascii")
                    qq_adapter_segs.append({
                        "type": "image",
                        "data": _sticker_image_data(f"base64://{_b64}", adapter),
                    })
                elif params.get("_fallback_base64"):
                    qq_adapter_segs.append({
                        "type": "image",
                        "data": _sticker_image_data(f"base64://{params['_fallback_base64']}", adapter),
                    })
        elif cmd == "image":
            image_ref = params.get("image_ref", "")
            if not image_ref:
                raise ImageLoadError("image_ref", "image segment missing image_ref")
            file_val = _load_browser_image_as_base64(str(image_ref))
            if file_val is _IMAGE_LOAD_FAILED:
                raise ImageLoadError(str(image_ref), "browser image ref not found")
            qq_adapter_segs.append({
                "type": "image",
                "data": {"file": file_val},
            })

    # @某人后面需要跟一个空格，否则补上
    result: list[dict] = []
    for i, seg in enumerate(qq_adapter_segs):
        result.append(seg)
        if seg.get("type") == "at":
            next_seg = qq_adapter_segs[i + 1] if i + 1 < len(qq_adapter_segs) else None
            if next_seg is None:
                result.append({"type": "text", "data": {"text": " "}})
            elif next_seg.get("type") == "text":
                text_content = next_seg["data"].get("text", "")
                if not text_content.startswith(" "):
                    result.append({"type": "text", "data": {"text": " "}})
            else:
                result.append({"type": "text", "data": {"text": " "}})
    return result


# ── 表情包加载辅助 ────────────────────────────────────────────────────────────

def _load_sticker_for_send(sticker_id: str):
    """懒加载表情包字节，供 llm_segments_to_qq_adapter 使用。返回 (bytes, mime) 或 None。"""
    try:
        from llm.media.sticker_collection import load_sticker_bytes
        return load_sticker_bytes(sticker_id)
    except Exception as e:
        _seg_logger.warning("懒加载表情包失败 id=%s: %s", sticker_id, e)
        return None


# ── 浏览器图片缓存加载辅助 ────────────────────────────────────────────────────

_IMAGE_LOAD_FAILED = "__image_load_failed__"


def _load_browser_image_as_base64(image_ref: str) -> str:
    try:
        from browser import read_browser_image_file

        item = read_browser_image_file(image_ref)
    except Exception as exc:
        _seg_logger.warning("[segments] 浏览器图片缓存读取失败 ref=%s — %s", image_ref, exc)
        return _IMAGE_LOAD_FAILED
    if item is None:
        _seg_logger.warning("[segments] 浏览器图片缓存不存在 ref=%s", image_ref)
        return _IMAGE_LOAD_FAILED
    raw, _mime = item
    if not raw:
        return _IMAGE_LOAD_FAILED
    return f"base64://{base64.b64encode(raw).decode('ascii')}"
