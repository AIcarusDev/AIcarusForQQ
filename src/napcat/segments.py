"""napcat/segments.py — 消息段格式互转

OneBot v11 消息段（NapCat 格式）与各种中间格式之间的互转：
  - NapCat 消息段 → 纯文本（napcat_segments_to_text）
  - NapCat 消息段 → 结构化内容段（build_content_segments）
  - LLM segments → NapCat 消息段（llm_segments_to_napcat）
"""

import base64
import uuid


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
    "108": "[阴险]", "109": "[亲亲]", "110": "[吓]", "111": "[可怜]",
    "112": "[菜刀]", "113": "[啤酒]", "114": "[篮球]", "115": "[乒乓]",
    "116": "[示爱]", "117": "[瓢虫]", "118": "[抱拳]", "119": "[勾引]",
    "120": "[拳头]", "121": "[差劲]", "122": "[爱你]", "123": "[NO]", "124": "[OK]",
    "171": "[茶]", "172": "[西瓜]", "173": "[啤酒杯]", "174": "[牵手]", "175": "[击掌]",
    "176": "[送花]", "177": "[骰子]", "178": "[快递]", "179": "[玫瑰花瓣]",
    "180": "[发呆]", "181": "[暴怒]", "182": "[跑步]",
    "277": "[汪汪]", "305": "[吃糖]", "306": "[惊喜]",
    "307": "[叹气]", "308": "[无语]",
    "323": "[酸了]", "324": "[yyds]", "326": "[让我看看]",
}

_SEG_LABEL: dict[str, str] = {
    "record": "[语音]",
    "video": "[视频]",
    "forward": "[合并转发]",
    "json": "[卡片消息]",
    "xml": "[XML消息]",
    "poke": "[戳一戳]",
}


# ── NapCat 消息段 → 纯文本 ────────────────────────────────────────────────────

def napcat_segments_to_text(
    message: list[dict],
    bot_id: str | None = None,
    bot_display_name: str = "",
) -> str:
    """将 NapCat 消息段列表转为人类可读的纯文本。

    用于填充 context_messages 的 content 字段。
    """
    parts: list[str] = []
    for seg in message:
        seg_type = seg.get("type", "")
        data = seg.get("data", {})

        if seg_type == "text":
            parts.append(data.get("text", ""))
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
            parts.append("[动画表情]" if data.get("sub_type", 0) == 1 else "[图片]")
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
    """从 NapCat 消息段中提取被回复的消息 ID。"""
    for seg in message:
        if seg.get("type") == "reply":
            return str(seg["data"].get("id", ""))
    return None


def build_content_segments(
    message: list[dict],
    bot_id: str | None = None,
    bot_display_name: str = "",
) -> list[dict]:
    """将 NapCat 消息段列表转为结构化内容段列表。

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
            text = data.get("text", "")
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
            sub_type = data.get("sub_type", 0)
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
            parts.append({"type": "forward", "forward_id": data.get("id", ""), "_needs_expand": True})
        elif seg_type == "record":
            voice_seg = {"type": "voice", "label": "语音"}
            duration = _coerce_duration_seconds(data)
            if duration is not None:
                voice_seg["duration"] = duration
            parts.append(voice_seg)
        elif seg_type in ("video", "json", "xml", "poke"):
            label_map = {
                "video": "视频",
                "json": "卡片消息", "xml": "XML消息", "poke": "戳一戳",
            }
            parts.append({"type": seg_type, "label": label_map.get(seg_type, seg_type)})
        else:
            parts.append({"type": seg_type, "label": seg_type})
    return parts


def _determine_content_type(message_segs: list[dict]) -> str:
    """根据消息段列表判断消息的主要内容类型。"""
    types = {seg.get("type") for seg in message_segs if seg.get("type") != "reply"}
    has_text = any(
        seg.get("type") == "text" and seg.get("data", {}).get("text", "").strip()
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
    if "image" in types and not has_text:
        has_real_image = any(
            seg.get("type") == "image" and seg.get("data", {}).get("sub_type", 0) != 1
            for seg in message_segs
        )
        return "image" if has_real_image else "sticker"
    return "text"


# ── LLM 输出 → NapCat 消息段 ─────────────────────────────────────────────────

def llm_segments_to_napcat(
    segments: list[dict],
    reply_message_id: str | None = None,
) -> list[dict]:
    """将 LLM 输出的 segments 转为 NapCat 消息段数组。

    LLM 输出格式: [{"command": "text", "params": {"content": "..."}}, ...]
    NapCat 输入格式: [{"type": "text", "data": {"text": "..."}}, ...]
    """
    napcat_segs: list[dict] = []

    if reply_message_id:
        napcat_segs.append({"type": "reply", "data": {"id": str(reply_message_id)}})

    for seg in segments:
        cmd = seg.get("command", "")
        params = seg.get("params", {})

        if cmd == "text":
            content = params.get("content", "")
            if content:
                napcat_segs.append({"type": "text", "data": {"text": content}})
        elif cmd == "at":
            user_id = params.get("user_id", "")
            if user_id:
                napcat_segs.append({"type": "at", "data": {"qq": str(user_id)}})
        elif cmd == "sticker":
            sticker_id = params.get("sticker_id", "")
            if sticker_id:
                _data = _load_sticker_for_send(sticker_id)
                if _data is not None:
                    _raw, _mime = _data
                    _b64 = base64.b64encode(_raw).decode("ascii")
                    napcat_segs.append({
                        "type": "image",
                        "data": {"file": f"base64://{_b64}", "sub_type": 1},
                    })

    # @某人后面需要跟一个空格，否则补上
    result: list[dict] = []
    for i, seg in enumerate(napcat_segs):
        result.append(seg)
        if seg.get("type") == "at":
            next_seg = napcat_segs[i + 1] if i + 1 < len(napcat_segs) else None
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
    """懒加载表情包字节，供 llm_segments_to_napcat 使用。返回 (bytes, mime) 或 None。"""
    try:
        from llm.media.sticker_collection import load_sticker_bytes
        return load_sticker_bytes(sticker_id)
    except Exception as e:
        import logging
        logging.getLogger("AICQ.napcat.segments").warning("懒加载表情包失败 id=%s: %s", sticker_id, e)
        return None
