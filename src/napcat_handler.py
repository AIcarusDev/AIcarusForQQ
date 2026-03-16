"""napcat_handler.py — NapCat WebSocket 连接层 + 消息翻译

直连 NapCat（OneBot v11 反向 WebSocket），省去中间 adapter 进程。
负责：
  1. 作为 WebSocket Server 接受 NapCat 连接
  2. 接收 NapCat 事件 → 翻译为 core 内部上下文格式
  3. 调用 LLM 获取回复 → 翻译为 NapCat 消息段 → 调用 NapCat API 发送
"""

import asyncio
import base64
import html as html_mod
import json
import logging
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Callable, Coroutine

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State as WsState
import random
from pypinyin import pinyin, Style

logger = logging.getLogger("AICQ.napcat")


# ── 图片下载工具 ──────────────────────────────────────────────────────────────

async def _fetch_image_b64(url: str) -> tuple[str, str] | None:
    """从 URL 下载图片，返回 (base64字符串, mime_type)，失败返回 None。"""
    loop = asyncio.get_event_loop()
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


# ── QQ 表情 ID → 文字映射（从 adapter 搬过来的精华部分） ─────────────

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


# ── NapCat 消息段 → 纯文本 ────────────────────────────────

def napcat_segments_to_text(message: list[dict], bot_id: str | None = None, bot_display_name: str = "") -> str:
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
        elif seg_type == "image":
            parts.append("[图片]")
        elif seg_type == "record":
            parts.append("[语音]")
        elif seg_type == "video":
            parts.append("[视频]")
        elif seg_type == "reply":
            pass  # 回复引用不显示在正文里
        elif seg_type == "forward":
            parts.append("[合并转发]")
        elif seg_type == "json":
            parts.append("[卡片消息]")
        elif seg_type == "xml":
            parts.append("[XML消息]")
        elif seg_type == "file":
            parts.append(f"[文件:{data.get('name', '未知')}]")
        elif seg_type == "poke":
            parts.append("[戳一戳]")
        else:
            parts.append(f"[{seg_type}]")

    return "".join(parts).strip()


def get_reply_message_id(message: list[dict]) -> str | None:
    """从 NapCat 消息段中提取被回复的消息 ID。"""
    for seg in message:
        if seg.get("type") == "reply":
            return str(seg["data"].get("id", ""))
    return None


def build_content_segments(message: list[dict], bot_id: str | None = None, bot_display_name: str = "") -> list[dict]:
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
            # 去掉 QQ_FACE 里的方括号以获取纯名称
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
                parts.append({"type": "mention", "uid": qq, "display": f"@{qq}"})
        elif seg_type == "image":
            parts.append({"type": "image"})
        elif seg_type == "file":
            parts.append({"type": "file", "filename": data.get("name", "未知")})
        elif seg_type == "reply":
            pass  # 回复引用单独处理，不放入 content_segments
        elif seg_type in ("record", "video", "forward", "json", "xml", "poke"):
            label_map = {
                "record": "语音", "video": "视频", "forward": "合并转发",
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
    if "file" in types:
        return "file"
    if "image" in types and not has_text:
        return "image"
    return "text"


# ── LLM 输出 → NapCat 消息段 ─────────────────────────────

def llm_segments_to_napcat(
    segments: list[dict],
    reply_message_id: str | None = None,
) -> list[dict]:
    """将 LLM 输出的 segments 转为 NapCat 消息段数组。

    LLM 输出格式: [{"command": "text", "params": {"content": "..."}}, ...]
    NapCat 输入格式: [{"type": "text", "data": {"text": "..."}}, ...]
    """
    napcat_segs: list[dict] = []

    # 如果有引用回复，放在最前面
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

    # QQ 默认行为：@某人后面需要跟一个空格，否则补上
    # 遍历所有段，若某段是 at，检查其后一段是否以空格开头
    result: list[dict] = []
    for i, seg in enumerate(napcat_segs):
        result.append(seg)
        if seg.get("type") == "at":
            next_seg = napcat_segs[i + 1] if i + 1 < len(napcat_segs) else None
            if next_seg is None:
                # @在末尾，追加一个空格
                result.append({"type": "text", "data": {"text": " "}})
            elif next_seg.get("type") == "text":
                text_content = next_seg["data"].get("text", "")
                if not text_content.startswith(" "):
                    result.append({"type": "text", "data": {"text": " "}})
            else:
                # 后面是非 text 段（例如另一个 at、图片等），同样补空格
                result.append({"type": "text", "data": {"text": " "}})
    return result


# ── NapCat 事件 → core 上下文条目 ────────────────────────

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
      images        — [{"base64": str, "mime": str}, ...]（可选）
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

    # 结构化内容段 & 回复引用
    content_segments = build_content_segments(message_segs, bot_id=bot_id, bot_display_name=bot_display_name)
    reply_to = get_reply_message_id(message_segs)
    content_type = _determine_content_type(message_segs)
    # 群聊才有 role 字段（owner/admin/member），私聊无
    sender_role = sender.get("role", "") if msg_type == "group" else ""

    # 并发下载本条消息中的所有图片
    images: list[dict] = []
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

    for kind, value, preset_mime in image_tasks:
        if kind == "b64":
            images.append({"base64": value, "mime": preset_mime})
        else:
            result = await _fetch_image_b64(value)
            if result:
                b64, mime = result
                images.append({"base64": b64, "mime": mime})

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


# ── NapCat 事件 → 调试用 XML ─────────────────────────────

async def napcat_event_to_debug_xml(
    event: dict,
    bot_id: str | None = None,
    timezone: Any = None,
) -> str:
    """将 NapCat 消息事件转为完整的调试 XML，展示原始结构和 LLM 视角。"""
    esc = html_mod.escape

    from zoneinfo import ZoneInfo
    tz = timezone or ZoneInfo("Asia/Shanghai")
    ts = datetime.fromtimestamp(event.get("time", 0), tz=tz).isoformat()

    msg_type = event.get("message_type", "unknown")
    msg_id = str(event.get("message_id", ""))
    sender = event.get("sender", {})
    sender_id = str(sender.get("user_id", "unknown"))
    nickname = esc(sender.get("nickname", ""))
    card = esc(sender.get("card", ""))
    group_id = str(event.get("group_id", "")) if msg_type == "group" else ""

    lines = [
        f'<napcat_event type="message" message_type="{esc(msg_type)}" timestamp="{esc(ts)}">',
        "  <source>",
    ]
    if msg_type == "group" and group_id:
        lines.append(f'    <group id="{esc(group_id)}" />')
    lines.append(
        f'    <sender id="{esc(sender_id)}" nickname="{nickname}" card="{card}" />'
    )
    lines.append("  </source>")

    # 原始消息段
    lines.append(f'  <raw_message id="{esc(msg_id)}">')
    for seg in event.get("message", []):
        seg_type = seg.get("type", "")
        data = seg.get("data", {})
        attrs = " ".join(f'{esc(k)}="{esc(str(v))}"' for k, v in data.items())
        if seg_type == "text":
            lines.append(f"    <segment type=\"text\">{esc(data.get('text', ''))}</segment>")
        else:
            lines.append(f"    <segment type=\"{esc(seg_type)}\" {attrs} />")
    lines.append("  </raw_message>")

    # LLM 看到的 context_entry 视角
    ctx = await napcat_event_to_context(event, bot_id=bot_id, timezone=timezone)
    lines.append("  <context_entry>")
    if ctx:
        safe_name = esc(ctx["sender_name"])
        safe_content = esc(ctx["content"], quote=False)
        lines.append(
            f'    <message id="{ctx["message_id"]}" '
            f'sender_id="{ctx["sender_id"]}" '
            f'sender_name="{safe_name}" '
            f'timestamp="{ctx["timestamp"]}">'
        )
        lines.append(f"      {safe_content}")
        lines.append("    </message>")
    else:
        lines.append("    <!-- 该事件未产生有效 context entry（可能内容为空） -->")
    lines.append("  </context_entry>")

    lines.append("</napcat_event>")
    return "\n".join(lines)


def should_respond(event: dict, bot_id: str | None, bot_name: str = "") -> bool:
    """判断是否应该回复这条消息。

    私聊：始终回复
    群聊：被 @、消息中提到 bot_name、或回复了 bot 的消息时回复
    """
    msg_type = event.get("message_type", "")

    # 私聊始终响应
    if msg_type == "private":
        return True

    # 群聊：检查是否被 @ 或被提及
    message_segs = event.get("message", [])
    for seg in message_segs:
        if seg.get("type") == "at":
            if str(seg.get("data", {}).get("qq", "")) == str(bot_id):
                return True
        if seg.get("type") == "text" and bot_name:
            if bot_name in seg.get("data", {}).get("text", ""):
                return True

    return False


# ══════════════════════════════════════════════════════════════
#  NapCat WebSocket 客户端
# ══════════════════════════════════════════════════════════════

class NapcatClient:
    """管理与 NapCat 的 WebSocket 连接。

    作为 WebSocket Server，等待 NapCat 主动连接（反向 WS）。
    """

    def __init__(self, bot_name: str = "Gemini"):
        self.bot_name: str = bot_name
        self.bot_id: str | None = None
        self._ws: ServerConnection | None = None
        self._server: Any = None
        self._api_futures: dict[str, asyncio.Future] = {}
        self._on_message: Callable[..., Coroutine] | None = None
        self._on_connect: Callable[[], Coroutine] | None = None
        # 同步完成前阻塞消息分发
        self._ready: asyncio.Event = asyncio.Event()
        # 主事件循环引用（start() 后设置），供工具函数在线程中跨线程调用 async API 使用
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.state is WsState.OPEN

    def set_message_handler(
        self,
        handler: Callable[[dict, str], Coroutine],
    ) -> None:
        """注册消息处理回调: async def handler(event: dict, conversation_id: str)"""
        self._on_message = handler

    def set_connect_handler(
        self,
        handler: Callable[[], Coroutine],
    ) -> None:
        """注册 NapCat 连接就绪回调: async def handler()"""
        self._on_connect = handler

    async def start(self, host: str = "127.0.0.1", port: int = 8078) -> None:
        """启动 WebSocket 服务器，等待 NapCat 连接。"""
        self._loop = asyncio.get_running_loop()
        self._server = await websockets.serve(
            self._connection_handler,
            host,
            port,
        )
        logger.info("NapCat WebSocket 服务已启动: ws://%s:%d", host, port)

    async def stop(self) -> None:
        """关闭服务器。"""
        # 先主动关闭当前活跃连接，否则 wait_closed() 会永远等待
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._server:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("NapCat WebSocket 服务关闭超时，已强制退出")
            logger.info("NapCat WebSocket 服务已关闭")

    async def send_api(
        self,
        action: str,
        params: dict,
        timeout: float = 15.0,
    ) -> dict | None:
        """调用 NapCat API 并等待响应（echo 匹配）。"""
        if not self.connected:
            logger.warning("NapCat 未连接，无法调用 API: %s", action)
            return None

        echo = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._api_futures[echo] = fut

        payload = json.dumps({"action": action, "params": params, "echo": echo})
        assert self._ws is not None
        await self._ws.send(payload)
        logger.debug("→ NapCat API: %s params=%s echo=%s", action, params, echo[:8])

        try:
            resp = await asyncio.wait_for(fut, timeout)
            if resp.get("status") == "ok":
                return resp.get("data")
            else:
                logger.warning(
                    "NapCat API %s 失败: status=%s msg=%s",
                    action, resp.get("status"), resp.get("message", ""),
                )
                return None
        except TimeoutError:
            self._api_futures.pop(echo, None)
            logger.error("NapCat API %s 超时 (%ss)", action, timeout)
            return None

    def _calculate_typing_delay(self, text: str) -> float:
        """计算模拟打字延迟."""
        key_delay_min = 0.03
        key_delay_max = 0.12
        char_selection_delay_min = 0.08
        char_selection_delay_max = 0.15
        space_pause = 0.1
        punctuation_pause_min = 0.2
        punctuation_pause_max = 0.45
        punctuation_to_pause = "，。！？；、,."
        initial_thinking_min = 0.15
        initial_thinking_max = 0.4
        max_total_delay = 20.0

        if not text:
            return 0.05

        total_delay = random.uniform(initial_thinking_min, initial_thinking_max)
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                try:
                    p_list = pinyin(char, style=Style.NORMAL)
                    p_str = p_list[0][0]
                    for _ in p_str:
                        total_delay += random.uniform(key_delay_min, key_delay_max)
                    total_delay += random.uniform(
                        char_selection_delay_min, char_selection_delay_max
                    )
                except IndexError:
                    total_delay += 0.2
            elif "a" <= char.lower() <= "z":
                total_delay += random.uniform(key_delay_min, key_delay_max)
            elif char in punctuation_to_pause:
                total_delay += random.uniform(punctuation_pause_min, punctuation_pause_max)
            elif char.isspace():
                total_delay += space_pause
            else:
                total_delay += random.uniform(key_delay_min, key_delay_max)

        if len(text) > 10 and random.random() < 0.15:
            correction_time = random.uniform(0.5, 1.2)
            total_delay += correction_time

        return min(total_delay, max_total_delay)

    async def send_message(
        self,
        *,
        group_id: int | str | None = None,
        user_id: int | str | None = None,
        message: list[dict],
    ) -> dict | None:
        """发送消息的快捷方法。"""
        # 模拟打字延迟
        text_content = napcat_segments_to_text(message)
        delay = self._calculate_typing_delay(text_content)
        if delay > 0.1:
            logger.debug(f"模拟打字延迟: {delay:.2f}s (len={len(text_content)})")
            await asyncio.sleep(delay)

        params: dict[str, Any] = {"message": message}
        if group_id is not None:
            params["group_id"] = int(group_id)
            params["message_type"] = "group"
        elif user_id is not None:
            params["user_id"] = int(user_id)
            params["message_type"] = "private"
        else:
            logger.error("send_message: 必须指定 group_id 或 user_id")
            return None

        return await self.send_api("send_msg", params)

    # ── 内部方法 ──────────────────────────────────────────

    async def _connection_handler(self, ws: ServerConnection) -> None:
        """NapCat 连接进来时的主处理循环。"""
        logger.info("NapCat 已连接: %s", ws.remote_address)
        self._ws = ws
        self._ready.clear()  # 断线重连时重置，等新一轮同步完成

        # 直接从握手 header 读取 bot QQ 号，避免竞态（响应在消息循环启动前就到了会被丢弃）
        try:
            req = ws.request
            self.bot_id = str((req.headers.get("X-Self-ID", "") if req else "") or "")
            if self.bot_id:
                logger.info("Bot ID (from header): QQ=%s", self.bot_id)
            else:
                logger.warning("未能从 header 读取 X-Self-ID，bot_id 暂为空")
        except Exception as e:
            logger.warning("读取 X-Self-ID header 失败: %s", e)

        try:
            async for raw in ws:
                try:
                    data: dict = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("NapCat 发来无法解析的数据: %s", str(raw)[:100])
                    continue

                # API 响应（带 echo）
                if echo := data.get("echo"):
                    fut = self._api_futures.pop(echo, None)
                    if fut and not fut.done():
                        fut.set_result(data)
                    continue

                post_type = data.get("post_type", "")

                if post_type == "message":
                    asyncio.create_task(self._dispatch_message(data))
                elif post_type == "meta_event":
                    await self._handle_meta(data)
                elif post_type == "notice":
                    logger.debug("NapCat 通知: %s", data.get("notice_type"))
                # message_sent、request 等直接忽略

        except websockets.ConnectionClosed:
            logger.info("NapCat 连接已断开")
        finally:
            self._ws = None
            self.bot_id = None
            self._ready.clear()

    async def _dispatch_message(self, event: dict) -> None:
        """分发消息事件给注册的回调。"""
        # 等待初始化同步完成，避免 prompt 中信息缺失
        await self._ready.wait()
        # 忽略自己发的消息
        self_id = str(event.get("self_id", ""))
        sender_id = str(event.get("sender", {}).get("user_id", ""))
        if self_id and sender_id == self_id:
            return

        if not self._on_message:
            logger.debug("未注册消息处理器，忽略消息")
            return

        conv_id = get_conversation_id(event)

        try:
            await self._on_message(event, conv_id)
        except Exception:
            logger.exception("处理 NapCat 消息时异常 (conv=%s)", conv_id)

    async def _handle_meta(self, data: dict) -> None:
        """处理元事件（心跳等）。"""
        meta_type = data.get("meta_event_type", "")
        if meta_type == "heartbeat":
            logger.debug("NapCat 心跳 ♥")
        elif meta_type == "lifecycle":
            sub = data.get("sub_type", "")
            logger.info("NapCat 生命周期: %s", sub)
            if sub == "connect":
                async def _run_connect() -> None:
                    if self._on_connect:
                        await self._on_connect()
                    self._ready.set()
                    logger.info("NapCat 就绪，开始处理消息")
                asyncio.create_task(_run_connect())
