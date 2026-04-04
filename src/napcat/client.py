"""napcat/client.py — NapCat WebSocket 客户端

作为 WebSocket Server，等待 NapCat 主动连接（反向 WS）。
负责：连接管理、NapCat API 调用、消息发送、打字延迟模拟。
"""

import asyncio
import json
import logging
import random
import uuid
from typing import Any, Callable, Coroutine

import websockets
from websockets.asyncio.server import ServerConnection
from websockets.protocol import State as WsState
from pypinyin import pinyin, Style

from .segments import napcat_segments_to_text
from .events import get_conversation_id

logger = logging.getLogger("AICQ.napcat")


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
        self._on_recall: Callable[[dict], Coroutine] | None = None
        self._on_poke: Callable[[dict], Coroutine] | None = None
        # 同步完成前阻塞消息分发
        self._ready: asyncio.Event = asyncio.Event()
        # 同会话消息串行锁：防止并发处理导致消息乱序 / 图片竞态
        self._conv_locks: dict[str, asyncio.Lock] = {}
        # 主事件循环引用（start() 后设置），供工具函数在线程中跨线程调用 async API 使用
        self._loop: asyncio.AbstractEventLoop | None = None
        # 等待 message_sent 事件确认投递的 Future 表：key 为 message_id 字符串
        self._pending_sent: dict[str, asyncio.Future] = {}

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.state is WsState.OPEN

    def set_message_handler(
        self,
        handler: Callable[[dict, str], Coroutine],
    ) -> None:
        """注册消息处理回调: async def handler(event: dict, conversation_id: str)"""
        self._on_message = handler

    def set_recall_handler(
        self,
        handler: Callable[[dict], Coroutine],
    ) -> None:
        """注册撤回通知回调: async def handler(event: dict)"""
        self._on_recall = handler

    def set_poke_handler(
        self,
        handler: Callable[[dict], Coroutine],
    ) -> None:
        """注册戳一戳通知回调: async def handler(event: dict)"""
        self._on_poke = handler

    def set_connect_handler(
        self,
        handler: Callable[[], Coroutine],
    ) -> None:
        """注册 NapCat 连接就绪回调: async def handler()"""
        self._on_connect = handler

    async def start(self, host: str = "127.0.0.1", port: int = 8078) -> None:
        """启动 WebSocket 服务器，等待 NapCat 连接。"""
        self._loop = asyncio.get_running_loop()
        self._server = await websockets.serve(  # nosec B112 - localhost-only server, WSS unnecessary
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
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
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

    async def send_api_raw(
        self,
        action: str,
        params: dict,
        timeout: float = 15.0,
    ) -> dict | None:
        """与 send_api 相同，但返回完整响应 dict（含 status/message/data），
        适用于 data 为 null 时需要通过 status 判断成功与否的 API。"""
        if not self.connected:
            logger.warning("NapCat 未连接，无法调用 API: %s", action)
            return None

        echo = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._api_futures[echo] = fut

        payload = json.dumps({"action": action, "params": params, "echo": echo})
        assert self._ws is not None
        await self._ws.send(payload)
        logger.debug("→ NapCat API (raw): %s params=%s echo=%s", action, params, echo[:8])

        try:
            resp = await asyncio.wait_for(fut, timeout)
            if resp.get("status") != "ok":
                logger.warning(
                    "NapCat API %s 失败: status=%s msg=%s",
                    action, resp.get("status"), resp.get("message", ""),
                )
            return resp
        except TimeoutError:
            self._api_futures.pop(echo, None)
            logger.error("NapCat API %s 超时 (%ss)", action, timeout)
            return None

    def _calculate_typing_delay(self, text: str) -> float:
        """计算模拟打字延迟。"""
        import app_state  # 延迟导入，避免模块加载时循环引用
        _KEY_DELAY_MIN = 0.03
        _KEY_DELAY_MAX = 0.12
        _CHAR_SELECTION_DELAY_MIN = 0.08
        _CHAR_SELECTION_DELAY_MAX = 0.15
        _SPACE_PAUSE = 0.1
        _PUNCTUATION_PAUSE_MIN = 0.2
        _PUNCTUATION_PAUSE_MAX = 0.45
        _PUNCTUATION_TO_PAUSE = "，。！？；、,."
        _INITIAL_THINKING_MIN = 0.15
        _INITIAL_THINKING_MAX = 0.4
        _MAX_TOTAL_DELAY = 20.0

        if not text:
            return 0.05

        total_delay = random.uniform(_INITIAL_THINKING_MIN, _INITIAL_THINKING_MAX)
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                try:
                    p_list = pinyin(char, style=Style.NORMAL)
                    p_str = p_list[0][0]
                    for _ in p_str:
                        total_delay += random.uniform(_KEY_DELAY_MIN, _KEY_DELAY_MAX)
                    total_delay += random.uniform(
                        _CHAR_SELECTION_DELAY_MIN, _CHAR_SELECTION_DELAY_MAX
                    )
                except IndexError:
                    total_delay += 0.2
            elif "a" <= char.lower() <= "z":
                total_delay += random.uniform(_KEY_DELAY_MIN, _KEY_DELAY_MAX)
            elif char in _PUNCTUATION_TO_PAUSE:
                total_delay += random.uniform(_PUNCTUATION_PAUSE_MIN, _PUNCTUATION_PAUSE_MAX)
            elif char.isspace():
                total_delay += _SPACE_PAUSE
            else:
                total_delay += random.uniform(_KEY_DELAY_MIN, _KEY_DELAY_MAX)

        if len(text) > 10 and random.random() < 0.15:
            total_delay += random.uniform(0.5, 1.2)

        speed = float(app_state.config.get("typing_speed", 1.0))
        if speed <= 0:
            speed = 1.0
        return min(total_delay, _MAX_TOTAL_DELAY) / speed

    async def send_message(
        self,
        *,
        group_id: int | str | None = None,
        user_id: int | str | None = None,
        message: list[dict],
        llm_elapsed: float = 0.0,
    ) -> dict | None:
        """发送消息的快捷方法。"""
        _MIN_DELAY_TO_APPLY = 0.1

        text_content = napcat_segments_to_text(message)
        delay = max(0.0, self._calculate_typing_delay(text_content) - llm_elapsed)
        if delay > _MIN_DELAY_TO_APPLY:
            logger.debug(f"模拟打字延迟: {delay:.2f}s (len={len(text_content)}, llm={llm_elapsed:.2f}s)")
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

        result = await self.send_api("send_msg", params)

        # NapCat 对含 base64 图片的 send_msg 会在图片上传完成前就返回 echo，
        # 若此时立刻发送下一条消息，后续纯文本会先到达 QQ，造成消息乱序。
        # 等待 NapCat 推送 message_sent 事件，确认消息真正投递后再返回。
        _has_base64_image = any(
            seg.get("type") == "image"
            and str(seg.get("data", {}).get("file", "")).startswith("base64://")
            for seg in message
        )
        if _has_base64_image and result and result.get("message_id") is not None:
            msg_id = str(result["message_id"])
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            self._pending_sent[msg_id] = fut
            try:
                await asyncio.wait_for(asyncio.shield(fut), timeout=10.0)
                logger.debug("sticker 投递已确认 message_id=%s", msg_id)
            except asyncio.TimeoutError:
                self._pending_sent.pop(msg_id, None)
                logger.warning("等待 sticker 投递确认超时 message_id=%s，继续发送", msg_id)

        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    async def _connection_handler(self, ws: ServerConnection) -> None:
        """NapCat 连接进来时的主处理循环。"""
        logger.info("NapCat 已连接: %s", ws.remote_address)
        self._ws = ws
        self._ready.clear()  # 断线重连时重置，等新一轮同步完成

        # 直接从握手 header 读取 bot QQ 号，避免竞态
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
                    asyncio.create_task(self._dispatch_message_serial(data))
                elif post_type == "meta_event":
                    await self._handle_meta(data)
                elif post_type == "notice":
                    notice_type = data.get("notice_type", "")
                    logger.debug("NapCat 通知: %s", notice_type)
                    if notice_type in ("group_recall", "friend_recall") and self._on_recall:
                        asyncio.create_task(self._on_recall(data))
                    elif (notice_type == "notify"
                          and data.get("sub_type") == "poke"
                          and self._on_poke):
                        asyncio.create_task(self._on_poke(data))
                elif post_type == "message_sent":
                    # NapCat 在消息真正投递到 QQ 后推送此事件
                    sent_msg_id = str(data.get("message_id", ""))
                    if sent_msg_id:
                        fut = self._pending_sent.pop(sent_msg_id, None)
                        if fut and not fut.done():
                            fut.set_result(True)
                # request 等直接忽略

        except websockets.ConnectionClosed:
            logger.info("NapCat 连接已断开")
        finally:
            self._ws = None
            self.bot_id = None
            self._ready.clear()
            self._conv_locks.clear()

    async def _dispatch_message_serial(self, event: dict) -> None:
        """串行分发：同会话消息按到达顺序依次处理，防止图片下载等异步操作导致竞态。"""
        # 等待初始化同步完成，避免 prompt 中信息缺失
        await self._ready.wait()
        # 忽略自己发的消息
        self_id = str(event.get("self_id", ""))
        sender_id = str(event.get("sender", {}).get("user_id", ""))
        if self_id and sender_id == self_id:
            # NapCat 可能以普通 message 而非 message_sent 上报自己的消息，
            # 在此解析投递确认，避免 _pending_sent 等待超时
            msg_id = str(event.get("message_id", ""))
            if msg_id:
                fut = self._pending_sent.pop(msg_id, None)
                if fut and not fut.done():
                    fut.set_result(True)
                    logger.debug("self-message 触发 sticker 投递确认 message_id=%s", msg_id)
            return

        if not self._on_message:
            logger.debug("未注册消息处理器，忽略消息")
            return

        conv_id = get_conversation_id(event)
        if conv_id not in self._conv_locks:
            self._conv_locks[conv_id] = asyncio.Lock()

        async with self._conv_locks[conv_id]:
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
