"""save_sticker.py — 将聊天记录中的图片/动画表情收藏为自己的表情包"""

import base64
import logging

logger = logging.getLogger("AICQ.tools")

DECLARATION = {
    "max_calls_per_response": 3,
    "name": "save_sticker",
    "description": (
        "将聊天记录中的[动画表情]或[图片]保存到自己的表情包收藏中。"
        "调用前需从上下文中获取目标图片的 ref（12位十六进制字符串）。"
        "保存成功后会返回该表情包的 ID，之后可在消息的 segments 中用 sticker 指令通过 ID 发送。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_ref": {
                "type": "string",
                "description": (
                    "目标图片/表情的 ref，12位十六进制字符串"
                    "（来自上下文 XML 中的 ref 标注）"
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "描述这个表情包的适用场景，尽量具体，例如："
                    "'表达无语/沉默时发送' / '开心大笑时用' / '表示赞同时'"
                ),
            },
        },
        "required": ["image_ref", "description"],
    },
}

# 需要 session 以便在上下文中查找图片 ref
REQUIRES_CONTEXT: list[str] = ["session"]


def make_handler(session):
    """工厂函数：绑定 session，返回工具处理函数。"""

    def handler(image_ref: str, description: str, **_) -> dict:
        from llm.image_cache import read_image_bytes
        from llm.sticker_collection import save_sticker

        # ── 1. 在上下文中查找图片 ──────────────────────────────
        target_img: dict | None = None
        for entry in session.context_messages:
            images: dict = entry.get("images") or {}
            if image_ref in images:
                target_img = images[image_ref]
                break

        if target_img is None:
            logger.warning("[tools] save_sticker: 未找到图片 ref=%s", image_ref)
            return {
                "error": (
                    f"未在当前上下文中找到 ref={image_ref!r} 的图片。"
                    "请检查 ref 是否正确，或图片可能已超出上下文窗口。"
                )
            }

        # ── 2. 获取原始字节 ─────────────────────────────────────
        b64: str = target_img.get("base64", "")
        mime: str = target_img.get("mime", "image/jpeg")
        phash: str | None = target_img.get("phash")

        raw_bytes: bytes | None = None
        if b64:
            try:
                raw_bytes = base64.b64decode(b64)
            except Exception as e:
                logger.warning("[tools] save_sticker: base64 解码失败 ref=%s: %s", image_ref, e)

        if raw_bytes is None and phash:
            raw_bytes = read_image_bytes(phash)

        if raw_bytes is None:
            logger.warning("[tools] save_sticker: 图片数据不可用 ref=%s", image_ref)
            return {"error": "图片原始数据不可用（可能已被清理），无法保存"}

        # ── 3. 保存到收藏 ────────────────────────────────────────
        sticker_id = save_sticker(raw_bytes, mime, description)
        logger.info("[tools] save_sticker: 已保存 id=%s ref=%s", sticker_id, image_ref)
        return {
            "sticker_id": sticker_id,
            "description": description,
            "message": f"表情包已保存，ID 为 \"{sticker_id}\"，可在消息 segments 中使用 sticker 指令发送。",
        }

    return handler
