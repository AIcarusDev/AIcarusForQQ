"""examine_image.py — 定向精细观察对话中的图片

主模型主动调用此工具，指定 image_ref（12位十六进制 ref）和 focus，
VisionBridge 带焦点重新询问 VLM，结果写入内存和 sidecar。

启用条件：session 和 vision_bridge 均在运行时上下文中就绪。
"""

import logging

from llm.media.image_cache import read_image_b64

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "max_calls_per_response": 3,
    "name": "examine_image",
    "description": (
        "对对话中的某张图片进行定向精细观察。"
        "当你在上下文中看到 [图片] 标记，需要了解图片特定区域或细节时调用。"
        "调用前必须从上下文中获取图片的 ref（12位十六进制字符串）。"
        "每次调用聚焦一个具体问题，结果会在本轮对话中持续可用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_ref": {
                "type": "string",
                "description": (
                    "目标图片的 ref，12位十六进制字符串"
                    "（来自上下文 XML 中 <description> 或工具响应里标注的 ref）"
                ),
            },
            "focus": {
                "type": "string",
                "description": (
                    "本次重点观察的内容，尽量具体，例如："
                    "'右侧的报错文字' / '人物的面部表情' / '左上角的数字' / '图中的二维码'"
                ),
            },
            "motivation": {
                "type": "string",
                "description": "调用此工具的原因（可选，供日志记录）",
            },
        },
        "required": ["image_ref", "focus", "motivation"],
    },
}

# 需要 session（遍历上下文消息找 ref）和 vision_bridge（调用 VLM）
REQUIRES_CONTEXT: list[str] = ["session", "vision_bridge"]


def make_handler(session, vision_bridge):
    """工厂函数：绑定 session 和 vision_bridge，返回工具处理函数。"""

    def handler(image_ref: str, focus: str, motivation: str, **_) -> dict:
        # ── 1. 在上下文中查找包含该 ref 的图片 ──────────────────
        target_img: dict | None = None
        for entry in session.context_messages:
            images: dict = entry.get("images") or {}
            if image_ref in images:
                target_img = images[image_ref]
                break

        if target_img is None:
            logger.warning("[tools] examine_image: 未找到图片 ref=%s", image_ref)
            return {
                "error": (
                    f"未在当前上下文中找到 ref={image_ref!r} 的图片。"
                    "请检查 ref 是否正确，或图片可能已超出上下文窗口。"
                )
            }

        # ── 2. 取出 base64 数据 ──────────────────────────────────
        b64: str = target_img.get("base64", "")
        mime: str = target_img.get("mime", "image/jpeg")
        phash: str | None = target_img.get("phash")

        if not b64:
            # base64 丢失时尝试从磁盘恢复
            if phash:
                cached = read_image_b64(phash)
                if cached:
                    b64, mime = cached
            if not b64:
                logger.warning("[tools] examine_image: base64 数据丢失 ref=%s", image_ref)
                return {"error": "图片原始数据不可用（可能已被清理），无法精查"}

        if not vision_bridge.enabled:
            logger.warning("[tools] examine_image: VisionBridge 未启用")
            return {"error": "视觉桥（VisionBridge）未启用，无法进行图片精查"}

        # ── 3. 调用 VLM 精查 ─────────────────────────────────────
        logger.info("[tools] examine_image: 开始精查 focus=%r ref=%s", focus, image_ref)
        result_text = vision_bridge.examine(phash, b64, mime, focus)
        if result_text is None:
            logger.warning("[tools] examine_image: VLM 返回为空 ref=%s", image_ref)
            return {"error": "精查失败，VLM 未能返回有效结果，请稍后重试"}

        logger.info("[tools] examine_image: 精查完成 ref=%s", image_ref)
        # ── 4. 同步更新内存中的 examinations ─────────────────────
        if "examinations" not in target_img:
            target_img["examinations"] = []
        target_img["examinations"].append(
            {"focus": focus, "result": result_text}
        )

        logger.info(
            "[examine_image] ref=%s focus=%r motivation=%r", image_ref, focus, motivation
        )

        return {
            "image_ref": image_ref,
            "focus": focus,
            "result": result_text,
        }

    return handler
