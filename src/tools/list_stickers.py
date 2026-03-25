"""list_stickers.py — 查看自己的表情包收藏"""

import logging

logger = logging.getLogger("AICQ.tools")

DECLARATION = {
    "max_calls_per_response": 1,
    "name": "list_stickers",
    "description": (
        "查看自己收藏的表情包列表，返回每个表情包的 ID 和应用场景描述。"
        "对于支持多模态的模型，还会附带图片预览。"
        "发送表情包前先调用此工具确认 ID。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
                "description": "调用此工具的原因（可选）",
            },
        },
    },
}

# 需要 config 以判断是否为视觉模型
REQUIRES_CONTEXT: list[str] = ["config"]


def make_handler(config: dict):
    """工厂函数：绑定 config，返回工具处理函数。"""
    vision_enabled: bool = config.get("vision", True)

    def handler(motivation: str = "", **_) -> dict:
        from llm.sticker_collection import list_all, load_sticker_bytes

        stickers = list_all()
        if not stickers:
            return {"count": 0, "stickers": [], "message": "暂无已收藏的表情包。"}

        result: dict = {
            "count": len(stickers),
            "stickers": [
                {"id": s["id"], "description": s["description"]}
                for s in stickers
            ],
        }

        if vision_enabled:
            multimodal_parts = []
            for s in stickers:
                data = load_sticker_bytes(s["id"])
                if data is None:
                    continue
                raw_bytes, mime = data
                multimodal_parts.append({
                    "mime_type": mime,
                    "display_name": f"sticker_{s['id']}",
                    "data": raw_bytes,
                })
            if multimodal_parts:
                result["_multimodal_parts"] = multimodal_parts

        logger.info("[tools] list_stickers: 返回 %d 个表情包 vision=%s", len(stickers), vision_enabled)
        return result

    return handler
