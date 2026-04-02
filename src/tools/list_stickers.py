"""list_stickers.py — 查看自己的表情包收藏"""

import logging
from llm.media.sticker_collection import MAX_STICKERS, get_sticker_grid_bytes, list_all

logger = logging.getLogger("AICQ.tools")

DECLARATION = {
    "name": "list_stickers",
    "description": (
        "查看自己收藏的表情包列表，返回每个表情包的 ID 和应用场景描述。"
        "对于支持多模态的模型，还会附带一张包含所有表情包的网格预览图，图中每格下方标有 ID。"
        "发送表情包前先调用此工具确认 ID。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
            },
        },
        "required": ["motivation"],
    },
}

# 需要 config 以判断是否为视觉模型，需要 provider 来判断是 Gemini 还是 OpenAI 兼容
REQUIRES_CONTEXT: list[str] = ["config", "provider"]


def make_handler(config: dict, provider: str):
    """工厂函数：绑定 config 和 provider，返回工具处理函数。"""
    vision_enabled: bool = config.get("vision", True)
    is_gemini: bool = provider == "gemini"

    def handler(motivation: str = "", **_) -> dict:

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

        if len(stickers) >= MAX_STICKERS:
            result["note"] = (
                f"表情包收藏已满（上限 {MAX_STICKERS} 个），"
                "如需添加新表情包，请先移除一些旧的。"
            )

        if vision_enabled:
            grid_bytes = get_sticker_grid_bytes()
            if grid_bytes:
                if is_gemini:
                    result["_multimodal_parts"] = [
                        {
                            "mime_type": "image/jpeg",
                            "display_name": "stickers_grid",
                            "data": grid_bytes,
                        }
                    ]
                else:
                    import base64
                    result["_multimodal_parts"] = [
                        {
                            "mime_type": "image/jpeg",
                            "display_name": "stickers_grid",
                            "data": base64.b64encode(grid_bytes).decode("utf-8"),
                        }
                    ]

        logger.info(
            "[tools] list_stickers: 返回 %d 个表情包, vision=%s, provider=%s",
            len(stickers), vision_enabled, provider,
        )
        return result

    return handler
