"""list_stickers.py — 查看自己的表情包收藏"""

import logging
from llm.media.sticker_collection import MAX_STICKERS, StickerCollectionError, get_sticker_snapshot

from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class ListStickersArgs(ToolArgsModel):
    pass


TOOL_CONTRACT = ToolContract(
    name="list_stickers",
    description=(
        "查看收藏的表情包列表，可见 image_ref 和应用场景描述。"
        "支持多模态的模型还会收到网格预览图，每格标注对应表情包的 image_ref。"
        "发送表情包前先调用此工具确认 ref。"
    ),
    args_model=ListStickersArgs,
)

# 需要 config 以判断是否为视觉模型。
REQUIRES_CONTEXT: list[str] = ["config"]
PARALLEL_SAFE = True
PARALLEL_KEY = "sticker_read"


def make_handler(config: dict):
    """工厂函数：绑定 config，返回工具处理函数。"""
    vision_enabled: bool = config.get("vision", True)

    def handler(**_) -> dict:

        try:
            stickers, grid_bytes = get_sticker_snapshot(include_grid=vision_enabled)
        except StickerCollectionError as exc:
            return {"error": str(exc), "code": exc.code}
        if not stickers:
            return {"count": 0, "stickers": [], "message": "暂无已收藏的表情包。"}

        result: dict = {
            "count": len(stickers),
            "stickers": [
                {"image_ref": s["image_ref"], "description": s["description"]}
                for s in stickers
            ],
        }

        if len(stickers) >= MAX_STICKERS:
            result["note"] = (
                f"表情包收藏已满（上限 {MAX_STICKERS} 个），"
                "如需添加新表情包，请先移除一些旧的。"
            )

        if vision_enabled:
            if grid_bytes:
                result["_multimodal_parts"] = [
                    {
                        "mime_type": "image/jpeg",
                        "display_name": "stickers_grid",
                        "data": grid_bytes,
                    }
                ]

        logger.info(
            "[tools] list_stickers: 返回 %d 个表情包, vision=%s",
            len(stickers), vision_enabled,
        )
        return result

    return handler
