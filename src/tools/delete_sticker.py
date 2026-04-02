"""delete_sticker.py — 从收藏中删除一个表情包"""

import logging
from llm.media.sticker_collection import delete_sticker

logger = logging.getLogger("AICQ.tools")

DECLARATION = {
    "name": "delete_sticker",
    "description": (
        "从自己的表情包收藏中删除指定 ID 的表情包。"
        "删除后剩余表情包会自动补位重编号（例如删除 001 后，002 变为 001，003 变为 002）。"
        "如果不确定 ID，请先调用 list_stickers 查看。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sticker_id": {
                "type": "string",
                "description": "要删除的表情包 ID（三位数字字符串，如 '003'）",
            },
        },
        "required": ["sticker_id"],
    },
}


def execute(sticker_id: str, **_) -> dict:

    success = delete_sticker(sticker_id)
    if not success:
        logger.warning("[tools] delete_sticker: ID 不存在 id=%s", sticker_id)
        return {"error": f"表情包 ID \"{sticker_id}\" 不存在，删除失败。"}

    logger.info("[tools] delete_sticker: 已删除 id=%s", sticker_id)
    return {"deleted_id": sticker_id, "message": f"表情包 \"{sticker_id}\" 已成功删除。"}
