"""update_sticker.py — 修改表情包的文字描述"""

import logging
from llm.media.sticker_collection import update_sticker_description

logger = logging.getLogger("AICQ.tools")

DECLARATION = {
    "max_calls_per_response": 5,
    "name": "update_sticker",
    "description": (
        "修改已收藏表情包的文字描述（适用场景说明）。"
        "当发现某个表情包的描述不够准确、太模糊，或想补充适用场景时使用。"
        "如果不确定 ID，请先调用 list_stickers 查看。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sticker_id": {
                "type": "string",
                "description": "要修改的表情包 ID（三位数字字符串，如 '003'）",
            },
            "description": {
                "type": "string",
                "description": (
                    "新的场景描述，尽量具体，例如："
                    "'表达无语/沉默时发送' / '开心大笑时用' / '表示赞同时'"
                ),
            },
        },
        "required": ["sticker_id", "description"],
    },
}


def execute(sticker_id: str, description: str, **_) -> dict:

    success = update_sticker_description(sticker_id, description)
    if not success:
        logger.warning("[tools] update_sticker: ID 不存在 id=%s", sticker_id)
        return {"error": f"表情包 ID \"{sticker_id}\" 不存在，修改失败。"}

    logger.info("[tools] update_sticker: 已更新描述 id=%s desc=%r", sticker_id, description)
    return {
        "sticker_id": sticker_id,
        "new_description": description,
        "message": f"表情包 \"{sticker_id}\" 的描述已更新。",
    }
