"""get_self_image.py — 获取机器人自身外观图片

条件启用：仅在 config["vision"] 为 True（默认）时加载。
"""

import logging
from pathlib import Path

logger = logging.getLogger("AICQ.tools")

# self_image 目录位于项目根目录的 data/ 下，需要从 tools/ 向上跳三级到达项目根
_SELF_IMAGE_DIR = Path(__file__).parent.parent.parent / "data" / "self_image"

_IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

DECLARATION: dict = {
    "max_calls_per_response": 1,
    "name": "get_self_image",
    "description": (
        "获取你自身的外观形象图片。"
        "返回内容仅自己可见。"
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


def condition(config: dict) -> bool:
    return config.get("vision", True)


def execute(**kwargs) -> dict:
    if not _SELF_IMAGE_DIR.is_dir():
        logger.warning("[tools] get_self_image: self_image 目录不存在")
        return {"error": "self_image 目录不存在"}

    logger.info("[tools] get_self_image: 开始搜索图片")
    images_found = []
    for f in sorted(_SELF_IMAGE_DIR.iterdir()):
        ext = f.suffix.lower()
        if ext in _IMAGE_EXTENSIONS and f.is_file():
            images_found.append(f)

    if not images_found:
        logger.warning("[tools] get_self_image: 未找到支持的图片文件")
        return {"error": "self_image 目录下没有找到支持的图片文件（支持 png/jpg/jpeg/webp）"}

    multimodal_parts = []
    for img_path in images_found:
        ext = img_path.suffix.lower()
        mime = _IMAGE_EXTENSIONS[ext]
        try:
            data = img_path.read_bytes()
        except Exception as e:
            logger.warning("[tools] get_self_image: 读取图片失败 path=%s — %s", img_path, e)
            continue
        multimodal_parts.append({
            "mime_type": mime,
            "display_name": img_path.name,
            "data": data,
        })

    if not multimodal_parts:
        logger.warning("[tools] get_self_image: 所有图片读取失败")
        return {"error": "所有图片读取均失败"}

    logger.info("[tools] get_self_image: 获取成功 image_count=%d", len(multimodal_parts))
    return {
        "image_count": len(multimodal_parts),
        "image_names": [mp["display_name"] for mp in multimodal_parts],
        "_multimodal_parts": multimodal_parts,
    }
