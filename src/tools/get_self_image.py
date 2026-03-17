"""get_self_image.py — 获取机器人自身外观图片

条件启用：仅在 config["vision"] 为 True（默认）时加载。
"""

import logging
from pathlib import Path

logger = logging.getLogger("AICQ.tools")

# self_image 目录与 src/ 同级子目录，Path.parent.parent 向上跳出 tools/ 回到 src/
_SELF_IMAGE_DIR = Path(__file__).parent.parent / "self_image"

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
        "当你需要通过视觉准确了解自己长什么样、想看看自己的外观、"
        "或被问及自身形象时可以调用此工具。"
        "返回内容仅自己可见。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
                "description": "调用此工具的动机或原因。",
            },
        },
    },
}


def condition(config: dict) -> bool:
    return config.get("vision", True)


def execute(**kwargs) -> dict:
    if not _SELF_IMAGE_DIR.is_dir():
        return {"error": "self_image 目录不存在"}

    images_found = []
    for f in sorted(_SELF_IMAGE_DIR.iterdir()):
        ext = f.suffix.lower()
        if ext in _IMAGE_EXTENSIONS and f.is_file():
            images_found.append(f)

    if not images_found:
        return {"error": "self_image 目录下没有找到支持的图片文件（支持 png/jpg/jpeg/webp）"}

    multimodal_parts = []
    for img_path in images_found:
        ext = img_path.suffix.lower()
        mime = _IMAGE_EXTENSIONS[ext]
        try:
            data = img_path.read_bytes()
        except Exception as e:
            logger.warning("[tools] 读取图片 %s 失败: %s", img_path, e)
            continue
        multimodal_parts.append({
            "mime_type": mime,
            "display_name": img_path.name,
            "data": data,
        })

    if not multimodal_parts:
        return {"error": "所有图片读取均失败"}

    return {
        "image_count": len(multimodal_parts),
        "image_names": [mp["display_name"] for mp in multimodal_parts],
        "_multimodal_parts": multimodal_parts,
    }
