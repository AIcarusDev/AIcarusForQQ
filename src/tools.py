"""tools.py — 自定义工具定义与注册表

所有可供 LLM 调用的自定义工具函数集中于此。
每个工具声明中包含 max_calls_per_response 字段，
用于控制单次回复中该工具的最大调用次数。
"""

import logging
import platform
import subprocess
from pathlib import Path

import psutil

logger = logging.getLogger("AICQ.tools")

# self_image 目录路径
_SELF_IMAGE_DIR = Path(__file__).parent / "self_image"

# 支持的图片 MIME 类型（Gemini 3 多模态函数响应）
_IMAGE_EXTENSIONS: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def get_device_info() -> dict:
    """获取设备基本信息：操作系统、内存（RAM）使用情况、GPU 显存情况。"""
    info: dict = {
        "os": f"{platform.system()} {platform.version()}",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
    }
    parts = [f"{platform.system()} {platform.version()} ({platform.machine()})"]

    try:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024 ** 3), 1)
        info["ram_available_gb"] = round(vm.available / (1024 ** 3), 1)
        info["ram_used_percent"] = vm.percent
        parts.append(
            f"RAM {info['ram_total_gb']}GB 总计 / "
            f"{info['ram_available_gb']}GB 可用 ({vm.percent}% 已用)"
        )
    except Exception:
        pass

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            gpus = []
            for line in proc.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) == 3:
                    gpus.append({
                        "name": p[0],
                        "vram_total_mb": int(p[1]),
                        "vram_free_mb": int(p[2]),
                    })
                    parts.append(
                        f"GPU {p[0]} 显存 {p[1]}MB 总计 / {p[2]}MB 空闲"
                    )
            if gpus:
                info["gpus"] = gpus
    except Exception:
        pass

    info["summary"] = "；".join(parts)
    return info


# ── 工具声明（Gemini 原生格式） ──────────────────────────
#
# 扩展字段:
#   max_calls_per_response — 单次回复中该工具的最大调用次数
#                            provider 会读取此字段并在工具循环中
#                            按工具粒度追踪剩余配额
#
# GeminiAdapter 直接使用此格式作为 function_declarations；
# OpenAICompatAdapter 会自动转为 OpenAI 的 {type, function} 包装格式。

def get_self_image() -> dict:
    """从 self_image 目录读取机器人的外观图片，返回多模态数据。"""
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
        display_name = img_path.name
        try:
            data = img_path.read_bytes()
        except Exception as e:
            logger.warning("[tools] 读取图片 %s 失败: %s", img_path, e)
            continue
        multimodal_parts.append({
            "mime_type": mime,
            "display_name": display_name,
            "data": data,
        })

    if not multimodal_parts:
        return {"error": "所有图片读取均失败"}

    return {
        "image_count": len(multimodal_parts),
        "image_names": [mp["display_name"] for mp in multimodal_parts],
        "_multimodal_parts": multimodal_parts,
    }


TOOL_DECLARATIONS = [
    {
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
    },
    {
        "max_calls_per_response": 1,
        "name": "get_device_info",
        "description": (
            "获取当前运行设备的粗略基本信息，"
            "包括操作系统版本、内存（RAM）使用情况和 GPU 显存情况。"
            "返回内容仅自己可见，若不主动透露则无法被他人知晓。"
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
    },
]

TOOL_REGISTRY: dict = {
    "get_self_image": get_self_image,
    "get_device_info": get_device_info,
}
