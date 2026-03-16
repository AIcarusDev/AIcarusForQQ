"""tools.py — 自定义工具定义与注册表

所有可供 LLM 调用的自定义工具函数集中于此。
每个工具声明中包含 max_calls_per_response 字段，
用于控制单次回复中该工具的最大调用次数。
"""

import asyncio
import logging
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable

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


# ── 群成员列表工具（需运行时注入上下文，不进入全局 TOOL_REGISTRY） ──────────────

def make_get_group_members_tool(napcat_client: Any, group_id: str) -> Callable:
    """为特定群聊会话创建 get_group_members 工具函数。

    返回的函数是同步的，内部通过 run_coroutine_threadsafe 跨线程
    调用 NapCat 异步 API，适合在 asyncio.to_thread 的工作线程中使用。
    """
    def get_group_members() -> dict:
        if not napcat_client or not napcat_client.connected:
            return {"error": "NapCat 未连接，无法获取群成员列表"}
        loop: asyncio.AbstractEventLoop | None = napcat_client._loop
        if loop is None or not loop.is_running():
            return {"error": "主事件循环不可用"}
        try:
            coro = napcat_client.send_api(
                "get_group_member_list",
                {"group_id": int(group_id)},
            )
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            raw: list[dict] | None = future.result(timeout=15)
        except Exception as e:
            return {"error": f"获取群成员列表失败: {e}"}

        if raw is None:
            return {"error": "API 返回为空（可能群号有误或权限不足）"}

        # 最多取前 20 条，防止 token 爆炸
        members_raw = raw[:20]
        members = []
        for m in members_raw:
            qq_id = str(m.get("user_id", ""))
            nickname = m.get("nickname", "")
            card = m.get("card", "") or nickname  # 群名片为空时回退到昵称
            members.append({"id": qq_id, "name": nickname, "card": card})

        return {
            "group_id": group_id,
            "total_in_group": len(raw),
            "returned": len(members),
            "note": "最多返回前 20 条，超出部分已截断",
            "members": members,
        }

    return get_group_members


GET_GROUP_MEMBERS_DECLARATION: dict = {
    "max_calls_per_response": 1,
    "name": "get_group_members",
    "description": (
        "获取当前群聊的成员列表（仅群聊会话中可用）。"
        "返回每位成员的 QQ 号（id）、QQ 昵称（name）和群名片（card）。"
        "最多返回前 20 条记录。"
        "当你需要知道群里有哪些人、查找某人的 QQ 号或群名片时可以调用。"
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
