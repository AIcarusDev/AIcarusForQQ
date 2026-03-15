"""tools.py — 自定义工具定义与注册表

所有可供 LLM 调用的自定义工具函数集中于此。
每个工具声明中包含 max_calls_per_response 字段，
用于控制单次回复中该工具的最大调用次数。
"""

import platform
import subprocess

import psutil


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


# ── 工具声明（OpenAI function calling 格式） ─────────────
#
# 扩展字段:
#   max_calls_per_response — 单次回复中该工具的最大调用次数
#                            provider 会读取此字段并在工具循环中
#                            按工具粒度追踪剩余配额

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "max_calls_per_response": 1,
        "function": {
            "name": "get_device_info",
            "description": (
                "获取当前运行设备的基本信息，"
                "包括操作系统版本、内存（RAM）使用情况和 GPU 显存情况。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motivation": {
                        "type": "string",
                        "description": "调用此工具的动机或原因，简短说明为什么需要这些信息。",
                    },
                },
            },
        },
    },
]

TOOL_REGISTRY: dict = {
    "get_device_info": get_device_info,
}
