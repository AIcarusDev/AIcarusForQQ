"""get_device_info.py — 获取设备基本信息（操作系统、RAM、GPU 显存）"""

import logging
import platform
import subprocess

import psutil

logger = logging.getLogger("AICQ.tools")

DECLARATION: dict = {
    "name": "get_device_info",
    "description": (
        "获取当前运行设备的粗略基本信息，"
        "包括操作系统版本、内存（RAM）使用情况和 GPU 显存情况。"
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


def execute(**kwargs) -> dict:
    logger.info("[tools] get_device_info: 开始收集设备信息")
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
    logger.info("[tools] get_device_info: 收集完成")
    return info
