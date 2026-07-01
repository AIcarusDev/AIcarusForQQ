"""wait.py - short fuzzy wait tool owned by core."""

import logging
import time
from typing import Any

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools.wait")

TOOL_KIND = "passive_wait"

DECLARATION: dict = {
    "name": "wait",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "integer",
                "minimum": 1,
                "maximum": 15,
                "description": "等待秒数，范围 1~15。",
            },
        },
        "required": ["seconds"],
    },
}

PROMPT_SIGNATURE = """
// 核心的通用短等待工具。
// 只等待一小段时间，然后进入下一轮观察。
// 例如在社交平台上看见对方还在叙事，话还没说完；或等待浏览器页面加载、图片加载等等，大多数情况都可用。
wait(args: {
  seconds: number; // 等待秒数，范围 1~15。
})
"""


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(args, dict):
        return args, []
    repaired = dict(args)
    changes: list[str] = []
    if "seconds" not in repaired and "timeout" in repaired:
        repaired["seconds"] = repaired.pop("timeout")
        changes.append("timeout -> seconds")
    if isinstance(repaired.get("seconds"), str):
        stripped_seconds = str(repaired["seconds"]).strip()
        if stripped_seconds.isdigit():
            repaired["seconds"] = int(stripped_seconds)
            changes.append("seconds: string -> int")
    return repaired, changes


def execute(seconds: int, **_kwargs) -> dict:
    timeout_secs = min(15, max(1, int(seconds)))
    started_at = time.time()
    try:
        time.sleep(timeout_secs)
    except KeyboardInterrupt:
        logger.info("[wait] 短等待被外部中断")
        return {"ok": False, "error": "wait 中断：进程被外部关闭"}

    elapsed = round(time.time() - started_at, 1)
    logger.info("[wait] 短等待完成 elapsed=%ss", elapsed)
    return {
        "ok": True,
        "resumed": "timeout",
        "elapsed_seconds": elapsed,
    }
