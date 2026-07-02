"""wait.py - short fuzzy wait tool owned by core."""

import logging
import time
from typing import Any

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

from .prompt import DESCRIPTION

logger = logging.getLogger("AICQ.tools.wait")

TOOL_KIND = "passive_wait"

class WaitArgs(ToolArgsModel):
    seconds: int = Field(
        ge=1,
        le=15,
        description="等待秒数。",
    )


TOOL_CONTRACT = ToolContract(
    name="wait",
    description=DESCRIPTION,
    args_model=WaitArgs,
)


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
