"""scripts/test_alert_integration.py — AlertManager 集成自测

不依赖 NapCat，直接调用 AlertManager 发送两封邮件：
    1. 模拟心跳超时 → notify_disconnect
    2. 模拟恢复       → notify_recover
确认 AlertManager 状态机与 SMTP 通道在主程序代码路径上正确工作。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# 极简 .env 加载（避免对 python-dotenv 的依赖）
_root = Path(__file__).resolve().parent.parent
_env_file = _root / ".env"
if _env_file.is_file():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

# 让 src/ 可被导入
sys.path.insert(0, str(_root / "src"))

from alerting import AlertManager  # noqa: E402


async def main() -> None:
    mgr = AlertManager({
        "enabled": True,
        "cooldown": 5,
        "subject_prefix": "[AIcarus 告警-集成测试]",
    })
    print("[1/2] 发送模拟掉线告警 ...")
    await mgr.notify_disconnect("集成测试: 模拟心跳超时 (此为测试邮件，可忽略)")
    # 等一会模拟"恢复"
    await asyncio.sleep(1)
    print("[2/2] 发送模拟恢复告警 ...")
    await mgr.notify_recover()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
