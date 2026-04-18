"""idle.py — 结束 activation，进入休眠/窥屏

Handler 无副作用，直接返回 ok。
Provider 检测到此工具被调用后立刻退出工具循环，并将
loop_action = {"action": "idle", "motivation": ...} 返回给上层。
上层（napcat_handler._run_active_loop）负责清空 adapter._contents 并调度 watcher。
"""

DECLARATION: dict = {
    "name": "idle",
    "description": (
        "结束当前 activation，进入休眠/窥屏状态。"
        "对于简单交互、话题自然结束、无需立即行动的情况使用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "motivation": {
                "type": "string",
            }
        },
        "required": ["motivation"],
    },
}


def execute(motivation: str, **kwargs) -> dict:
    return {"ok": True}
