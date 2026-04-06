"""wait.py — wait 工具实现"""

DECLARATION: dict = {
    "name": "wait",
    "description": (
        "等待一段时间或等待新消息到达后继续。"
        "用于需要等待对方回复、或暂缓决策，但不想结束当前激活的场景。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "timeout": {
                "type": "integer",
                "minimum": 1,
                "maximum": 300,
                "description": "最长等待秒数。",
            },
            "early_trigger": {
                "type": "string",
                "enum": ["new_message", "mentioned"],
                "description": "提前唤醒条件（可选）：new_message=有任何新消息，mentioned=被@或被回复。",
            },
            "motivation": {
                "type": "string",
                "description": "等待的原因。",
            },
        },
        "required": ["timeout", "motivation"],
    },
}

RESULT_MAX_CHARS: int = 0


def execute(timeout: int, motivation: str, early_trigger: str | None = None, **kwargs) -> dict:
    return {"ok": True}
