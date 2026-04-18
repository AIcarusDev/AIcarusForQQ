"""wait.py — wait 工具实现"""

from .prompt import DESCRIPTION

DECLARATION: dict = {
    "name": "wait",
    "description": DESCRIPTION,
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
                "type": "string"
            },
        },
        "required": ["timeout", "motivation"],
    },
}


def execute(timeout: int, motivation: str, early_trigger: str | None = None, **kwargs) -> dict:
    return {"ok": True}
