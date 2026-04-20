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
                "maximum": 600,
                "description": "最长等待秒数。",
            },
            "early_trigger": {
                "type": "object",
                "description": "提前唤醒条件。",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["session", "global"],
                        "description": "监听范围：session=仅当前会话有新消息时触发，global=任意会话的消息均可触发。",
                    },
                    "condition": {
                        "type": "string",
                        "enum": ["any_message", "mentioned"],
                        "description": "触发条件：any_message=有任何新消息，mentioned=被@或被回复。",
                    },
                },
                "required": ["scope", "condition"],
            },
            "motivation": {
                "type": "string"
            },
        },
        "required": ["timeout", "motivation", "early_trigger"],
    },
}


def execute(timeout: int, motivation: str, early_trigger: dict, **kwargs) -> dict:
    return {"deferred": True}
