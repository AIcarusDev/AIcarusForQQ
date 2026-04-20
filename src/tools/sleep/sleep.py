"""sleep.py — sleep 工具实现"""

from .prompt import DESCRIPTION

DECLARATION: dict = {
    "name": "sleep",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "duration": {
                "type": "integer",
                "minimum": 30,
                "maximum": 600,
                "description": "想睡多久？单位分钟，范围 30~600。",
            },
            "motivation": {
                "type": "string",
            },
        },
        "required": ["duration", "motivation"],
    },
}


def execute(duration: int, motivation: str, **kwargs) -> dict:
    return {"deferred": True}
