"""think_deeply.py — 深度思考工具实现。"""

import random

from llm.slow_thinking import INTENTS, call_inner_voice

from .prompt import DESCRIPTION

DECLARATION: dict = {
    "name": "think_deeply",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "思考的出发点/认知模式，不填则随机选择",
                "enum": [
                    "affirmation",
                    "criticism",
                    "solving",
                    "inspiration",
                    "simulate"
                ],
            },
            "content": {
                "type": "string",
                "description": "需要深入思考的问题、情境或命题，用第一视角自然语言描述",
            },
        },
        "required": ["content"],
    },
}

REQUIRES_CONTEXT: list[str] = ["session"]


def condition(config: dict) -> bool:
    return config.get("slow_thinking", {}).get("enabled", True)


def make_handler(session):
    def execute(content: str, intent: str | None = None, **kwargs) -> dict:
        chosen = intent if intent else random.choice(INTENTS)
        result = call_inner_voice(chosen, content, session)
        return {"intent": chosen, "result": result}

    return execute