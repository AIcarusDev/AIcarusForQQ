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

PROMPT_SIGNATURE = """
// 对一个具体问题或情境进行深度思考，返回进一步的思考内容。
// 可指定思考出发点 intent 以锚定不同认知方向；不指定则随机选择。
// 适用场合示例：
// - 当你感到纠结、不确定如何行动时；或任何你觉得需要"想一想"，或思考一下的情景。
// 可用 intent 及大致方向：
// - affirmation: 寻求自我认同、自嗨、自我鼓励、意淫
// - criticism: 寻求自我批判、质疑、反思
// - solving: 寻求问题解决方案、分析对策
// - inspiration: 寻求灵感、发散、想象力
// - simulate: 模拟推演、预演事态发展（如果...会怎样？）
think_deeply(args: {
  content: string; // 需要深入思考的问题、情境或命题，用第一视角自然语言描述
  intent?: "affirmation" | "criticism" | "solving" | "inspiration" | "simulate"; // 思考的出发点/认知模式，不填则随机选择
})
"""

REQUIRES_CONTEXT: list[str] = ["session"]


def condition(config: dict) -> bool:
    return config.get("slow_thinking", {}).get("enabled", True)


def make_handler(session):
    def execute(content: str, intent: str | None = None, **kwargs) -> dict:
        chosen = intent if intent else random.choice(INTENTS)
        result = call_inner_voice(chosen, content, session)
        return {"intent": chosen, "result": result}

    return execute
