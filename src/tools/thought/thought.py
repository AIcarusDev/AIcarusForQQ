"""thought.py — 内心想法工具

让模型在做出任何决策前先记录一段内心想法/推理过程。
Handler 无副作用，直接返回 ok。

RESULT_MAX_CHARS = 0：result 不写进摘要，但 args（思考内容）
在 _contents 里自然保留，模型可以回看。

与 native thinking 互斥：若 thought 工具在列，Provider 层会强制
关闭 thinking_config（见 GeminiAdapter.call）。
"""

from .prompt import DESCRIPTION

DECLARATION: dict = {
    "name": "thought",
    "description": DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "你当前的内心想法，是私密的、自然的心理活动。也可以进行推理思考",
            }
        },
        "required": ["content"],
    },
    "max_calls_per_response": 30,
}

# result 字段不写入摘要（=0），args 在 _contents 中自然保留
RESULT_MAX_CHARS: int = 0


def execute(content: str, **kwargs) -> dict:
    return {"ok": True}
