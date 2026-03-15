"""schema.py — LLM 结构化输出 JSON Schema 定义"""

RESPONSE_SCHEMA = {
    "type": "object",
    "description": "你的内心状态与要发送的消息。",
    "properties": {
        "mood": {
            "type": "string",
            "description": "你当前的情绪，是下意识的第一反应。",
        },
        "think": {
            "type": "string",
            "description": "你当前的内心想法，是自然真实且私密的，可以简短，也可以是非常丰富深度思考。",
        },
        "intent": {
            "type": "string",
            "description": "你当前最直接的、短期的意图或打算。",
        },
        "messages": {
            "type": "array",
            "description": (
                "要发送的消息列表。数组中的每一个元素代表一条独立发送的消息。"
                "长消息可以分为多个 segment 发送。通常情况下建议：一条消息控制在 10 字以下，"
                "消息中的标点符号均可省略，并且自然口语化。"
                "如果不需要发送消息，则不需要输出此项。"
            ),
            "items": {
                "type": "object",
                "description": "单条消息的结构",
                "properties": {
                    "reply_message_id": {
                        "type": ["string", "null"],
                        "description": (
                            "要引用回复的目标消息ID。"
                            "仅在需要明确上下文或特别提醒时使用，不要滥用。"
                        ),
                    },
                    "segments": {
                        "type": "array",
                        "description": "该条消息的具体内容片段（文本或@某人）",
                        "items": {
                            "oneOf": [
                                {
                                    "title": "@某人",
                                    "type": "object",
                                    "properties": {
                                        "command": {
                                            "type": "string",
                                            "enum": ["at"],
                                        },
                                        "params": {
                                            "type": "object",
                                            "properties": {
                                                "user_id": {
                                                    "type": "string",
                                                    "description": "被 @ 用户的 ID",
                                                }
                                            },
                                            "required": ["user_id"],
                                        },
                                    },
                                    "required": ["command", "params"],
                                },
                                {
                                    "title": "文本",
                                    "type": "object",
                                    "properties": {
                                        "command": {
                                            "type": "string",
                                            "enum": ["text"],
                                        },
                                        "params": {
                                            "type": "object",
                                            "properties": {
                                                "content": {
                                                    "type": "string",
                                                    "description": (
                                                        "文本内容，建议控制长度（例如 10 字以下），"
                                                        "建议省略标点符号，省略主语。"
                                                    ),
                                                }
                                            },
                                            "required": ["content"],
                                        },
                                    },
                                    "required": ["command", "params"],
                                },
                            ]
                        },
                    },
                },
                "required": ["segments"],
            },
        },
        "motivation": {
            "type": "string",
            "description": "你发送消息或选择不发送消息的原因。",
        },
        "cycle_action": {
            "type": "string",
            "enum": ["continue", "stop"],
            "description": (
                "循环管理。"
                "'continue'：在本轮所有消息确认发出并同步后，立即激活下一轮循环（消耗一次剩余循环次数）。"
                "'stop'：结束当前循环，等待被动激活。"
                "当剩余连续循环次数为 0 时，你必须选择 'stop'。"
            ),
        },
    },
    "required": ["mood", "think", "intent", "motivation", "cycle_action"],
}
