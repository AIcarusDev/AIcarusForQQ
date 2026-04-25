"""记忆自动归档的结构化提取工具协议。"""

from typing import Any
from .archive_prompt import ARCHIVE_TOOL_PROMPT
from llm.core.internal_tool import InternalToolSpec


DECLARATION: dict[str, Any] = {
    "name": "archive_memories",
    "description": ARCHIVE_TOOL_PROMPT,
    "parameters": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "description": "Neo-Davidsonian 多角色事件列表。",
                "items": {
                    "type": "object",
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "description": "简短事件标签, 如 teaching/correcting/asking/sharing/liking/disliking/promising/experiencing。",
                        },
                        "summary": {
                            "type": "string",
                            "description": "一句话事件摘要 (<=30 字), 用于检索与渲染。",
                        },
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": "表达否定意图时用 negative, 不要塞进 event_type。",
                        },
                        "modality": {
                            "type": "string",
                            "enum": ["actual", "hypothetical", "possible"],
                            "description": "事实用 actual, 「如果」「可能」用对应值。",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0~1.0, 事实约 0.7, 推测约 0.4。",
                        },
                        "context_type": {
                            "type": "string",
                            "enum": ["meta", "contract", "episodic"],
                            "description": (
                                "meta=Bot 永久自我认知 (跨所有会话恒激活); "
                                "contract=角色扮演/临时承诺 (可被撤销); "
                                "episodic=普通对话事件 (默认)。"
                            ),
                        },
                        "recall_scope": {
                            "type": "string",
                            "description": "global | group:qq_{group_id} | private:qq_{user_id} (依对话片段开头 [场景:] 决定)。",
                        },
                        "roles": {
                            "type": "array",
                            "description": "参与者数组。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "enum": [
                                            "agent", "patient", "theme", "recipient",
                                            "instrument", "location", "time", "attribute",
                                        ],
                                        "description": "角色名, 仅这 8 个取值。",
                                    },
                                    "entity": {
                                        "type": "string",
                                        "description": (
                                            "实体标识。'User' -> 自动替换为当前发言用户; "
                                            "'Bot' -> 自动替换为 Bot:self; 其他保留为外部实体。"
                                        ),
                                    },
                                    "value_text": {
                                        "type": "string",
                                        "description": "当承载是一段文本/概念而非已知实体时使用 (如 theme 是被传授的内容)。",
                                    },
                                },
                                "required": ["role"],
                            },
                        },
                    },
                    "required": ["event_type", "summary", "roles"],
                },
            },
            "assertions": {
                "type": "array",
                "description": "静态本体二元事实列表 (仅限永久属性)。",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "'User' (当前用户) / 'Bot' (Bot 自己) / 其他外部实体名。",
                        },
                        "predicate": {
                            "type": "string",
                            "description": "二元谓词, 如 'isA' / '职业是' / '生于'。",
                        },
                        "object_text": {
                            "type": "string",
                            "description": "宾语文本。",
                        },
                        "recall_scope": {
                            "type": "string",
                            "description": "global | group:qq_{group_id} | private:qq_{user_id}。",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "0.0~1.0。",
                        },
                    },
                    "required": ["subject", "predicate", "object_text"],
                },
            },
        },
        "required": ["events", "assertions"],
    },
}

ARCHIVE_GEN: dict[str, Any] = {
    "temperature": 0.3,
    "max_output_tokens": 5000,
}


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """补齐缺失的空数组，避免单侧遗漏直接让整次归档失效。"""
    repaired = args
    changes: list[str] = []

    for field in ("events", "assertions"):
        if field in repaired and repaired[field] is not None:
            continue
        if repaired is args:
            repaired = dict(args)
        repaired[field] = []
        changes.append(f"filled missing {field} with []")

    return repaired, changes


TOOL = InternalToolSpec(
    declaration=DECLARATION,
    schema_repairer=repair_schema_args,
)


def read_result(result: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读取结构化归档结果；非法形状统一回退为空数组。"""
    if not isinstance(result, dict):
        return [], []

    events = result.get("events") or []
    assertions = result.get("assertions") or []
    if not isinstance(events, list):
        events = []
    if not isinstance(assertions, list):
        assertions = []
    return events, assertions