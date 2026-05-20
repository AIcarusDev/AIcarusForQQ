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
                            "enum": [
                                "say", "share", "complain", "joke", "update",
                                "teach", "correct", "ask", "answer",
                                "promise", "refuse", "agree",
                                "like", "dislike", "feel", "experience",
                                "own", "be", "do", "isA"
                            ],
                            "description": (
                                "事件的核心动词谓词，描述「谁对谁做了/处于什么关系」中的那个「做了什么」。"
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": "事件摘要，用于检索；须是一句含主谓宾、脱离上下文也能独立阅读的完整句子。",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "以数值判断事件真实程度，范围为0~1.0，越大越可信",
                        },
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": (
                                "positive=肯定/赞同/喜欢; "
                                "negative=否定/拒绝/厌恶。判断说话者态度，非表层有无'不/没'。"
                            ),
                        },
                        "recall_scope": {
                            "type": "string",
                            "enum": ["global", "current_session"],
                            "description": (
                                "global=适用于所有会话的通用事实; "
                                "current_session=仅在当前对话所在群组/私聊内有意义的事实。"
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "填写 merge_into 或 supersedes 时说明理由（可选）。"
                            ),
                        },
                        "merge_into": {
                            "type": "integer",
                            "description": (
                                "若存在事件与“某条<existing_candidates>”完全“重复”，填入“某条<existing_candidates>”的ID"
                            ),
                        },
                        "supersedes": {
                            "type": "integer",
                            "description": (
                                "若存在事件“推翻”了“某条<existing_candidates>”，填入“某条<existing_candidates>”的 ID"
                            ),
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
                                        "description": "参与者在事件中扮演的语义角色名",
                                    },
                                    "entity": {
                                        "type": "string",
                                        "description": (
                                            "参与该事件的实体标识符，格式为「命名空间:ID」"
                                        ),
                                    },
                                    "value_text": {
                                        "type": "string",
                                        "description": "当参与者是一段文本或抽象概念、而非可命名实体时填此字段，与 entity 二选一。",
                                    },
                                    "target_event": {
                                        "type": "integer",
                                        "description": (
                                            "当参与者是另一个事件时（如'反驳事件#3'），"
                                            "填入该事件的 event_id。与 entity / value_text 三选一。"
                                        ),
                                    },
                                },
                                "required": ["role"],
                            },
                        },
                    },
                    "required": ["event_type", "summary", "roles"],
                },
            },
        },
        "required": ["events"],
    },
}

ARCHIVE_GEN: dict[str, Any] = {
    "temperature": 0.3,
    "max_output_tokens": 10000,
}


def repair_schema_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """补齐缺失的 events 数组，避免整次归档失效。"""
    repaired = args
    changes: list[str] = []

    if repaired.get("events") is None:
        repaired = dict(args)
        repaired["events"] = []
        changes.append("filled missing events with []")

    return repaired, changes


TOOL = InternalToolSpec(
    declaration=DECLARATION,
    schema_repairer=repair_schema_args,
)


def read_result(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """读取结构化归档结果；非法形状统一回退为空数组。"""
    if not isinstance(result, dict):
        return []
    events = result.get("events") or []
    if not isinstance(events, list):
        events = []
    return events
