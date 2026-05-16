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
                                "own", "be", "do"," isA"
                            ],
                            "description": (
                                "简短动词标签。"
                                "说话者意图差异（语气/讽刺/玩笑）不要写进 event_type，编码到 attribute 角色。"
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": "事件摘要，注意不要过于冗长，用于检索与渲染。",
                        },
                        "modality": {
                            "type": "string",
                            "enum": ["actual", "hypothetical", "possible"],
                            "description": (
                                "actual=默认, 真实发生/存在。"
                                "possible=含'可能/也许/大概/估计/或许'等认知不确定词。"
                                "hypothetical=含'如果/假如/要是/万一/假设'等反事实条件。"
                                "反例: '他可能在睡觉' 是 possible 而非 hypothetical。"
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "description": (
                                "四档锚点, 选最接近的一个, 不要猜小数: "
                                "0.95=当事人亲口直述; "
                                "0.80=上下文可直接推断; "
                                "0.50=合理猜测缺直接证据; "
                                "0.30=八卦/玩笑/趣闻。"
                            ),
                        },
                        "context_type": {
                            "type": "string",
                            "enum": ["episodic", "hypothetical"],
                            "description": (
                                "默认 episodic。episodic=真实发生、真实陈述、偏好、状态、角色扮演设定、"
                                "Bot 自身描述等。hypothetical=只在假设/反事实语境下成立, 不应当作事实回忆。"
                            ),
                        },
                        "recall_scope": {
                            "type": "string",
                            "enum": ["global", "current_session"],
                            "description": (
                                "global=适用于所有会话的通用事实; "
                                "current_session=仅在当前对话所在群组/私聊内有意义的事实 (依对话片段开头 [场景:] 判断)。"
                            ),
                        },
                        "merge_into": {
                            "type": "integer",
                            "description": (
                                "Read-Before-Write: 若本事件与 <existing_candidates> 中某条 id=X 表达"
                                "完全相同的事实(同 agent + 同 theme), 写 X, 系统会把 X 的 occurrences+1, "
                                "本事件不再新建。仅用于「我又一次说过同一件事」, 不要用于「相关」或「相似」。"
                            ),
                        },
                        "supersedes": {
                            "type": "integer",
                            "description": (
                                "Read-Before-Write: 若本事件**改写/推翻**了 <existing_candidates> 中某条 id=X 的旧事实"
                                "(例如旧的'我喜欢苹果' 被新的'我现在不喜欢苹果了'取代), 写 X。"
                                "系统会软删 X 并记录链路。仅用于真正的语义反转/更新, 不要用于补充细节。"
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
                                        "description": "角色名, 仅这 8 个取值。",
                                    },
                                    "entity": {
                                        "type": "string",
                                        "description": (
                                            "实体标识 (跨事件保持稳定, 才能在图谱中连接)。"
                                            "对话中说话人=`User:qq_{id}` (取每行行首的形式, 切勿写成 `User`/`User(昵称)`); "
                                            "Bot 自己=`Bot:self`; "
                                            "外界第三方使用规范命名空间: `Tool:qwen` / `Org:OpenAI` / `Person:马斯克` 等。"
                                            "纯抽象内容(被传授的概念/被讨论的话题文本)走 value_text。"
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
