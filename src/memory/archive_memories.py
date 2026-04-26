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
                            "description": (
                                "简短动词标签, 优先用闭合小词表: "
                                "say/teach/correct/ask/answer/promise/refuse/agree/"
                                "like/dislike/feel/experience/own/be/do。"
                                "sharing/joking/sarcasm 等说话意图差异不要塞进来, 编码到 attribute 角色。"
                            ),
                        },
                        "summary": {
                            "type": "string",
                            "description": "一句话事件摘要 (<=30 字), 用于检索与渲染。",
                        },
                        "polarity": {
                            "type": "string",
                            "enum": ["positive", "negative"],
                            "description": (
                                "说话者对事件的态度/承诺方向, 不是句子表层是否含'不/没'。"
                                "Q1 表达好恶/拒绝/反对? 是→negative。"
                                "Q2 被承诺为真的客观陈述? 是→positive(即使含'不')。"
                                "反例: 'Python 不是编译语言' → positive (是被肯定的事实)。"
                                "正例: '我不喜欢香菜' → negative。"
                            ),
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
                            "enum": ["meta", "contract", "episodic"],
                            "description": (
                                "按 (跨会话恒真? × 可被对话覆盖?) 二维判定: "
                                "meta=恒真且不可覆盖 (例: '我是 AI'); "
                                "contract=恒真但可被撤销 (例: '这次扮演吹雪'); "
                                "episodic=仅本次对话有效, 默认 (例: 偏好/今天的事)。"
                                "反例: '我喜欢科幻' 是 episodic 而非 meta (偏好可变)。"
                                "反例: '我现在是吹雪' 是 contract 而非 meta (可撤销)。"
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
                            "description": (
                                "四档锚点: 0.95=当事人直述; 0.80=上下文直接推断; "
                                "0.50=合理猜测; 0.30=八卦/玩笑。"
                            ),
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