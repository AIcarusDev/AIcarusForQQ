"""第二轨 evidence 提取工具协议。

evidence 事件语义：「有人说了关于第三方实体的某件事」。
与 episodic 事件的区别：
  - episodic: 记录说话者自身的状态/行为/经历（agent = 说话者）
  - evidence: 记录说话者描述的第三方命题（agent ≠ 说话者 = instrument）
"""

from __future__ import annotations

from typing import Any

from llm.core.internal_tool import InternalToolSpec

# ── 生成参数 ──────────────────────────────────────────────────────────────────

EVIDENCE_GEN: dict[str, Any] = {
    "temperature": 0.3,
    "max_output_tokens": 6000,
}

# ── 系统提示 ──────────────────────────────────────────────────────────────────

EVIDENCE_SYSTEM_PROMPT = """
你是「证据提取助手」。本任务以函数调用形式工作：你必须且只能调用工具 extract_evidence。

输入是一批从群聊提取的 episodic 事件（记录「谁说了/做了什么」）。
你的任务是识别：这些话语中，是否有人描述了一个第三方实体的状态/行为/属性？
如果有，提取一条 evidence 事件，描述那个第三方实体，并标注说话者为证人。

核心原则：
- 你不判断命题是否为真——只判断「这段话构成某命题成立的证据」
- 必须有第三方实体：说话者说的是关于他人/工具/组织的事才值得提取
- 说话者说关于自己的事（agent = instrument）→ 已在 episodic 中，跳过
- 无可提取内容时仍要调用工具并把 events 填为空数组
""".strip()

# ── 工具描述（作为 description 传给 LLM）────────────────────────────────────

_TOOL_DESCRIPTION = """
从群聊的 episodic 事件列表中，识别哪些事件暗示了第三方实体的状态/行为/属性，提取为 evidence 事件。

=== 核心转换模式 ===

模式1：转述他人行为
  Episodic: [say] User:qq_123 说了关于 "B砸了C家的窗户" 的事
  → Evidence: agent=Person:B, theme_text="砸了C家的窗户",
              instrument=User:qq_123, conf=0.50

模式2：描述第三方工具/产品属性
  Episodic: [complain] User:qq_456 说了关于 "RTX 4090 散热太差" 的事
  → Evidence: agent=Tool:RTX4090, theme_text="散热性能差",
              instrument=User:qq_456, conf=0.70

模式3：推测性询问暗示对方状态
  Episodic: [ask] User:qq_789 问 User:qq_123 "你最近是不是在玩原神？"
  → Evidence: agent=User:qq_123, theme_text="可能在玩原神",
              instrument=User:qq_789, conf=0.40

=== 不要提取 ===
- agent == instrument（说话者说关于自己的事）→ 已在 episodic，跳过
- 泛泛感慨，无具体实体 → 跳过
- 无法锁定命题主语 → 跳过

=== 字段说明 ===
source_episodic_idx: 来源 episodic 事件序号（从 1 开始）
agent:        被描述的第三方实体（非说话者！）
              格式: User:qq_xxx / Tool:工具名 / Person:人名 / Org:组织名
event_type:   描述实体状态/行为的动词标签（be/do/own/experience/feel/say/share/complain 等）
summary:      命题的简洁摘要，格式「[实体] [命题]」，脱离上下文可独立阅读
theme_text:   关于该实体的完整命题文字（必填）
instrument:   提供这条证据的说话者（User:qq_xxx 格式）
confidence:   证据强度 0.30～0.80（不能超过 0.80，证据只是「指向」而非「确认」）
  0.80 = 说话者亲身接触后描述的第三方事物
  0.60 = 说话者明确陈述他人/第三方情况
  0.50 = 二手转述，来源不确定
  0.40 = 推测性询问或间接暗示
  0.30 = 极弱暗示，仅提供参考
raw_quote:    触发这条证据的原始文字片段（简短，用于溯源）
"""

# ── 工具声明 ──────────────────────────────────────────────────────────────────

DECLARATION: dict[str, Any] = {
    "name": "extract_evidence",
    "description": _TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "required": ["events"],
        "properties": {
            "events": {
                "type": "array",
                "description": "证据事件列表（空数组表示无可提取的证据）",
                "items": {
                    "type": "object",
                    "required": [
                        "source_episodic_idx", "agent", "event_type",
                        "summary", "theme_text", "instrument", "confidence",
                    ],
                    "properties": {
                        "source_episodic_idx": {
                            "type": "integer",
                            "description": "来源 episodic 事件的序号（从 1 开始）",
                        },
                        "agent": {
                            "type": "string",
                            "description": "被描述的第三方实体（非说话者），格式: User:qq_xxx / Tool:xxx / Person:xxx / Org:xxx",
                        },
                        "event_type": {
                            "type": "string",
                            "enum": [
                                "be", "do", "own", "experience", "feel",
                                "say", "share", "complain", "like", "dislike",
                                "teach", "correct", "ask", "answer", "promise",
                            ],
                            "description": "描述第三方实体状态/行为的动词标签",
                        },
                        "summary": {
                            "type": "string",
                            "description": "命题的简洁摘要，格式'[实体] [命题]'，脱离上下文可独立阅读",
                        },
                        "theme_text": {
                            "type": "string",
                            "description": "关于该实体的完整命题文字（必填）",
                        },
                        "instrument": {
                            "type": "string",
                            "description": "提供这条证据的说话者（User:qq_xxx 格式）",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "证据强度 0.30～0.80",
                        },
                        "raw_quote": {
                            "type": "string",
                            "description": "触发证据的原始文字（简短，用于溯源）",
                        },
                    },
                },
            }
        },
    },
}

TOOL = InternalToolSpec(declaration=DECLARATION)

# ── 结果读取 ──────────────────────────────────────────────────────────────────


def read_result(raw: dict | None) -> list[dict]:
    """从 _call_forced_tool 返回值中解析 evidence 事件列表。"""
    if not isinstance(raw, dict):
        return []
    events = raw.get("events") or []
    if not isinstance(events, list):
        return []
    return [e for e in events if isinstance(e, dict)]


# ── 输入格式化 ────────────────────────────────────────────────────────────────


def format_episodic_for_evidence(episodic_events: list[dict]) -> str:
    """将 Track1 写入的 episodic 事件格式化为 Track2 输入文本。

    每条事件标注 1-based 序号，供 LLM 在 source_episodic_idx 中引用。
    事件 dict 需包含：event_id, event_type, summary, modality, confidence,
    context_type, roles（来自 archiver 写入循环收集的 track1_written）。
    """
    if not episodic_events:
        return "（本段对话无 episodic 事件）"

    lines: list[str] = ["=== 待处理的 Episodic 事件列表 ===", ""]
    for i, ev in enumerate(episodic_events, 1):
        etype = ev.get("event_type", "?")
        conf = ev.get("confidence", 0)
        ctx = ev.get("context_type", "")
        mod = ev.get("modality", "actual")
        summary = ev.get("summary", "")
        roles = ev.get("roles") or []

        lines.append(f"#{i}  [{etype}/{mod}]  conf={conf}  ctx={ctx}")
        lines.append(f"    摘要: {summary}")

        role_parts: list[str] = []
        for r in roles:
            rname = r.get("role", "?")
            entity = r.get("entity", "")
            vtext = r.get("value_text", "")
            if entity:
                role_parts.append(f"{rname}={entity}")
            elif vtext:
                role_parts.append(f'{rname}="{str(vtext)[:50]}"')
        if role_parts:
            lines.append("    roles: " + ", ".join(role_parts))
        lines.append("")

    lines.append("请对每条 episodic 事件，判断是否暗示了第三方实体的命题，提取为 evidence 事件。")
    return "\n".join(lines)
