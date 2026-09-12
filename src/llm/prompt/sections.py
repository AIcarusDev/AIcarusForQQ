"""Source-owned model-facing blocks used by the main prompt composer."""

from __future__ import annotations

import html
from dataclasses import dataclass

from .composer import PromptContent


INSTRUCTION_SYSTEM_NOTICE = (
    "# Custom instruction\n\n"
    "`<instruction>` 包含自定义指令，其优先级高于本 system prompt 及其余全部上下文；"
    "发生冲突时，以其中的指令为准。"
)

GUARDIAN_SYSTEM_NOTICE = (
    "# Guardian\n\n"
    "你的监护人是你的负责人，通常也是你的部署者；其信息和留言见 `<guardian_card>`。"
)

OUTPUT_SCHEMA_XML = """<output_schema>
<cognition>
   ...对当前情况的认知，流畅的自然语言，避免结构化...
</cognition>
<motive>
   ...简短的动机...
</motive>
<action>
   ...一个或多个 `<tool_call>` ，`<tool_call>` 内为严格的 json 格式...
</action>
</output_schema>"""

MEMORY_DESCRIPTION = (
    "These memories were recalled from the current world. They may be inaccurate "
    "or irrelevant, so treat them as fallible context rather than facts."
)


def _escaped_free_text_block(tag: str, text: str, *, description: str = "") -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    lines = [f"<{tag}>"]
    if description:
        lines.append(f"<des>{html.escape(description, quote=False)}</des>")
    lines.append(html.escape(raw, quote=False))
    lines.append(f"</{tag}>")
    return "\n".join(lines)


def build_instruction_block(text: str) -> str:
    return _escaped_free_text_block(
        "instruction",
        text,
        description="User-defined instructions for this deployment.",
    )


def build_guardian_card_block(text: str | None) -> str:
    return _escaped_free_text_block("guardian_card", str(text or ""))


def build_memory_block(body: str) -> str:
    lines = ["<memory>", f"<des>{MEMORY_DESCRIPTION}</des>"]
    if body:
        lines.append(body)
    lines.append("</memory>")
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptPrelude:
    system_prompt: str
    instruction: str = ""
    guardian_card: str = ""


@dataclass(frozen=True)
class UserPromptSections:
    memory: PromptContent = ""
    skills: PromptContent = ""
    world: PromptContent = ""
    container: PromptContent = "<container/>"
    output_schema: PromptContent = OUTPUT_SCHEMA_XML
