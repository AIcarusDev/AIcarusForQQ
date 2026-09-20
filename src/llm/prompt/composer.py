"""Small, order-preserving primitives for composing model messages."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


PromptContent = str | list[dict[str, Any]]
PromptMessage = dict[str, Any]


def prompt_content_is_empty(content: PromptContent | None) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            if str(part.get("text") or "").strip():
                return False
        else:
            return False
    return True


def _append_text_part(parts: list[dict[str, Any]], text: str) -> None:
    if not text:
        return
    if parts and parts[-1].get("type") == "text":
        parts[-1] = {
            **parts[-1],
            "text": str(parts[-1].get("text") or "") + text,
        }
        return
    parts.append({"type": "text", "text": text})


def merge_prompt_contents(contents: Iterable[PromptContent | None]) -> PromptContent:
    """Join source-owned blocks while preserving multimodal part order."""

    selected = [content for content in contents if not prompt_content_is_empty(content)]
    if not selected:
        return ""
    if all(isinstance(content, str) for content in selected):
        return "\n".join(str(content) for content in selected)

    parts: list[dict[str, Any]] = []
    for index, content in enumerate(selected):
        if index:
            _append_text_part(parts, "\n")
        if isinstance(content, str):
            _append_text_part(parts, content)
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                _append_text_part(parts, str(part.get("text") or ""))
            else:
                parts.append(copy.deepcopy(part))
    return parts


class ComposerSection(Protocol):
    def build_messages(self) -> list[PromptMessage]: ...


@dataclass(frozen=True)
class MessageSection:
    role: str
    content: PromptContent | None
    omit_if_empty: bool = True

    def build_messages(self) -> list[PromptMessage]:
        if self.omit_if_empty and prompt_content_is_empty(self.content):
            return []
        return [{"role": self.role, "content": self.content or ""}]


@dataclass(frozen=True)
class UserMessageSection:
    contents: tuple[PromptContent | None, ...]

    def __init__(self, contents: Iterable[PromptContent | None]) -> None:
        object.__setattr__(self, "contents", tuple(contents))

    def build_messages(self) -> list[PromptMessage]:
        content = merge_prompt_contents(self.contents)
        if prompt_content_is_empty(content):
            return []
        return [{"role": "user", "content": content}]


@dataclass
class PromptComposer:
    sections: list[ComposerSection] = field(default_factory=list)

    def add_section(self, section: ComposerSection) -> "PromptComposer":
        self.sections.append(section)
        return self

    def compose(self) -> list[PromptMessage]:
        messages: list[PromptMessage] = []
        for section in self.sections:
            messages.extend(section.build_messages())
        return messages

