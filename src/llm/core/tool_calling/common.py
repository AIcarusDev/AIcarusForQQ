"""通用文本辅助函数。"""

import re

_NORMALIZED_WHITESPACE_RE = re.compile(r"\s+")


def normalize_repeat_text(text: str) -> str:
    """归一化文本，用于保守判断是否只是复读。"""
    return _NORMALIZED_WHITESPACE_RE.sub(" ", text).strip()


def merge_motivation_texts(values: list[str]) -> tuple[str | None, bool]:
    """合并多个 motivation；纯复读时保留一个，否则按顺序拼接。"""
    unique_values: list[str] = []
    seen_markers: set[str] = set()

    for value in values:
        stripped = value.strip()
        if not stripped:
            continue
        marker = normalize_repeat_text(stripped)
        if not marker or marker in seen_markers:
            continue
        seen_markers.add(marker)
        unique_values.append(stripped)

    if not unique_values:
        return None, False
    if len(unique_values) == 1:
        return unique_values[0], len(values) > 1
    return "\n\n".join(unique_values), True