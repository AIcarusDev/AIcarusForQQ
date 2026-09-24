"""Bound model-facing previews without changing stored memory text."""

MEMORY_PREVIEW_CHARS = 500


def preview(content: str) -> tuple[str, bool]:
    return content[:MEMORY_PREVIEW_CHARS], len(content) > MEMORY_PREVIEW_CHARS
