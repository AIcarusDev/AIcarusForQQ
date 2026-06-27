"""Namespace-bound skill body loading.

MVP rules:
- namespace metadata may bind one main skill by id;
- active namespaces decide whether the skill is visible;
- rendering exposes active skills inside one <skills> container;
- resources and independent skill lifecycle are intentionally out of scope.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any

logger = logging.getLogger("AICQ.skills")

_SKILLS_DIR = Path(__file__).resolve().parent
_SKILL_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _strip_frontmatter(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return normalized.strip("\n")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return normalized.strip("\n")
    return normalized[end + len("\n---\n"):].strip("\n")


@lru_cache(maxsize=64)
def load_skill_body(skill_id: str) -> str:
    skill_id = str(skill_id or "").strip()
    if not skill_id or not _SKILL_ID_RE.fullmatch(skill_id):
        return ""
    path = _SKILLS_DIR / skill_id / "SKILL.md"
    try:
        return _strip_frontmatter(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("[skills] skill body not found: %s", path)
    except Exception:
        logger.warning("[skills] failed to read skill body: %s", path, exc_info=True)
    return ""


def build_skill_block_for_namespaces(
    active_namespaces: list[str] | tuple[str, ...],
    namespace_registry: Any,
) -> str:
    seen_skill_ids: set[str] = set()
    skill_blocks: list[str] = []
    for namespace in active_namespaces:
        spec = namespace_registry.get(namespace) if namespace_registry is not None else None
        skill_id = str(getattr(spec, "skill", "") or "").strip()
        if not skill_id or skill_id in seen_skill_ids:
            continue
        seen_skill_ids.add(skill_id)
        body = load_skill_body(skill_id).strip()
        if body:
            skill_name = escape(skill_id, quote=True)
            skill_blocks.append(f'<skill name="{skill_name}">\n{body}\n</skill>')
    if not skill_blocks:
        return ""
    return "<skills>\n" + "\n".join(skill_blocks) + "\n</skills>"
