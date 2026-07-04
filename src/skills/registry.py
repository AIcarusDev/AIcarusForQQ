"""Namespace-bound skill body/resource loading.

MVP rules:
- namespace metadata may bind one main skill by id;
- active namespaces decide whether the skill is visible;
- rendering exposes active skills inside one <skills> container;
- resources are loaded only by explicit tool calls, not by prompt rendering;
- independent skill lifecycle is intentionally out of scope.
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
_RESOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SKILL_FILENAME = "SKILL.md"
_SKILL_TEMPLATE_FILENAME = "SKILL.md.template"


def _strip_frontmatter(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return normalized.strip("\n")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return normalized.strip("\n")
    return normalized[end + len("\n---\n"):].strip("\n")


def _normalize_skill_body(text: str) -> str:
    body = _strip_frontmatter(text)
    if body.startswith("# ") and not body.startswith("## "):
        return "#" + body
    return body


def _split_frontmatter(text: str) -> tuple[str, str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return "", normalized.strip("\n")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        return "", normalized.strip("\n")
    frontmatter = normalized[: end + len("\n---\n")]
    body = normalized[end + len("\n---\n"):].strip("\n")
    return frontmatter, body


def _skill_dir(skill_id: str) -> Path | None:
    skill_id = str(skill_id or "").strip()
    if not skill_id or not _SKILL_ID_RE.fullmatch(skill_id):
        return None
    return _SKILLS_DIR / skill_id


def _skill_file_path(skill_id: str) -> Path | None:
    directory = _skill_dir(skill_id)
    return None if directory is None else directory / _SKILL_FILENAME


def _skill_template_path(skill_id: str) -> Path | None:
    directory = _skill_dir(skill_id)
    return None if directory is None else directory / _SKILL_TEMPLATE_FILENAME


def ensure_skill_user_file(skill_id: str) -> Path | None:
    """Create ignored user skill file from tracked template when missing."""
    path = _skill_file_path(skill_id)
    template_path = _skill_template_path(skill_id)
    if path is None or template_path is None:
        return None
    if path.exists():
        return path
    try:
        template_text = template_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("[skills] skill template not found: %s", template_path)
        return None
    except Exception:
        logger.warning("[skills] failed reading skill template: %s", template_path, exc_info=True)
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template_text, encoding="utf-8")
        logger.info("[skills] created user skill from template: %s", path)
        return path
    except Exception:
        logger.warning("[skills] failed creating user skill: %s", path, exc_info=True)
        return None


def load_skill_user_body(skill_id: str) -> str:
    path = ensure_skill_user_file(skill_id)
    if path is None:
        return ""
    try:
        return _normalize_skill_body(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("[skills] failed reading user skill: %s", path, exc_info=True)
        return ""


def save_skill_user_body(skill_id: str, body: str) -> bool:
    path = ensure_skill_user_file(skill_id)
    template_path = _skill_template_path(skill_id)
    if path is None:
        return False
    frontmatter = ""
    for candidate in (path, template_path):
        if candidate is None:
            continue
        try:
            frontmatter = _split_frontmatter(candidate.read_text(encoding="utf-8"))[0]
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning("[skills] failed reading skill metadata: %s", candidate, exc_info=True)
            continue
        if frontmatter:
            break
    text = str(body or "").strip("\r\n")
    payload = f"{frontmatter}\n{text}\n" if frontmatter else f"{text}\n"
    try:
        path.write_text(payload, encoding="utf-8")
        load_skill_body.cache_clear()
        return True
    except Exception:
        logger.warning("[skills] failed saving user skill: %s", path, exc_info=True)
        return False


@lru_cache(maxsize=64)
def load_skill_body(skill_id: str) -> str:
    path = ensure_skill_user_file(skill_id)
    if path is None:
        return ""
    try:
        return _normalize_skill_body(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("[skills] skill body not found: %s", path)
    except Exception:
        logger.warning("[skills] failed to read skill body: %s", path, exc_info=True)
    return ""


def _normalize_resource_id(resource_id: str) -> str:
    resource_id = str(resource_id or "").strip()
    if resource_id.endswith(".md"):
        resource_id = resource_id[:-3]
    if not resource_id or not _RESOURCE_ID_RE.fullmatch(resource_id):
        return ""
    return resource_id


def load_skill_resource(
    skill_id: str,
    resource_id: str,
    *,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Read one markdown resource under ``src/skills/<skill>/references``."""
    skill_id = str(skill_id or "").strip()
    resource = _normalize_resource_id(resource_id)
    result_base = {
        "skill": skill_id,
        "resource": resource or str(resource_id or "").strip(),
    }
    if not skill_id or not _SKILL_ID_RE.fullmatch(skill_id):
        return {**result_base, "ok": False, "error": "invalid skill id"}
    if not resource:
        return {**result_base, "ok": False, "error": "invalid resource id"}

    references_dir = (_SKILLS_DIR / skill_id / "references").resolve()
    path = (references_dir / f"{resource}.md").resolve()
    try:
        path.relative_to(references_dir)
    except ValueError:
        return {**result_base, "ok": False, "error": "invalid resource path"}

    try:
        content = path.read_text(encoding="utf-8").strip("\n")
    except FileNotFoundError:
        return {
            **result_base,
            "ok": False,
            "error": "resource not found",
            "path": f"references/{resource}.md",
        }
    except Exception:
        logger.warning("[skills] failed to read skill resource: %s", path, exc_info=True)
        return {**result_base, "ok": False, "error": "failed to read resource"}

    truncated = False
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars].rstrip()
        truncated = True

    return {
        **result_base,
        "ok": True,
        "path": f"references/{resource}.md",
        "content": content,
        "truncated": truncated,
    }


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
