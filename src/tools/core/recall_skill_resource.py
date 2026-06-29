"""Recall one active skill reference resource."""

from __future__ import annotations

from typing import Any

from skills import load_skill_resource
from tools.namespaces import load_namespace_registry


DECLARATION: dict = {
    "name": "recall_skill_resource",
    "description": (
        "回忆当前已激活 skill 的 resource 资源。当 skill 正文的 References "
        "提示某个资源 id，且你需要时使用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": "当前已激活的 skill name。",
            },
            "resource": {
                "type": "string",
                "description": "References 中列出的资源名称。",
            },
        },
        "required": ["skill", "resource"],
    },
}


def execute(skill: str, resource: str, **_: Any) -> dict[str, Any]:
    skill_id = str(skill or "").strip()
    resource_id = str(resource or "").strip()
    if skill_id not in _active_skill_ids():
        return {
            "ok": False,
            "error": "skill is not active; open the related namespace first",
            "skill": skill_id,
            "resource": resource_id,
        }
    return load_skill_resource(skill_id, resource_id)


def _active_skill_ids() -> set[str]:
    try:
        import app_state

        state = getattr(app_state, "namespace_runtime_state", None)
        if state is None:
            return set()
        registry = load_namespace_registry()
        skills: set[str] = set()
        for namespace in state.active_namespaces(registry):
            spec = registry.get(namespace)
            skill = str(getattr(spec, "skill", "") or "").strip()
            if skill:
                skills.add(skill)
        return skills
    except Exception:
        return set()
