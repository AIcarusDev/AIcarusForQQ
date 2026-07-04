"""Structured focus references for platform conversations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FocusRef:
    platform: str
    target_type: str
    target_id: str
    target_name: str = ""

    def key(self) -> str:
        return session_key_for_focus(self)

    def with_name(self, target_name: str) -> "FocusRef":
        return FocusRef(
            platform=self.platform,
            target_type=self.target_type,
            target_id=self.target_id,
            target_name=target_name,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "platform": self.platform,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_name": self.target_name,
        }


def normalize_focus(value: Any) -> FocusRef | None:
    if isinstance(value, FocusRef):
        return value
    if isinstance(value, dict):
        platform = str(value.get("platform") or "").strip()
        target_type = str(value.get("target_type") or value.get("type") or "").strip()
        target_id = str(value.get("target_id") or value.get("id") or "").strip()
        target_name = str(value.get("target_name") or value.get("name") or "").strip()
        if platform and target_type and target_id:
            return FocusRef(platform, target_type, target_id, target_name)
    if isinstance(value, str):
        return focus_from_session_key(value)
    return None


def session_key_for_focus(focus: FocusRef | None) -> str:
    if focus is None:
        return ""
    return f"{focus.platform}:{focus.target_type}:{focus.target_id}"


def focus_from_session_key(session_key: str | None, *, default_platform: str = "qq") -> FocusRef | None:
    raw = str(session_key or "").strip()
    if not raw:
        return None
    parts = raw.split(":", 2)
    if len(parts) == 3 and all(parts):
        return FocusRef(parts[0], parts[1], parts[2])

    legacy_type, sep, legacy_id = raw.partition("_")
    if sep and legacy_type in {"group", "private", "temp"} and legacy_id:
        return FocusRef(default_platform, legacy_type, legacy_id)
    return None


def current_focus_key(value: Any) -> str:
    return session_key_for_focus(normalize_focus(value))


def focus_matches(value: Any, session_key: str) -> bool:
    return current_focus_key(value) == session_key
