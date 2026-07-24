"""Configuration helpers for browser prompt/world behavior."""

from __future__ import annotations

from typing import Any

from llm.compression.config import (
    DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
    normalize_world_multimodal_image_limit,
)
from browser.image_resources import normalize_source_url_mode

DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT = DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT
DEFAULT_BROWSER_ANNOTATE_SCREENSHOTS = False
DEFAULT_BROWSER_PROFILE_DIR = "cache/browser_profile/default"
DEFAULT_BROWSER_IMAGE_SOURCE_URL = "full"
DEFAULT_BROWSER_IMAGE_SEND_CONFIRMATION = "off"


def _normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(value)


def _normalize_string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalize_confirmation_mode(value: Any) -> str:
    mode = str(value or DEFAULT_BROWSER_IMAGE_SEND_CONFIRMATION).strip().lower().replace("-", "_")
    return mode if mode in {"off", "high_risk"} else DEFAULT_BROWSER_IMAGE_SEND_CONFIRMATION


def normalize_browser_control_config(raw_cfg: dict | None) -> dict[str, int | bool | str]:
    """Return the public browser settings shape used by settings UI/API."""
    browser_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    return {
        "profile_dir": _normalize_string(
            browser_cfg.get("profile_dir"),
            DEFAULT_BROWSER_PROFILE_DIR,
        ),
        "multimodal_image_limit": normalize_world_multimodal_image_limit(
            browser_cfg.get(
                "multimodal_image_limit",
                DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT,
            )
        ),
        "annotate_screenshots": _normalize_bool(
            browser_cfg.get("annotate_screenshots"),
            DEFAULT_BROWSER_ANNOTATE_SCREENSHOTS,
        ),
        "image_source_url": normalize_source_url_mode(
            browser_cfg.get("image_source_url", DEFAULT_BROWSER_IMAGE_SOURCE_URL)
        ),
        "image_send_confirmation": _normalize_confirmation_mode(
            browser_cfg.get(
                "image_send_confirmation",
                DEFAULT_BROWSER_IMAGE_SEND_CONFIRMATION,
            )
        ),
    }


def browser_multimodal_image_limit(config: dict[str, Any] | None) -> int:
    """Read the browser-only multimodal image budget from runtime config."""
    cfg = config if isinstance(config, dict) else {}
    if not bool(cfg.get("vision", True)):
        return 0
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return int(normalize_browser_control_config(browser_cfg)["multimodal_image_limit"])


def browser_profile_dir(config: dict[str, Any] | None) -> str:
    """Read persistent browser profile directory from runtime config."""
    cfg = config if isinstance(config, dict) else {}
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return str(normalize_browser_control_config(browser_cfg)["profile_dir"])


def browser_screenshot_annotations_enabled(config: dict[str, Any] | None) -> bool:
    """Read whether browser viewport screenshots should include visual overlays."""
    cfg = config if isinstance(config, dict) else {}
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return bool(normalize_browser_control_config(browser_cfg)["annotate_screenshots"])


def browser_image_source_url_mode(config: dict[str, Any] | None) -> str:
    cfg = config if isinstance(config, dict) else {}
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return str(normalize_browser_control_config(browser_cfg)["image_source_url"])


def browser_image_send_confirmation(config: dict[str, Any] | None) -> str:
    cfg = config if isinstance(config, dict) else {}
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return str(normalize_browser_control_config(browser_cfg)["image_send_confirmation"])
