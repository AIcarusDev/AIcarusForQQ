"""Configuration helpers for browser prompt/world behavior."""

from __future__ import annotations

from typing import Any

from llm.compression.config import (
    DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
    normalize_world_multimodal_image_limit,
)

DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT = DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT
DEFAULT_BROWSER_ANNOTATE_SCREENSHOTS = False
DEFAULT_BROWSER_RESULT_LIMITS = {
    "text_preview_chars": 800,
    "page_image_urls": 12,
    "click_targets": 16,
    "visible_images": 8,
    "cached_images": 12,
    "response_errors": 5,
    "url_chars": 180,
    "text_chars": 160,
}


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


def normalize_browser_control_config(raw_cfg: dict | None) -> dict[str, Any]:
    """Return the public browser settings shape used by settings UI/API."""
    browser_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    raw_limits = browser_cfg.get("result_limits")
    raw_limits = raw_limits if isinstance(raw_limits, dict) else {}
    result_limits: dict[str, int] = {}
    for key, default in DEFAULT_BROWSER_RESULT_LIMITS.items():
        try:
            value = int(raw_limits.get(key, default))
        except (TypeError, ValueError):
            value = default
        result_limits[key] = max(0, value)
    return {
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
        "result_limits": result_limits,
    }


def browser_multimodal_image_limit(config: dict[str, Any] | None) -> int:
    """Read the browser-only multimodal image budget from runtime config."""
    cfg = config if isinstance(config, dict) else {}
    if not bool(cfg.get("vision", True)):
        return 0
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return int(normalize_browser_control_config(browser_cfg)["multimodal_image_limit"])


def browser_screenshot_annotations_enabled(config: dict[str, Any] | None) -> bool:
    """Read whether browser viewport screenshots should include visual overlays."""
    cfg = config if isinstance(config, dict) else {}
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return bool(normalize_browser_control_config(browser_cfg)["annotate_screenshots"])
