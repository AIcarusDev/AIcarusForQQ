"""Configuration helpers for browser prompt/world behavior."""

from __future__ import annotations

from typing import Any

from llm.compression.config import (
    DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
    normalize_world_multimodal_image_limit,
)

DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT = DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT


def normalize_browser_control_config(raw_cfg: dict | None) -> dict[str, int]:
    """Return the public browser settings shape used by settings UI/API."""
    browser_cfg = raw_cfg if isinstance(raw_cfg, dict) else {}
    return {
        "multimodal_image_limit": normalize_world_multimodal_image_limit(
            browser_cfg.get(
                "multimodal_image_limit",
                DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT,
            )
        )
    }


def browser_multimodal_image_limit(config: dict[str, Any] | None) -> int:
    """Read the browser-only multimodal image budget from runtime config."""
    cfg = config if isinstance(config, dict) else {}
    if not bool(cfg.get("vision", True)):
        return 0
    browser_cfg = cfg.get("browser_control") if isinstance(cfg.get("browser_control"), dict) else {}
    return normalize_browser_control_config(browser_cfg)["multimodal_image_limit"]
