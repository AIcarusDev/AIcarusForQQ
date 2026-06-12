"""Built-in browser feature package.

This package owns browser runtime state, configuration, cached images, and
world snapshots. Model-facing tool registration stays under ``tools``.
"""

from .config import (
    DEFAULT_BROWSER_ANNOTATE_SCREENSHOTS,
    DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT,
    browser_multimodal_image_limit,
    browser_screenshot_annotations_enabled,
    normalize_browser_control_config,
)
from .session import (
    BrowserSession,
    browser_debug_state,
    browser_image_path,
    browser_world_signature,
    browser_world_snapshot,
    browser_world_view_state,
    close_browser_session,
    make_image_data_url,
    read_browser_image_file,
)
from .world_prompt import build_browser_world_content, render_browser_world_content

__all__ = [
    "BrowserSession",
    "DEFAULT_BROWSER_ANNOTATE_SCREENSHOTS",
    "DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT",
    "browser_debug_state",
    "browser_image_path",
    "browser_multimodal_image_limit",
    "browser_screenshot_annotations_enabled",
    "browser_world_signature",
    "browser_world_snapshot",
    "browser_world_view_state",
    "build_browser_world_content",
    "close_browser_session",
    "make_image_data_url",
    "normalize_browser_control_config",
    "read_browser_image_file",
    "render_browser_world_content",
]
