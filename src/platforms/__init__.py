"""Platform runtime registry and shared platform abstractions."""

from .base import PlatformAccount, PlatformRuntime, PlatformToolContext
from .focus import FocusRef, current_focus_key, focus_from_session_key, session_key_for_focus
from .registry import PlatformRegistry, get_platform

__all__ = [
    "FocusRef",
    "PlatformAccount",
    "PlatformRegistry",
    "PlatformRuntime",
    "PlatformToolContext",
    "current_focus_key",
    "focus_from_session_key",
    "get_platform",
    "session_key_for_focus",
]
