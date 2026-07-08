"""Platform runtime registry and shared platform abstractions."""

from .attention import AttentionEvent
from .base import PlatformAccount, PlatformRuntime, PlatformToolContext, PlatformWorldBlock
from .focus import FocusRef, current_focus_key, focus_from_session_key, session_key_for_focus
from .registry import PlatformRegistry, get_platform

__all__ = [
    "FocusRef",
    "AttentionEvent",
    "PlatformAccount",
    "PlatformRegistry",
    "PlatformRuntime",
    "PlatformToolContext",
    "PlatformWorldBlock",
    "current_focus_key",
    "focus_from_session_key",
    "get_platform",
    "session_key_for_focus",
]
