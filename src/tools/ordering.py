"""Canonical prompt-facing order for tools.

This order is intentionally independent from filesystem discovery order.  Keep
the stable, shared tool prefix first so the rendered ``<tools>`` payload can
reuse as much prompt cache as possible across sessions.  Put conditional tools
and runtime-dynamic schemas near the end.
"""

from __future__ import annotations


CACHE_BOUNDARY_MARKER = "# ================== CACHE boundary =================="

TOOL_ORDER: tuple[str, ...] = (
    # Stable, shared active tools.
    "namespace_manage",
    "calculator",
    "wait",
    "sleep",
    "shift",
    "think_deeply",
    "recall_memory",
    "goal_manage",
    "restart",
    "view_image_by_ref",
    "examine_image",
    "web_search",
    "web_extract",
    "get_weather",
    # ================== CACHE boundary ==================
    # QQ social.
    "send_message",
    "send_voice",
    "recall_message",
    "poke",
    "plus_one",
    # Stickers.
    "list_stickers",
    "save_sticker",
    "update_sticker",
    "delete_sticker",
    # Chat view.
    "scroll_chat_log",
    "browse_forward",
    "search_history",
    # Profile and contacts.
    "get_qq_signature",
    "set_qq_signature",
    "get_avatar",
    "list_contact",
    "search_session",
    # Group info.
    "get_group_members",
    "get_group_notice_list",
    "get_group_notice_detail",
    "set_group_card",
    # Browser.
    "browser_control",
    "browser_locator",
)

_ORDER_INDEX = {name: index for index, name in enumerate(TOOL_ORDER)}
_CACHE_BOUNDARY_AFTER_TOOL = "get_weather"
_CACHE_BOUNDARY_INDEX = _ORDER_INDEX[_CACHE_BOUNDARY_AFTER_TOOL] + 1


def tool_order_key(name: str) -> tuple[int, str]:
    """Return a stable sort key for prompt-facing tool lists."""
    return (_ORDER_INDEX.get(name, len(TOOL_ORDER)), name)


def cacheable_tool_names() -> tuple[str, ...]:
    """Tool names above the prompt-cache boundary marker."""
    return TOOL_ORDER[:_CACHE_BOUNDARY_INDEX]
