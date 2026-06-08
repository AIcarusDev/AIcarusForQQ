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
    "send_message",
    "sleep",
    "wait",
    "shift",
    "search_session",
    "get_tools",
    "create_goal",
    "think_deeply",
    "scroll_chat_log",
    "open_forward_message",
    "browse_forward_view",
    "recall_message",
    "poke",
    "recall_memory",
    "web_search",
    "web_extract",
    "get_weather",
    "get_self_image",
    "list_stickers",
    "save_sticker",
    "update_sticker",
    "delete_sticker",
    "check_physical_state",
    # ================== CACHE boundary ==================
    # Conditional but schema-stable tools.
    "resolve_goal",
    "examine_image",
    "plus_one",
    # Latent/hidden tools: shared first, group-only after.
    "restart_self",
    "search_current_session_chat_history",
    "get_contact_list",
    "get_qq_signature",
    "get_user_avatar",
    "set_self_qq_signature",
    "get_group_members",
    "get_group_notice_list",
    "get_group_notice_detail",
    "set_self_group_card",
    # Runtime-dynamic schema.
    "send_voice_message",
)

_ORDER_INDEX = {name: index for index, name in enumerate(TOOL_ORDER)}
_CACHE_BOUNDARY_AFTER_TOOL = "check_physical_state"
_CACHE_BOUNDARY_INDEX = _ORDER_INDEX[_CACHE_BOUNDARY_AFTER_TOOL] + 1


def tool_order_key(name: str) -> tuple[int, str]:
    """Return a stable sort key for prompt-facing tool lists."""
    return (_ORDER_INDEX.get(name, len(TOOL_ORDER)), name)


def cacheable_tool_names() -> tuple[str, ...]:
    """Tool names above the prompt-cache boundary marker."""
    return TOOL_ORDER[:_CACHE_BOUNDARY_INDEX]
