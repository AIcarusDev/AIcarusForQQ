"""通用参数辅助函数。"""

from typing import Any


def strip_legacy_motivation_fields(value: Any) -> tuple[Any, bool]:
    """递归移除旧工具参数中的 motivation 字段。"""
    if isinstance(value, dict):
        changed = False
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "motivation":
                changed = True
                continue
            cleaned_item, item_changed = strip_legacy_motivation_fields(item)
            cleaned[key] = cleaned_item
            changed = changed or item_changed
        return (cleaned if changed else value), changed

    if isinstance(value, list):
        changed = False
        cleaned_items: list[Any] = []
        for item in value:
            cleaned_item, item_changed = strip_legacy_motivation_fields(item)
            cleaned_items.append(cleaned_item)
            changed = changed or item_changed
        return (cleaned_items if changed else value), changed

    return value, False