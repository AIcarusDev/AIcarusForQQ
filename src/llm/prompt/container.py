"""container.py — 模型上下文契约 container 管理

全局维护一个内存中的 container 条目列表，包含 preset 与 custom 两个分节。
其上下文位置位于主模型 user prompt 的最末尾。
启动时从数据库恢复，运行时通过底层接口更新，并在 prompt 组装时序列化为 XML。
全空状态下理论上永远输出: <container/>
"""

import html
import secrets
import time
from typing import Any


_items: list[dict] = []
VALID_SECTIONS: tuple[str, ...] = ("preset", "custom")


def restore(rows: list[dict]) -> None:
    """从外部数据源（如数据库）恢复内存条目列表。"""
    global _items
    _items = list(rows)


def get_all() -> list[dict]:
    """获取所有活跃条目的浅拷贝。"""
    return list(_items)


def get_items(section: str | None = None) -> list[dict]:
    """按分节（preset / custom）获取条目列表。"""
    if section is None:
        return list(_items)
    return [it for it in _items if it.get("section") == section]


def _next_id() -> str:
    used = {it.get("item_id") for it in _items}
    while True:
        candidate = f"cont_{secrets.token_hex(4)}"
        if candidate not in used:
            return candidate


def _render_section(tag: str, items: list[dict]) -> list[str]:
    """渲染分节 XML。若为空则输出自闭合标签保留骨架结构。"""
    if not items:
        return [f"  <{tag}/>"]
    lines = [f"  <{tag}>"]
    for it in items:
        item_id = html.escape(str(it.get("item_id", "")))
        key = html.escape(str(it.get("item_key", "") or it.get("key", "")))
        content = html.escape(str(it.get("content", "")))
        attrs = [f'id="{item_id}"']
        if key:
            attrs.append(f'key="{key}"')
        attrs_str = " ".join(attrs)
        lines.append(f"    <item {attrs_str}>{content}</item>")
    lines.append(f"  </{tag}>")
    return lines


def build_container_xml() -> str:
    """构建 container 上下文 XML 字符串。

    - 当完全无条目时，直接返回自闭合的 `<container/>`。
    - 当有条目时，返回包含 `<preset>` 与 `<custom>` 的完整骨架结构。
    """
    preset_items = [it for it in _items if it.get("section") == "preset"]
    custom_items = [it for it in _items if it.get("section") == "custom"]

    if not preset_items and not custom_items:
        return "<container/>"

    lines = ["<container>"]
    lines.extend(_render_section("preset", preset_items))
    lines.extend(_render_section("custom", custom_items))
    lines.append("</container>")
    return "\n".join(lines)


async def add_item(
    section: str,
    content: str,
    key: str = "",
    metadata: dict[str, Any] | None = None,
    *,
    persist: bool = True,
) -> dict:
    """添加一条新条目到指定 section，并可选异步持久化到数据库。"""
    if section not in VALID_SECTIONS:
        raise ValueError(f"无效的 container section: {section}，必须为 {VALID_SECTIONS} 之一")

    now = int(time.time() * 1000)
    item_id = _next_id()
    entry = {
        "item_id": item_id,
        "section": section,
        "item_key": key,
        "content": content,
        "metadata": dict(metadata or {}),
        "created_at": now,
        "updated_at": now,
        "is_deleted": 0,
    }
    _items.append(entry)

    if persist:
        from database import write_container_item

        await write_container_item(
            item_id=item_id,
            section=section,
            item_key=key,
            content=content,
            metadata=metadata,
        )

    return entry


async def remove_item(item_id: str, *, persist: bool = True) -> bool:
    """根据 item_id 移除条目，并可选在数据库中进行软删除。"""
    global _items
    target_idx = -1
    for idx, it in enumerate(_items):
        if it.get("item_id") == item_id:
            target_idx = idx
            break

    if target_idx == -1:
        return False

    _items.pop(target_idx)

    if persist:
        from database import soft_delete_container_item

        await soft_delete_container_item(item_id)

    return True


async def clear(section: str | None = None) -> None:
    """清空指定 section 或全部内存条目。"""
    global _items
    if section is None:
        _items = []
    else:
        _items = [it for it in _items if it.get("section") != section]
