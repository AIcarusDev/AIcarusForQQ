"""container.py — 模型上下文契约 container 管理

全局维护一个内存中的 container 条目列表，包含 preset 与 custom 两个分节。
其上下文位置位于主模型尾部 user message 中、output_schema 之前。
启动时从数据库恢复，运行时通过底层接口更新，并在 prompt 组装时序列化为 XML。
全空状态下理论上永远输出: <container/>
"""

import asyncio
import html
import secrets
import time
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


_items: list[dict] = []
_update_lock = asyncio.Lock()
VALID_SECTIONS: tuple[str, ...] = ("preset", "custom")


@dataclass(frozen=True)
class ContainerContract:
    """由内容所属逻辑声明注入格式和语义。

    render 返回受信任的内部 XML（动态文本须自行转义）；返回空串则不注入，
    连同 des 一起省略。状态和持久化由提供方负责，容器只负责组合。
    """

    tag: str
    section: str
    description: str
    render: Callable[[datetime], str]

    def __post_init__(self) -> None:
        if self.section not in VALID_SECTIONS:
            raise ValueError(f"无效的 container section: {self.section}")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", self.tag):
            raise ValueError(f"无效的 container tag: {self.tag}")
        if not self.description.strip():
            raise ValueError("container description 不能为空")

    def build_xml(self, now: datetime) -> str:
        body = self.render(now)
        if not body.strip():
            return ""
        return (
            f"<{self.tag}>\n<des>{html.escape(self.description)}</des>\n"
            f"{body}\n</{self.tag}>"
        )


def restore(rows: list[dict]) -> None:
    """从外部数据源（如数据库）恢复内存条目列表。"""
    global _items, _update_lock
    _items = list(rows)
    _update_lock = asyncio.Lock()


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


def _render_section(tag: str, items: list[dict], blocks: list[str]) -> list[str]:
    """渲染分节 XML。若为空则输出自闭合标签保留骨架结构。"""
    if not items and not blocks:
        return [f"  <{tag}/>"]
    lines = [f"  <{tag}>"]
    for block in blocks:
        lines.extend(f"    {line}" for line in block.splitlines())
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


def build_container_xml(
    now: datetime | None = None,
    *,
    contracts: Iterable[ContainerContract] | None = None,
) -> str:
    """构建 container 上下文 XML 字符串。

    - 当完全无条目时，直接返回自闭合的 `<container/>`。
    - 当有条目时，返回包含 `<preset>` 与 `<custom>` 的完整骨架结构。
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if contracts is None:
        from .container_providers import CONTRACTS

        contracts = CONTRACTS
    blocks: dict[str, list[str]] = {section: [] for section in VALID_SECTIONS}
    for contract in contracts:
        if block := contract.build_xml(now):
            blocks[contract.section].append(block)

    preset_items = [it for it in _items if it.get("section") == "preset"]
    custom_items = [it for it in _items if it.get("section") == "custom"]

    if not preset_items and not custom_items and not any(blocks.values()):
        return "<container/>"

    lines = [
        "<container>",
        "  <des>This section stores context supplied by runtime features and tools.</des>",
    ]
    lines.extend(_render_section("preset", preset_items, blocks["preset"]))
    lines.extend(_render_section("custom", custom_items, blocks["custom"]))
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

    async with _update_lock:
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

        if persist:
            from database import write_container_item

            await write_container_item(
                item_id=item_id,
                section=section,
                item_key=key,
                content=content,
                metadata=metadata,
            )

        _items.append(entry)
        return entry


async def remove_item(
    item_id: str, *, section: str | None = None, persist: bool = True,
) -> bool:
    """按 ID 删除条目；指定 section 时只允许删除该分节的条目。"""
    async with _update_lock:
        target = next((it for it in _items if it.get("item_id") == item_id
                       and (section is None or it.get("section") == section)), None)
        if target is None:
            return False
        if persist:
            from database import soft_delete_container_item

            await soft_delete_container_item(item_id)
        _items.remove(target)
        return True


async def clear(section: str | None = None) -> None:
    """清空指定 section 或全部内存条目。"""
    global _items
    if section is None:
        _items = []
    else:
        _items = [it for it in _items if it.get("section") != section]
