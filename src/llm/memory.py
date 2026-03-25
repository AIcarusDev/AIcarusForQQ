"""memory.py — 模型长期记忆管理

全局维护一个内存中的记忆列表，支持写入/删除/渲染为 XML。
启动时从数据库恢复，运行时通过 write_memory / delete_memory 工具调用更新。

设计与 activity_log.py 一致：内存缓存 + 异步持久化到 DB。
"""

import secrets
import time
from datetime import datetime, timezone


# ── 全局状态 ────────────────────────────────────────────

_memories: list[dict] = []
_max_entries: int = 15


# ── 配置 & 启动恢复 ──────────────────────────────────────

def configure(max_entries: int) -> None:
    global _max_entries
    _max_entries = max_entries


def restore(rows: list[dict]) -> None:
    """从数据库行列表恢复内存缓存（启动时调用）。"""
    global _memories
    _memories = list(rows)


def get_all() -> list[dict]:
    return list(_memories)


# ── 辅助 ─────────────────────────────────────────────────

def _age_text(created_at_ms: int, now: datetime) -> str:
    """把毫秒时间戳转换为自然语言时间差，例如"3天前"。"""
    delta_sec = int(now.timestamp() - created_at_ms / 1000)
    if delta_sec < 60:
        return "刚刚"
    elif delta_sec < 3600:
        return f"{delta_sec // 60}分钟前"
    elif delta_sec < 86400:
        return f"{delta_sec // 3600}小时前"
    elif delta_sec < 86400 * 7:
        return f"{delta_sec // 86400}天前"
    elif delta_sec < 86400 * 30:
        return f"{delta_sec // (86400 * 7)}周前"
    else:
        return f"{delta_sec // (86400 * 30)}个月前"


def _source_display(source: str, conv_name: str, conv_id: str) -> str:
    """拼合模型描述与结构化溯源，例如"和小明聊天时 · 测试群(123456)"。"""
    if conv_id:
        suffix = f"{conv_name}({conv_id})" if conv_name else conv_id
        return f"{source} · {suffix}"
    return source


# ── XML 渲染 ─────────────────────────────────────────────

def build_active_memory_xml(now: datetime | None = None) -> str:
    """渲染 <active> XML 块，注入 system prompt 的 <memory> 内部。

    每个 item 包含：id（供 delete_memory 使用）、content、source、age、reason。
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if not _memories:
        return (
            "<active/>\n"
        )
    lines = ["<active>"]
    for m in _memories:
        mid = m["memory_id"]
        age = _age_text(m["created_at"], now)
        src = _source_display(m["source"], m.get("conv_name", ""), m.get("conv_id", ""))
        lines.append(f'  <item id="{mid}">')
        lines.append(f'    <content>{m["content"]}</content>')
        lines.append(f'    <source>{src}</source>')
        lines.append(f'    <age>{age}</age>')
        lines.append(f'    <reason>{m["reason"]}</reason>')
        lines.append('  </item>')
    lines.append("</active>")
    return "\n".join(lines)


# ── 写入 / 删除 ──────────────────────────────────────────

def _next_id() -> str:
    """生成唯一记忆 ID，格式为 mem_xxxxxxxx（8位十六进制）。"""
    used = {m["memory_id"] for m in _memories}
    for _ in range(20):
        candidate = f"mem_{secrets.token_hex(4)}"
        if candidate not in used:
            return candidate
    return f"mem_{secrets.token_hex(4)}"


async def add_memory(
    content: str,
    source: str,
    reason: str,
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
) -> str:
    """写入新记忆，更新内存缓存并持久化到 DB。超出上限时软删除最旧的。

    返回新记忆的 memory_id。
    """
    from database import write_memory as _db_write, soft_delete_memory as _db_delete
    memory_id = _next_id()
    created_at = int(time.time() * 1000)
    entry = {
        "memory_id": memory_id,
        "created_at": created_at,
        "content": content,
        "source": source,
        "reason": reason,
        "conv_type": conv_type,
        "conv_id": conv_id,
        "conv_name": conv_name,
    }
    # 超出上限先删最旧的
    while len(_memories) >= _max_entries:
        oldest = _memories.pop(0)
        await _db_delete(oldest["memory_id"])
    _memories.append(entry)
    await _db_write(
        memory_id=memory_id,
        content=content,
        source=source,
        reason=reason,
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
    )
    return memory_id


async def remove_memory(memory_id: str) -> bool:
    """从内存缓存中移除一条记忆，并软删除 DB 记录。返回是否找到。"""
    from database import soft_delete_memory as _db_delete
    global _memories
    before = len(_memories)
    _memories = [m for m in _memories if m["memory_id"] != memory_id]
    found = len(_memories) < before
    if found:
        await _db_delete(memory_id)
    return found
