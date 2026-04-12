"""memory.py — 模型长期记忆管理（Phase 1：MemoryTriples + FTS5）

全局维护两个内存缓存池：
  _active_memories  — 模型主动写入（write_memory 工具触发），origin='active'
  _passive_memories — 系统自动归档（memory_archiver 触发），origin='passive'

每条记忆为 MemoryTriples 格式：
  {id, subject, predicate, object_text, origin, confidence, context,
   created_at, last_accessed, source, reason, conv_type, conv_id, conv_name}

启动时从 MemoryTriples 恢复；运行时通过 write_memory / delete_memory 工具更新。
每轮对话前由 session.prepare_memory_recall() 执行 FTS5 召回并将结果存在 session 上，
build_memory_xml() 优先使用召回结果注入 system prompt，不再全量加载。
"""

import html
import time
from datetime import datetime, timezone


# ── 全局状态 ────────────────────────────────────────────

_active_memories: list[dict] = []   # origin='active'：模型主动写入
_passive_memories: list[dict] = []  # origin='passive'：系统自动归档
_max_active: int = 8
_max_passive: int = 15

# 最近一轮 FTS5 召回命中的三元组 ID 集合，供 delete_memory 工具扩充可删除范围
# 模型在 <memory> 块中看到哪些 ID，此集合就包含哪些 ID
_last_recalled_ids: set[int] = set()


# ── 配置 & 启动恢复 ──────────────────────────────────────

def configure(max_active: int = 8, max_passive: int = 15) -> None:
    global _max_active, _max_passive
    _max_active = max_active
    _max_passive = max_passive


def restore(rows: list[dict]) -> None:
    """从 MemoryTriples 行列表恢复内存缓存（启动时调用）。

    rows 来自 load_all_triples()，已按 created_at ASC 排序。
    按 origin 分流，分别只保留最新 max_active / max_passive 条。
    """
    global _active_memories, _passive_memories
    valid = [r for r in rows if "id" in r]
    _active_memories  = [r for r in valid if r.get("origin", "passive") == "active" ][-_max_active:]
    _passive_memories = [r for r in valid if r.get("origin", "passive") == "passive"][-_max_passive:]


def get_all() -> list[dict]:
    """返回所有记忆（主动 + 被动），供工具/UI 展示用。"""
    return list(_active_memories) + list(_passive_memories)


def get_active_count() -> int:
    return len(_active_memories)


def get_max_active() -> int:
    return _max_active


def get_passive_count() -> int:
    return len(_passive_memories)


def get_max_passive() -> int:
    return _max_passive


def get_deletable_ids() -> list[str]:
    """返回当前模型可主动删除的记忆 ID 列表（字符串形式）。

    包含两个来源的并集：
      - _active_memories / _passive_memories 缓存
      - _last_recalled_ids（上一轮 FTS5 召回命中的条目）
    模型在 <memory> 块中看到的所有 ID 均覆盖在此范围内。
    """
    ids: set[str] = {str(m["id"]) for m in _active_memories if "id" in m}
    ids.update(str(i) for i in _last_recalled_ids)
    return sorted(ids, key=lambda x: int(x))


# ── 辅助 ─────────────────────────────────────────────────

def _ms() -> int:
    return int(time.time() * 1000)


def _age_text(created_at_ms: int, now: datetime) -> str:
    """把毫秒时间戳转换为自然语言时间差。"""
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
    """拼合模型描述与结构化溯源，例如「和小明聊天时 · 测试群(123456)」。"""
    if conv_id:
        suffix = f"{conv_name}({conv_id})" if conv_name else conv_id
        return f"{source} · {suffix}"
    return source


# ── XML 渲染 ─────────────────────────────────────────────

def _render_memory_block(
    tag: str,
    total: int,
    cap: int,
    entries: list[dict],
    now: datetime,
    recalled: list[dict] | None,
) -> str:
    """渲染单个 <active> 或 <passive> XML 块。"""
    if not entries:
        recalled_attr = ' recalled="0"' if recalled is not None else ""
        return f'<{tag} items="{total}/{cap}"{recalled_attr}/>'

    recalled_attr = f' recalled="{len(entries)}"' if recalled is not None else ""
    lines = [f'<{tag} items="{total}/{cap}"{recalled_attr}>']
    if tag == "passive":
        lines.append("  <des>无意间记住的事情，不一定是准确无误的</des>")
    for m in entries:

        subject = m.get("subject", "")
        predicate = m.get("predicate", "")
        content = m.get("object_text", m.get("content", ""))
        age = _age_text(m.get("created_at", 0), now)
        src = _source_display(
            m.get("source", ""),
            m.get("conv_name", ""),
            m.get("conv_id", ""),
        )
        if tag == "passive":
            lines.append('  <item>')
        else:
            mid = str(m.get("id", "?"))
            lines.append(f'  <item id="{mid}">')
        # predicate 为 [note] 时不展示（自由文本，subject/predicate 无额外信息量）
        if predicate and not (predicate.startswith("[") and predicate.endswith("]")):
            lines.append(f'    <subject>{html.escape(subject)}</subject>')
            lines.append(f'    <predicate>{html.escape(predicate)}</predicate>')
        lines.append(f'    <content>{html.escape(content)}</content>')
        lines.append(f'    <age>{age}</age>')
        if tag == "passive":
            conv_id   = m.get("conv_id", "")
            conv_name = m.get("conv_name", "")
            if conv_id:
                from_text = f"{conv_name}({conv_id})" if conv_name else conv_id
                lines.append(f'    <from>{html.escape(from_text)}</from>')
        else:
            lines.append(f'    <source>{html.escape(src)}</source>')
            lines.append(f'    <reason>{html.escape(m.get("reason", ""))}</reason>')
        lines.append('  </item>')
    lines.append(f'</{tag}>')
    return "\n".join(lines)


def build_memory_xml(
    now: datetime | None = None,
    recalled: list[dict] | None = None,
) -> str:
    """渲染完整 <active>…</active>\\n<passive>…</passive> 块，注入 system prompt 的 <memory> 内部。

    recalled: FTS5 召回后经精排的相关记忆列表（session.prepare_memory_recall() 的结果）。
              若为 None，回退到全量缓存（兼容无召回场景，如测试/直接调用）。
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if recalled is not None:
        # 按 origin 将召回结果分别投影到两个块，并更新全局 _last_recalled_ids
        recalled_active  = [r for r in recalled if r.get("origin", "passive") == "active"]
        recalled_passive = [r for r in recalled if r.get("origin", "passive") == "passive"]
        global _last_recalled_ids
        _last_recalled_ids = {r["id"] for r in recalled if "id" in r and r.get("origin", "passive") == "active"}
        active_entries  = recalled_active
        passive_entries = recalled_passive
    else:
        active_entries  = _active_memories
        passive_entries = _passive_memories

    active_block  = _render_memory_block(
        "active",  len(_active_memories),  _max_active,  active_entries,  now, recalled
    )
    passive_block = _render_memory_block(
        "passive", len(_passive_memories), _max_passive, passive_entries, now, recalled
    )
    return f"{active_block}\n{passive_block}"


# 兼容旧调用方（watcher_prompt 等），逐步迁移后可删除
def build_active_memory_xml(
    now: datetime | None = None,
    recalled: list[dict] | None = None,
) -> str:
    return build_memory_xml(now=now, recalled=recalled)


# ── FTS5 召回（async，在 build_system_prompt 前调用）───────

async def recall_memories(
    message_text: str,
    sender_id: str = "",
    config: dict | None = None,
) -> list[dict]:
    """按消息内容执行 FTS5 双通道召回，返回相关性排序后的记忆列表。

    在 ChatSession.prepare_memory_recall() 中调用，结果存于 session.recalled_memories。
    config 为 app_state.config 中的 memory 节点，若为 None 则使用默认超参。
    """
    from database import search_triples
    from llm.memory_tokenizer import build_fts_query

    cfg = config or {}
    ranking = cfg.get("ranking", {})
    alpha = float(ranking.get("alpha", 0.5))
    beta = float(ranking.get("beta", 0.3))
    gamma = float(ranking.get("gamma", 0.2))
    recall_top_k = int(ranking.get("recall_top_k", 20))
    inject_top_n = int(ranking.get("inject_top_n", 8))

    subject_filter = f"User:qq_{sender_id}" if sender_id else ""
    fts_query = build_fts_query(message_text)

    results = await search_triples(
        fts_query=fts_query,
        subject_filter=subject_filter,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        recall_top_k=recall_top_k,
    )
    top = results[:inject_top_n]
    return top


# ── 写入 / 删除 ──────────────────────────────────────────

async def add_memory(
    content: str,
    source: str,
    reason: str,
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    subject: str = "Self",
    predicate: str = "[note]",
    origin: str = "passive",
) -> int:
    """写入新记忆到 MemoryTriples，更新内存缓存并初始化 jieba 词典。

    origin: 'active'（模型主动写入）或 'passive'（系统自动归档）。
    对应池缓存超出上限时仅从内存中淘汰最旧条目，DB 记录保留（FTS5 仍可召回）。
    返回新三元组的 INTEGER id。
    """
    from database import write_triple as _db_write
    from llm.memory_tokenizer import tokenize as _tokenize, register_word as _register

    object_text_tok = _tokenize(content)
    _register(content)

    created_at = _ms()
    entry: dict = {
        "subject": subject,
        "predicate": predicate,
        "object_text": content,
        "origin": origin,
        "confidence": 0.6,
        "context": "truth",
        "created_at": created_at,
        "last_accessed": created_at,
        "source": source,
        "reason": reason,
        "conv_type": conv_type,
        "conv_id": conv_id,
        "conv_name": conv_name,
    }

    # 根据 origin 路由到对应池，超出上限时淘汰最旧条目
    if origin == "active":
        pool = _active_memories
        cap = _max_active
    else:
        pool = _passive_memories
        cap = _max_passive
    while len(pool) >= cap:
        pool.pop(0)

    triple_id = await _db_write(
        subject=subject,
        predicate=predicate,
        object_text=content,
        object_text_tok=object_text_tok,
        source=source,
        reason=reason,
        conv_type=conv_type,
        conv_id=conv_id,
        conv_name=conv_name,
        origin=origin,
    )
    entry["id"] = triple_id
    pool.append(entry)
    return triple_id


async def remove_memory(memory_id_str: str) -> bool:
    """从内存缓存移除并软删除 DB 记录。

    memory_id_str 为 str(triple_id)，与 delete_memory 工具的 enum 值对应。
    即使 ID 不在缓存中（仅在 FTS5 召回结果里出现），也能正确删除。
    """
    from database import soft_delete_triple as _db_delete
    global _active_memories

    try:
        triple_id = int(memory_id_str)
    except (ValueError, TypeError):
        return False

    # 被动记忆不可由模型主动遗忘
    if any(m.get("id") == triple_id for m in _passive_memories):
        return False

    # 从主动缓存池中移除（若存在）
    _active_memories = [m for m in _active_memories if m.get("id") != triple_id]
    # 无论是否在缓存中，都直接对 DB 执行软删除
    return await _db_delete(triple_id)

