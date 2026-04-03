"""memory.py — 模型长期记忆管理（Phase 1：MemoryTriples + FTS5）

全局维护一个内存缓存列表（_memories），条目为 MemoryTriples 格式：
  {id, subject, predicate, object_text, confidence, context,
   created_at, last_accessed, source, reason, conv_type, conv_id, conv_name}

启动时从 MemoryTriples 恢复；运行时通过 write_memory / delete_memory 工具更新。
每轮对话前由 session.prepare_memory_recall() 执行 FTS5 召回并将结果存在 session 上，
build_active_memory_xml() 优先使用召回结果注入 system prompt，不再全量加载。
"""

import html
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
    """从 MemoryTriples 行列表恢复内存缓存（启动时调用）。

    rows 来自 load_all_triples()，已按 created_at ASC 排序。
    只保留最新 max_entries 条。
    """
    global _memories
    _memories = [r for r in rows if "id" in r][-_max_entries:]


def get_all() -> list[dict]:
    return list(_memories)


def get_max_entries() -> int:
    return _max_entries


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

def build_active_memory_xml(
    now: datetime | None = None,
    recalled: list[dict] | None = None,
) -> str:
    """渲染 <active> XML 块，注入 system prompt 的 <memory> 内部。

    recalled: FTS5 召回后经精排的相关记忆列表（session.prepare_memory_recall() 的结果）。
              若为 None，回退到全量 _memories（兼容无召回的场景，如测试/直接调用）。
    每个 item 包含 id（供 delete_memory 使用）、content、source、age、reason。
    """
    if now is None:
        now = datetime.now(timezone.utc)

    total = len(_memories)
    cap = _max_entries
    entries = recalled if recalled is not None else _memories

    if not entries:
        tag = f'recalled="0" ' if recalled is not None else ""
        return f'<active items="{total}/{cap}" {tag}/>\n'.replace("  ", " ")

    recalled_count = len(entries)
    recalled_attr = f' recalled="{recalled_count}"' if recalled is not None else ""
    lines = [f'<active items="{total}/{cap}"{recalled_attr}>']
    for m in entries:
        mid = str(m.get("id", "?"))
        subject = m.get("subject", "")
        predicate = m.get("predicate", "")
        content = m.get("object_text", m.get("content", ""))
        age = _age_text(m.get("created_at", 0), now)
        src = _source_display(
            m.get("source", ""),
            m.get("conv_name", ""),
            m.get("conv_id", ""),
        )
        lines.append(f'  <item id="{mid}">')
        # structual predicate 为 [note] 时不展示（自由文本，subject/predicate 无额外信息量）
        if predicate and not (predicate.startswith("[") and predicate.endswith("]")):
            lines.append(f'    <subject>{html.escape(subject)}</subject>')
            lines.append(f'    <predicate>{html.escape(predicate)}</predicate>')
        lines.append(f'    <content>{html.escape(content)}</content>')
        lines.append(f'    <source>{html.escape(src)}</source>')
        lines.append(f'    <age>{age}</age>')
        lines.append(f'    <reason>{html.escape(m.get("reason", ""))}</reason>')
        lines.append('  </item>')
    lines.append("</active>")
    return "\n".join(lines)


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
    return results[:inject_top_n]


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
) -> int:
    """写入新记忆到 MemoryTriples，更新内存缓存并初始化 jieba 词典。

    predicate: 关系谓语，默认 "[note]"（自由文本）；Phase 2 起可传入结构化谓语如"喜欢"/"职业是"。
    超出 max_entries 上限时软删除最旧一条（DB + 缓存均删除）。
    返回新三元组的 INTEGER id。
    """
    from database import write_triple as _db_write, soft_delete_triple as _db_delete
    from llm.memory_tokenizer import tokenize as _tokenize, register_word as _register

    object_text_tok = _tokenize(content)
    _register(content)

    created_at = _ms()
    entry: dict = {
        "subject": subject,
        "predicate": predicate,
        "object_text": content,
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

    # 超出上限先软删除最旧的
    while len(_memories) >= _max_entries:
        oldest = _memories.pop(0)
        await _db_delete(oldest["id"])

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
    )
    entry["id"] = triple_id
    _memories.append(entry)
    return triple_id


async def remove_memory(memory_id_str: str) -> bool:
    """从内存缓存移除并软删除 DB 记录。

    memory_id_str 为 str(triple_id)，与 delete_memory 工具的 enum 值对应。
    """
    from database import soft_delete_triple as _db_delete
    global _memories

    triple_id: int | None = None
    for m in _memories:
        if str(m.get("id", "")) == memory_id_str:
            triple_id = m["id"]
            break

    if triple_id is None:
        return False

    _memories = [m for m in _memories if str(m.get("id", "")) != memory_id_str]
    await _db_delete(triple_id)
    return True

