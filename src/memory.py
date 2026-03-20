"""memory.py — 动态认知记忆系统

基于 SQLite FTS5 三元组的长效记忆，四阶段流水线：
  Stage 1: FTS5 粗排召回（Trigram 全文检索）
  Stage 2: 内存精排（BM25 翻转归一化 + 置信度 + 时间衰减）
  Stage 3: CTE 嵌套组装（深度 ≤ 5，仅穿透 truth）
  Stage 4: LLM 仲裁 + 拓扑写入

触发时机：意识流（processing_lock）结束后，以 asyncio.create_task 异步触发，
不阻塞主对话流程。
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone

import aiosqlite

from database import DB_PATH

logger = logging.getLogger("AICQ.memory")


# ══════════════════════════════════════════════════════════════════
#  Schema DDL
# ══════════════════════════════════════════════════════════════════

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS MemoryTriples (
    id            INTEGER  PRIMARY KEY AUTOINCREMENT,
    subject       TEXT     NOT NULL,
    predicate     TEXT     NOT NULL,
    object_text   TEXT,
    object_id     INTEGER  REFERENCES MemoryTriples(id),
    context       TEXT     NOT NULL DEFAULT 'truth',
    confidence    REAL     NOT NULL DEFAULT 0.6,
    source        TEXT     NOT NULL DEFAULT 'private',
    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (object_text IS NOT NULL AND object_id IS NULL) OR
        (object_text IS NULL  AND object_id IS NOT NULL)
    )
);

CREATE INDEX IF NOT EXISTS idx_mem_sub_pred ON MemoryTriples(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_mem_ctx_conf ON MemoryTriples(context, confidence);
CREATE INDEX IF NOT EXISTS idx_mem_obj_id   ON MemoryTriples(object_id);

CREATE VIRTUAL TABLE IF NOT EXISTS MemorySearch USING fts5(
    subject,
    predicate,
    object_text,
    content='MemoryTriples',
    content_rowid='id',
    tokenize="trigram"
);

CREATE TRIGGER IF NOT EXISTS mem_fts_insert
AFTER INSERT ON MemoryTriples BEGIN
    INSERT INTO MemorySearch(rowid, subject, predicate, object_text)
    VALUES (new.id, new.subject, new.predicate, new.object_text);
END;

CREATE TRIGGER IF NOT EXISTS mem_fts_delete
AFTER DELETE ON MemoryTriples BEGIN
    INSERT INTO MemorySearch(MemorySearch, rowid, subject, predicate, object_text)
    VALUES ('delete', old.id, old.subject, old.predicate, old.object_text);
END;

CREATE TRIGGER IF NOT EXISTS mem_fts_update
AFTER UPDATE OF subject, predicate, object_text ON MemoryTriples BEGIN
    INSERT INTO MemorySearch(MemorySearch, rowid, subject, predicate, object_text)
    VALUES ('delete', old.id, old.subject, old.predicate, old.object_text);
    INSERT INTO MemorySearch(rowid, subject, predicate, object_text)
    VALUES (new.id, new.subject, new.predicate, new.object_text);
END;

CREATE TRIGGER IF NOT EXISTS mem_prevent_cycle_insert
BEFORE INSERT ON MemoryTriples
FOR EACH ROW
WHEN NEW.object_id IS NOT NULL
 AND NEW.object_id >= (SELECT IFNULL(MAX(id), 0) + 1 FROM MemoryTriples)
BEGIN
    SELECT RAISE(ABORT, 'Time arrow violation: object_id must point to a historical record.');
END;

CREATE TRIGGER IF NOT EXISTS mem_prevent_cycle_update
BEFORE UPDATE OF object_id ON MemoryTriples
FOR EACH ROW
WHEN NEW.object_id IS NOT NULL AND NEW.object_id >= NEW.id
BEGIN
    SELECT RAISE(ABORT, 'Time arrow violation: object_id must point to an older record.');
END;
"""


# ══════════════════════════════════════════════════════════════════
#  LLM 仲裁相关常量
# ══════════════════════════════════════════════════════════════════

# 记忆仲裁 LLM 的输出 Schema
_MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "write": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["subject", "predicate", "object_text", "context", "confidence"],
                "properties": {
                    "subject":     {"type": "string"},
                    "predicate":   {"type": "string"},
                    "object_text": {"type": "string"},
                    "context":     {"type": "string", "enum": ["truth", "hypothetical"]},
                    "confidence":  {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
        "outdate":   {"type": "array", "items": {"type": "integer"}},
        "reinforce": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["write", "outdate", "reinforce"],
}

# 记忆仲裁 system prompt 模板
_MEMORY_SYSTEM_PROMPT = """\
你是记忆仲裁模块。请从对话中提取值得长期记忆的认知三元组，并裁决现有记忆的有效性。

## 三元组规则
- subject：使用下方映射表中的 person_id，不要用昵称或 QQ 号。
- predicate：自然语言谓语，允许带时态（如"最近不喜欢"/"一直讨厌"/"搬家到了"）。
- object_text：原样保留宾语，禁止强行归一化（保留"红彤彤的水果"，不要改成"苹果"）。
- context：
    - "truth" — 发言者认真表述的事实或偏好。
    - "hypothetical" — 明显的假设、玩笑、"如果"句、反问、角色扮演。
- confidence 初始值参考：
    - 私聊事实 = 0.6
    - 群聊事实 = 0.45
    - 假设/玩笑再 −0.1
    - 转述他人的话再 −0.1

## 提取筛选原则
1. 只提取**稳定的个人认知信息**（偏好、属性、关系、习惯、观点、经历等）。
2. 纯问候/感谢/日常闲聊/系统操作 → 不提取。
3. Bot 自身的发言通常不提取（除非在描述自己的真实属性）。
4. 若新信息与现有记忆**完全一致** → 不重复写入，将现有记忆的 id 放入 reinforce。
5. 若对话中新信息**否定/更新**了现有记忆 → 新信息放入 write，旧记忆 id 放入 outdate。

## 人员 person_id 映射
{person_map_text}

## 现有相关记忆（供冲突裁决参考）
{existing_triples_text}
"""

# 记忆仲裁专用的轻量 generation 参数
_MEMORY_GEN: dict = {
    "temperature": 0.2,
    "max_output_tokens": 2048,
    "max_tool_rounds": 0,
}


# ══════════════════════════════════════════════════════════════════
#  初始化
# ══════════════════════════════════════════════════════════════════

async def init_memory_db() -> None:
    """在现有数据库中创建记忆表（幂等，若已存在则跳过）。"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript(_INIT_SQL)
        await db.commit()
    logger.info("[memory] 记忆表初始化完成")


# ══════════════════════════════════════════════════════════════════
#  Stage 1: FTS5 粗排召回
# ══════════════════════════════════════════════════════════════════

def _extract_keywords(context_messages: list[dict]) -> list[str]:
    """从最近的上下文消息中提取用于 FTS5 检索的候选关键词。

    取发言者昵称 + 消息内容中的 CJK 词（≥2字）和 ASCII 词（≥3字），
    去重后限制 25 个，全局联想用，不按发言人过滤。
    """
    tokens: list[str] = []
    for msg in context_messages[-20:]:
        name = msg.get("sender_name", "")
        if name:
            tokens.append(name)
        content = msg.get("content", "") or ""
        for word in re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\u4e00-\u9fa5]{2,}|[a-zA-Z0-9_]{3,}', content):
            tokens.append(word)
    seen: set[str] = set()
    result: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
        if len(result) >= 25:
            break
    return result


def _build_fts5_query(keywords: list[str]) -> str:
    """将关键词列表构建为 FTS5 MATCH 查询字符串（双引号转义 + OR 连接）。"""
    if not keywords:
        return ""
    terms = ['"' + kw.replace('"', '') + '"' for kw in keywords if kw]
    return " OR ".join(terms)


async def _stage1_recall(keywords: list[str], limit: int = 50) -> list[dict]:
    """Stage 1: FTS5 Trigram 粗排召回。"""
    query = _build_fts5_query(keywords)
    if not query:
        return []
    sql = """
        SELECT t.id, t.subject, t.predicate, t.object_text, t.object_id,
               t.context, t.confidence, t.source,
               t.created_at, t.last_accessed, fts.rank
        FROM MemoryTriples t
        JOIN MemorySearch fts ON t.id = fts.rowid
        WHERE MemorySearch MATCH ?
        ORDER BY fts.rank ASC
        LIMIT ?
    """
    rows: list[dict] = []
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, (query, limit)) as cur:
                async for row in cur:
                    rows.append(dict(row))
    except Exception as e:
        logger.warning("[memory] Stage1 FTS5 召回异常: %s", e)
    return rows


# ══════════════════════════════════════════════════════════════════
#  Stage 2: 内存精排
# ══════════════════════════════════════════════════════════════════

def _stage2_rescore(
    rows: list[dict],
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.001,
    top_k: int = 20,
) -> list[dict]:
    """Stage 2: BM25 极性翻转归一化 + 置信度 + 时间衰减，截取 Top-K。

    BM25 rank 为负数（越小越相关），先翻转归一化为正向分数：
        normalize(r_i) = (max - r_i) / (max - min)
    除零保护：若 max == min，统一归一化为 1.0。
    """
    if not rows:
        return []

    now_ts = datetime.now(timezone.utc).timestamp()
    ranks = [float(r["rank"]) for r in rows]
    r_min, r_max = min(ranks), max(ranks)
    denom = r_max - r_min

    def _parse_ts(s: str | None) -> float:
        if not s:
            return now_ts
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            return now_ts

    scored: list[tuple[float, dict]] = []
    for row in rows:
        norm_bm25 = 1.0 if denom == 0 else (r_max - float(row["rank"])) / denom
        confidence = float(row.get("confidence") or 0.6)
        delta_days = (now_ts - _parse_ts(row.get("last_accessed"))) / 86400.0
        score = alpha * norm_bm25 + beta * confidence - gamma * delta_days
        scored.append((score, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_k]]


# ══════════════════════════════════════════════════════════════════
#  Stage 3: CTE 嵌套组装
# ══════════════════════════════════════════════════════════════════

async def _stage3_assemble(top_rows: list[dict]) -> list[dict]:
    """Stage 3: 对含 object_id 的 Triple 做 CTE 递归穿透。

    深度限制 ≤ 5，只穿透 context='truth' 的节点，
    防止大量 outdated 历史被拖入上下文（Token 爆炸防护）。
    """
    if not top_rows:
        return []

    # 只对含嵌套引用的 Triple 做穿透
    seed_ids = [r["id"] for r in top_rows if r.get("object_id")]
    if not seed_ids:
        return top_rows

    placeholders = ",".join("?" * len(seed_ids))
    cte_sql = f"""
        WITH RECURSIVE MemoryChain AS (
            SELECT id, subject, predicate, object_text, object_id,
                   context, confidence, source, created_at, last_accessed, 1 AS depth
            FROM MemoryTriples
            WHERE id IN ({placeholders})
            UNION ALL
            SELECT t.id, t.subject, t.predicate, t.object_text, t.object_id,
                   t.context, t.confidence, t.source, t.created_at, t.last_accessed,
                   c.depth + 1
            FROM MemoryTriples t
            JOIN MemoryChain c ON t.id = c.object_id
            WHERE t.context = 'truth' AND c.depth < 5
        )
        SELECT DISTINCT id, subject, predicate, object_text, object_id,
               context, confidence, source, created_at, last_accessed
        FROM MemoryChain
    """
    extra_rows: list[dict] = []
    existing_ids = {r["id"] for r in top_rows}
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(cte_sql, seed_ids) as cur:
                async for row in cur:
                    d = dict(row)
                    if d["id"] not in existing_ids:
                        extra_rows.append(d)
                        existing_ids.add(d["id"])
    except Exception as e:
        logger.warning("[memory] Stage3 CTE 穿透异常: %s", e)
    return top_rows + extra_rows


# ══════════════════════════════════════════════════════════════════
#  Stage 4: LLM 仲裁 + 写入
# ══════════════════════════════════════════════════════════════════

def _format_triples_for_prompt(rows: list[dict]) -> str:
    """将召回的 Triple 列表格式化为可读文本，供 LLM 参考。"""
    if not rows:
        return "（暂无相关记忆）"
    lines: list[str] = []
    for r in rows:
        ctx_tag = f" [{r['context']}]" if r.get("context") != "truth" else ""
        conf = float(r.get("confidence") or 0.0)
        obj = r.get("object_text") or f"→Triple#{r.get('object_id')}"
        lines.append(
            f"  #{r['id']} {r['subject']} | {r['predicate']} | {obj}{ctx_tag} conf={conf:.2f}"
        )
    return "\n".join(lines)


def _format_person_map(person_map: dict[str, tuple[str, str]]) -> str:
    """将 person_map 格式化为 prompt 中的映射说明段落。"""
    if not person_map:
        return "  （无人员映射信息）"
    lines: list[str] = []
    for sender_id, (person_id, display_name) in person_map.items():
        lines.append(f"  QQ:{sender_id} ({display_name}) → {person_id}")
    return "\n".join(lines)


async def _stage4_arbitrate_and_write(
    adapter,
    chat_xml: str,
    assembled: list[dict],
    person_map: dict[str, tuple[str, str]],
    source: str,
) -> None:
    """Stage 4: 将对话 XML + 召回记忆投喂给 LLM 仲裁，然后拓扑写入结果。"""
    system_prompt = _MEMORY_SYSTEM_PROMPT.format(
        person_map_text=_format_person_map(person_map),
        existing_triples_text=_format_triples_for_prompt(assembled),
    )

    def _builder(tool_budget, rounds_used=0, max_rounds=None, tool_budget_suffix=""):
        return system_prompt

    try:
        result, _, _, _, _ = await asyncio.to_thread(
            adapter.call,
            _builder,
            chat_xml,
            _MEMORY_GEN,
            _MEMORY_SCHEMA,
        )
    except Exception as e:
        logger.error("[memory] LLM 仲裁调用异常: %s", e)
        return

    if not result:
        logger.warning("[memory] LLM 仲裁返回 None，跳过写入")
        return

    write_list: list[dict] = result.get("write") or []
    outdate_ids: list[int] = [
        int(i) for i in (result.get("outdate") or [])
        if isinstance(i, (int, float))
    ]
    reinforce_ids: list[int] = [
        int(i) for i in (result.get("reinforce") or [])
        if isinstance(i, (int, float))
    ]

    await _commit_memory(write_list, outdate_ids, reinforce_ids, source)
    logger.info(
        "[memory] 仲裁完成: 写入 %d 条，降权 %d 条，强化 %d 条",
        len(write_list), len(outdate_ids), len(reinforce_ids),
    )


async def _commit_memory(
    write_list: list[dict],
    outdate_ids: list[int],
    reinforce_ids: list[int],
    source: str,
) -> None:
    """将仲裁结果原子性地写入数据库。"""
    now_str = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")

        # 1. 降权旧记忆：置信度压到 0.2 以下，标记为 outdated
        for oid in outdate_ids:
            await db.execute(
                """UPDATE MemoryTriples
                   SET context='outdated', confidence=MIN(confidence, 0.2), last_accessed=?
                   WHERE id=? AND context != 'outdated'""",
                (now_str, oid),
            )

        # 2. 强化被召回且被采纳的旧记忆（艾宾浩斯强化）
        for rid in reinforce_ids:
            await db.execute(
                """UPDATE MemoryTriples
                   SET confidence=MIN(1.0, confidence + 0.05), last_accessed=?
                   WHERE id=?""",
                (now_str, rid),
            )

        # 3. 写入新 Triple（纯文本节点，不支持嵌套 object_id —— 留给 v2）
        for item in write_list:
            subj = str(item.get("subject") or "").strip()
            pred = str(item.get("predicate") or "").strip()
            obj_text = str(item.get("object_text") or "").strip()
            ctx = str(item.get("context") or "truth")
            if ctx not in ("truth", "hypothetical"):
                ctx = "truth"
            conf = float(item.get("confidence") or 0.6)
            conf = max(0.0, min(1.0, conf))

            if not subj or not pred or not obj_text:
                continue

            try:
                await db.execute(
                    """INSERT INTO MemoryTriples
                       (subject, predicate, object_text, context, confidence,
                        source, created_at, last_accessed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (subj, pred, obj_text, ctx, conf, source, now_str, now_str),
                )
            except Exception as e:
                logger.warning(
                    "[memory] 写入 Triple 异常 subj=%s pred=%s: %s", subj, pred, e
                )

        await db.commit()


# ══════════════════════════════════════════════════════════════════
#  对外主接口
# ══════════════════════════════════════════════════════════════════

async def run_memory_pipeline(
    context_messages: list[dict],
    chat_xml: str,
    adapter,
    person_map: dict[str, tuple[str, str]],
    source: str = "private",
) -> None:
    """记忆流水线主入口，以 asyncio.create_task 异步触发。

    Args:
        context_messages: session.context_messages 的快照（在锁内已复制）。
        chat_xml:         本轮对话的 XML 字符串快照（get_chat_log_display() 结果）。
        adapter:          LLM 适配器实例（与主 LLM 共享）。
        person_map:       {sender_id: (person_id, display_name)} 映射。
        source:           'private' | 'group'，影响初始置信度参考值。
    """
    try:
        t0 = time.monotonic()
        keywords = _extract_keywords(context_messages)
        if not keywords:
            logger.debug("[memory] 无有效关键词，跳过本轮流水线")
            return

        logger.debug("[memory] 流水线启动，关键词: %s", keywords[:10])

        raw_rows = await _stage1_recall(keywords)
        top_rows = _stage2_rescore(raw_rows)
        assembled = await _stage3_assemble(top_rows)

        logger.debug(
            "[memory] 召回 %d 条 → 精排 %d 条 → 组装 %d 条",
            len(raw_rows), len(top_rows), len(assembled),
        )

        await _stage4_arbitrate_and_write(adapter, chat_xml, assembled, person_map, source)
        logger.info("[memory] 流水线完成，耗时 %.2fs", time.monotonic() - t0)

    except Exception:
        logger.exception("[memory] 流水线未预期异常")
