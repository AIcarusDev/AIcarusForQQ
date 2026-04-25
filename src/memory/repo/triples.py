"""Memory triple repository implementation."""

from ._common import _connect, _ms, aiosqlite, logger

__all__ = [
    "load_all_triples",
    "search_triples",
    "soft_delete_triple",
    "update_triple_confidence",
    "write_triple",
]


async def write_triple(
    subject: str,
    predicate: str,
    object_text: str,
    object_text_tok: str,
    source: str = "",
    reason: str = "",
    conv_type: str = "",
    conv_id: str = "",
    conv_name: str = "",
    confidence: float = 0.6,
    context: str = "truth",
    origin: str = "passive",
    recall_scope: str = "global",
    cluster_id: int | None = None,
) -> int:
    """Write a memory triple and return its integer id."""
    now = _ms()
    async with _connect() as db:
        cur = await db.execute(
            """INSERT OR IGNORE INTO MemoryTriples
               (subject, predicate, object_text, object_text_tok,
                context, confidence, created_at, last_accessed,
                source, reason, conv_type, conv_id, conv_name, origin,
                recall_scope, cluster_id, is_deleted)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
            (
                subject,
                predicate,
                object_text,
                object_text_tok,
                context,
                confidence,
                now,
                now,
                source,
                reason,
                conv_type,
                conv_id,
                conv_name,
                origin,
                recall_scope,
                cluster_id,
            ),
        )
        await db.commit()
        if cur.rowcount > 0:
            new_id: int = cur.lastrowid or 0
            logger.debug(
                "已写入 MemoryTriple id=%d subject=%s origin=%s scope=%s",
                new_id,
                subject,
                origin,
                recall_scope,
            )
        else:
            async with db.execute(
                "SELECT id FROM MemoryTriples"
                " WHERE subject=? AND predicate=? AND object_text=? AND is_deleted=0 LIMIT 1",
                (subject, predicate, object_text),
            ) as cur2:
                row = await cur2.fetchone()
            new_id = row[0] if row else 0
            if new_id:
                await db.execute(
                    "UPDATE MemoryTriples SET last_accessed=? WHERE id=?",
                    (now, new_id),
                )
                await db.commit()
            logger.debug(
                "MemoryTriple 已存在 id=%d subject=%s（INSERT OR IGNORE 去重，已刷新 last_accessed）",
                new_id,
                subject,
            )
    return new_id


async def soft_delete_triple(triple_id: int) -> bool:
    """Soft-delete one memory triple."""
    async with _connect() as db:
        cur = await db.execute(
            "UPDATE MemoryTriples SET is_deleted=1 WHERE id=? AND is_deleted=0",
            (triple_id,),
        )
        await db.commit()
    return cur.rowcount > 0


async def update_triple_confidence(
    triple_ids: list[int],
    delta: float,
    cap: float = 1.0,
) -> None:
    """Adjust confidence for recalled triples and refresh last_accessed."""
    now = _ms()
    async with _connect() as db:
        for triple_id in triple_ids:
            await db.execute(
                """UPDATE MemoryTriples
                   SET confidence = MIN(?, confidence + ?),
                       last_accessed = ?
                   WHERE id = ? AND is_deleted = 0""",
                (cap, delta, now, triple_id),
            )
        await db.commit()


async def load_all_triples() -> list[dict]:
    """Load all non-deleted triples ordered by created_at ascending."""
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, subject, predicate, object_text, object_text_tok,
                      confidence, context, created_at, last_accessed,
                      source, reason, conv_type, conv_id, conv_name, origin,
                      recall_scope, cluster_id
               FROM MemoryTriples
               WHERE is_deleted = 0
               ORDER BY created_at ASC"""
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def search_triples(
    fts_query: str,
    subject_filter: str = "",
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    recall_top_k: int = 20,
    context_scope: str = "",
) -> list[dict]:
    """Recall triples via FTS5 and rank them with bm25/confidence/time decay."""
    import time as _time

    if not fts_query:
        return await _load_recent_triples(recall_top_k, context_scope=context_scope)

    results_a: list[dict] = []
    results_b: list[dict] = []
    scope_clause = (
        "AND (t.recall_scope = 'global' OR t.recall_scope = ?)"
        if context_scope else ""
    )
    cols = """t.id, t.subject, t.predicate, t.object_text,
              t.confidence, t.context, t.created_at, t.last_accessed,
              t.source, t.reason, t.conv_type, t.conv_id, t.conv_name, t.origin,
              t.recall_scope, t.cluster_id,
              COALESCE(c.confidence, t.confidence) AS effective_confidence,
              fts.rank AS rank"""

    async with _connect() as db:
        db.row_factory = aiosqlite.Row

        if subject_filter:
            try:
                params_a: list = [fts_query, subject_filter]
                if context_scope:
                    params_a.append(context_scope)
                async with db.execute(
                    f"""SELECT {cols}
                        FROM MemorySearch fts
                        JOIN MemoryTriples t ON fts.rowid = t.id
                        LEFT JOIN MemoryClusters c ON t.cluster_id = c.cluster_id
                        WHERE MemorySearch MATCH ?
                          AND t.is_deleted = 0
                          AND t.subject = ?
                          {scope_clause}
                        ORDER BY fts.rank ASC
                        LIMIT 50""",
                    params_a,
                ) as cur:
                    results_a = [dict(row) for row in await cur.fetchall()]
            except Exception as exc:
                logger.debug("FTS5 通道 A 查询失败（忽略）: %s", exc)

        try:
            params_b: list = [fts_query]
            if context_scope:
                params_b.append(context_scope)
            async with db.execute(
                f"""SELECT {cols}
                    FROM MemorySearch fts
                    JOIN MemoryTriples t ON fts.rowid = t.id
                    LEFT JOIN MemoryClusters c ON t.cluster_id = c.cluster_id
                    WHERE MemorySearch MATCH ?
                      AND t.is_deleted = 0
                      {scope_clause}
                    ORDER BY fts.rank ASC
                    LIMIT 50""",
                params_b,
            ) as cur:
                results_b = [dict(row) for row in await cur.fetchall()]
        except Exception as exc:
            logger.debug("FTS5 通道 B 查询失败（忽略）: %s", exc)

    seen: set[int] = set()
    merged: list[dict] = []
    for row in results_a:
        if row["id"] not in seen:
            seen.add(row["id"])
            merged.append(row)
    for row in results_b:
        if row["id"] not in seen:
            seen.add(row["id"])
            merged.append(row)

    if not merged:
        return []

    now_ms = int(_time.time() * 1000)
    ranks = [row["rank"] for row in merged]
    max_r = max(ranks)
    min_r = min(ranks)
    eps = 1e-5

    def _bm25(rank: float) -> float:
        if abs(max_r - min_r) < eps:
            return 1.0
        return (max_r - rank) / (max_r - min_r)

    scored: list[tuple[float, dict]] = []
    for row in merged:
        delta_days = (now_ms - row["last_accessed"]) / 86_400_000
        effective_confidence = row.get("effective_confidence") or row["confidence"]
        score = (
            alpha * _bm25(row["rank"])
            + beta * effective_confidence
            - gamma * delta_days
        )
        scored.append((score, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_results = [item for _, item in scored[:recall_top_k]]

    cluster_ids_found: set[int] = {
        row["cluster_id"] for row in top_results if row.get("cluster_id") is not None
    }
    if cluster_ids_found:
        already_ids: set[int] = {row["id"] for row in top_results}
        placeholders = ",".join("?" * len(cluster_ids_found))
        scope_extra = "AND (t.recall_scope = 'global' OR t.recall_scope = ?)" if context_scope else ""
        params_extra: list = list(cluster_ids_found) + ([context_scope] if context_scope else [])
        async with _connect() as db:
            db.row_factory = aiosqlite.Row
            try:
                async with db.execute(
                    f"""SELECT t.id, t.subject, t.predicate, t.object_text,
                               t.confidence, t.context, t.created_at, t.last_accessed,
                               t.source, t.reason, t.conv_type, t.conv_id, t.conv_name,
                               t.origin, t.recall_scope, t.cluster_id,
                               COALESCE(c.confidence, t.confidence) AS effective_confidence,
                               0.0 AS rank
                        FROM MemoryTriples t
                        LEFT JOIN MemoryClusters c ON t.cluster_id = c.cluster_id
                        WHERE t.cluster_id IN ({placeholders})
                          AND t.is_deleted = 0
                          {scope_extra}
                        LIMIT 30""",
                    params_extra,
                ) as cur:
                    cluster_members = [dict(row) for row in await cur.fetchall()]
                for member in cluster_members:
                    if member["id"] not in already_ids:
                        top_results.append(member)
                        already_ids.add(member["id"])
            except Exception as exc:
                logger.debug("聚类激活传播查询失败（忽略）: %s", exc)

    return top_results


async def _load_recent_triples(limit: int, context_scope: str = "") -> list[dict]:
    if context_scope:
        sql = """SELECT id, subject, predicate, object_text,
                      confidence, context, created_at, last_accessed,
                      source, reason, conv_type, conv_id, conv_name, origin,
                      recall_scope, cluster_id, confidence AS effective_confidence,
                      0.0 AS rank
               FROM MemoryTriples
               WHERE is_deleted = 0
                 AND (recall_scope = 'global' OR recall_scope = ?)
               ORDER BY created_at DESC
               LIMIT ?"""
        params: tuple = (context_scope, limit)
    else:
        sql = """SELECT id, subject, predicate, object_text,
                      confidence, context, created_at, last_accessed,
                      source, reason, conv_type, conv_id, conv_name, origin,
                      recall_scope, cluster_id, confidence AS effective_confidence,
                      0.0 AS rank
               FROM MemoryTriples
               WHERE is_deleted = 0
               ORDER BY created_at DESC
               LIMIT ?"""
        params = (limit,)
    async with _connect() as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]