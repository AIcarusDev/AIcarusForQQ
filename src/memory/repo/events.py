"""Memory event repository implementation."""

from ._common import _connect, _ms, aiosqlite, logger

__all__ = [
	"load_events_for_recall",
	"merge_event_occurrence",
	"prefetch_candidates_for_archiver",
	"soft_delete_event",
	"write_event",
]


VALID_ROLES: frozenset[str] = frozenset({
	"agent", "patient", "theme", "recipient",
	"instrument", "location", "time", "attribute",
})

VALID_CONTEXT_TYPES: frozenset[str] = frozenset({
	"meta", "contract", "episodic", "hypothetical",
})

VALID_POLARITY: frozenset[str] = frozenset({"positive", "negative"})
VALID_MODALITY: frozenset[str] = frozenset({"actual", "hypothetical", "possible"})


async def write_event(
	event_type: str,
	summary: str,
	summary_tok: str = "",
	polarity: str = "positive",
	modality: str = "actual",
	confidence: float = 0.6,
	context_type: str = "episodic",
	recall_scope: str = "global",
	source: str = "",
	reason: str = "",
	conv_type: str = "",
	conv_id: str = "",
	conv_name: str = "",
	roles: list[dict] | None = None,
	supersedes: int | None = None,
) -> int:
	"""Write an event with its role edges and return event_id.

	If `supersedes` is provided, the referenced event is soft-deleted in the
	same transaction and the new event records the link.
	"""
	if context_type not in VALID_CONTEXT_TYPES:
		context_type = "episodic"
	if polarity not in VALID_POLARITY:
		polarity = "positive"
	if modality not in VALID_MODALITY:
		modality = "actual"
	confidence = max(0.0, min(1.0, float(confidence)))

	now = _ms()
	async with _connect() as db:
		cur = await db.execute(
			"""INSERT INTO MemoryEvents
			   (event_type, summary, summary_tok, polarity, modality,
				confidence, context_type, recall_scope, occurred_at, last_accessed,
				last_seen_at, occurrences, supersedes,
				source, reason, conv_type, conv_id, conv_name, is_deleted)
			   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
			(
				event_type,
				summary,
				summary_tok,
				polarity,
				modality,
				confidence,
				context_type,
				recall_scope,
				now,
				now,
				now,
				1,
				int(supersedes) if supersedes else None,
				source,
				reason,
				conv_type,
				conv_id,
				conv_name,
			),
		)
		event_id: int = cur.lastrowid or 0

		if roles:
			for role in roles:
				role_name = (role.get("role") or "").strip().lower()
				if role_name not in VALID_ROLES:
					logger.debug("跳过非法 role：%s", role_name)
					continue
				entity = role.get("entity") or None
				value_text = role.get("value_text") or None
				value_tok = role.get("value_tok") or ""
				target_event = role.get("target_event")
				if entity is None and value_text is None and target_event is None:
					continue
				if target_event is not None and int(target_event) >= event_id:
					logger.debug(
						"跳过非法 target_event=%s（>= 当前事件 id=%s）",
						target_event,
						event_id,
					)
					target_event = None
					if entity is None and value_text is None:
						continue
				await db.execute(
					"""INSERT INTO MemoryRoles
					   (event_id, role, entity, value_text, value_tok, target_event)
					   VALUES (?,?,?,?,?,?)""",
					(event_id, role_name, entity, value_text, value_tok, target_event),
				)

		# supersedes：把旧事件软删，并在旧事件上写 merge_into=新 id 便于追溯
		if supersedes:
			await db.execute(
				"UPDATE MemoryEvents SET is_deleted=1, merge_into=? "
				"WHERE event_id=? AND is_deleted=0",
				(event_id, int(supersedes)),
			)

		await db.commit()
	logger.debug(
		"已写入 MemoryEvent id=%d type=%s context=%s supersedes=%s",
		event_id, event_type, context_type, supersedes,
	)
	return event_id


async def merge_event_occurrence(event_id: int) -> bool:
	"""把同一事实的再次观测合并进现有事件：occurrences+1, last_seen_at=now。

	置信度按对数衰减小幅上涨：0.95 是硬上限。
	"""
	now = _ms()
	async with _connect() as db:
		async with db.execute(
			"SELECT confidence, occurrences FROM MemoryEvents "
			"WHERE event_id=? AND is_deleted=0",
			(int(event_id),),
		) as cur:
			row = await cur.fetchone()
		if not row:
			return False
		conf, occ = float(row[0]), int(row[1])
		# 每次重复观测 +0.03，硬顶 0.95；首次 0.6 → 1 次合并 0.63 → ...
		new_conf = min(0.95, conf + 0.03)
		new_occ = occ + 1
		await db.execute(
			"UPDATE MemoryEvents SET occurrences=?, last_seen_at=?, confidence=? "
			"WHERE event_id=?",
			(new_occ, now, new_conf, int(event_id)),
		)
		await db.commit()
	logger.debug("合并 MemoryEvent id=%d occ=%d conf=%.2f", event_id, new_occ, new_conf)
	return True


# ── 召回 ──────────────────────────────────────────────────

# 融合权重（详见 V2.0 §4）
_W_BM25     = 0.5
_W_ENTITY   = 0.3
_W_RECENCY  = 0.1
_W_CONF     = 0.1
_RECENT_WINDOW_MS = 30 * 24 * 3600 * 1000  # 30 天


async def _query_meta(db) -> list[dict]:
	"""必拉：所有 context_type='meta' 的事件。"""
	async with db.execute(
		"SELECT * FROM MemoryEvents WHERE is_deleted=0 AND context_type='meta'"
	) as cur:
		return [dict(r) for r in await cur.fetchall()]


async def _query_by_entity(
	db, related_entities: list[str], context_scope: str, limit: int,
) -> list[dict]:
	if not related_entities:
		return []
	ent_placeholders = ",".join("?" * len(related_entities))
	scope_clause = (
		"AND (e.recall_scope='global' OR e.recall_scope=?)"
		if context_scope else ""
	)
	sql = f"""
		SELECT DISTINCT e.* FROM MemoryEvents e
		WHERE e.is_deleted=0
		  AND e.event_id IN (
			  SELECT event_id FROM MemoryRoles
			  WHERE entity IN ({ent_placeholders})
		  )
		  {scope_clause}
		ORDER BY e.occurred_at DESC
		LIMIT ?
	"""
	params: list = list(related_entities)
	if context_scope:
		params.append(context_scope)
	params.append(limit)
	async with db.execute(sql, params) as cur:
		return [dict(r) for r in await cur.fetchall()]


async def _query_by_fts(
	db, fts_query: str, context_scope: str, limit: int,
) -> list[tuple[dict, float]]:
	"""FTS5 候选 + 归一化 BM25 分。bm25() 返回越小越相关。"""
	if not fts_query:
		return []
	scope_clause = (
		"AND (e.recall_scope='global' OR e.recall_scope=?)"
		if context_scope else ""
	)
	sql = f"""
		SELECT e.*, bm25(MemorySearch) AS rank
		FROM MemorySearch
		JOIN MemoryEvents e ON e.event_id = MemorySearch.rowid
		WHERE MemorySearch MATCH ?
		  AND e.is_deleted=0
		  {scope_clause}
		ORDER BY rank
		LIMIT ?
	"""
	params: list = [fts_query]
	if context_scope:
		params.append(context_scope)
	params.append(limit)
	try:
		async with db.execute(sql, params) as cur:
			rows = [dict(r) for r in await cur.fetchall()]
	except Exception:
		# FTS5 query 解析失败（如非法字符），降级为空
		logger.debug("FTS5 查询失败：%r", fts_query, exc_info=True)
		return []
	if not rows:
		return []
	# 归一化：rank 越小越好；映射到 [0,1]，最相关 = 1.0
	ranks = [r["rank"] for r in rows]
	r_min, r_max = min(ranks), max(ranks)
	span = (r_max - r_min) or 1.0
	out: list[tuple[dict, float]] = []
	for r in rows:
		norm = 1.0 - (r["rank"] - r_min) / span  # 最小 rank → 1.0
		out.append((r, norm))
	return out


async def _query_episodic_recent(
	db, context_scope: str, limit: int,
) -> list[dict]:
	scope_clause = (
		"AND (e.recall_scope='global' OR e.recall_scope=?)"
		if context_scope else ""
	)
	sql = f"""
		SELECT * FROM MemoryEvents e
		WHERE e.is_deleted=0 AND e.context_type='episodic'
		  {scope_clause}
		ORDER BY e.occurred_at DESC
		LIMIT ?
	"""
	params: list = []
	if context_scope:
		params.append(context_scope)
	params.append(limit)
	async with db.execute(sql, params) as cur:
		return [dict(r) for r in await cur.fetchall()]


def _fuse(
	cand_meta: list[dict],
	cand_entity: list[dict],
	cand_fts: list[tuple[dict, float]],
	cand_episodic: list[dict],
	related_entities_set: set[str],
	now: int,
) -> list[dict]:
	"""按 §4 融合分对候选事件去重排序。返回排好序的事件 dict 列表。"""
	scores: dict[int, float] = {}
	store: dict[int, dict] = {}
	entity_hit_ids: set[int] = {ev["event_id"] for ev in cand_entity}

	def _add(ev: dict, score: float) -> None:
		eid = ev["event_id"]
		store[eid] = ev
		scores[eid] = max(scores.get(eid, 0.0), score)

	# 必入：meta 用极大分（确保排第一档）
	for ev in cand_meta:
		_add(ev, 100.0)

	# contract 也加 0.2（不强制必入，但占优）
	def _bonus(ev: dict) -> float:
		base = 0.0
		if ev.get("context_type") == "contract":
			base += 0.2
		# recency
		oa = int(ev.get("occurred_at") or 0)
		if oa and now - oa <= _RECENT_WINDOW_MS:
			base += _W_RECENCY * (1.0 - (now - oa) / _RECENT_WINDOW_MS)
		# confidence
		base += _W_CONF * float(ev.get("confidence") or 0.0)
		# occurrences 加成（重复说过的事更可信）
		occ = int(ev.get("occurrences") or 1)
		if occ > 1:
			base += min(0.2, 0.05 * (occ - 1))
		# 极短 summary 降权（纯打招呼类，让位给有信息量的事件）
		if len(str(ev.get("summary") or "")) < 8:
			base -= 0.15
		return base

	# entity 命中
	for ev in cand_entity:
		_add(ev, _W_ENTITY + _bonus(ev))

	# FTS5 命中
	for ev, bm in cand_fts:
		score = _W_BM25 * bm + _bonus(ev)
		# entity + FTS 双命中：再加成（这是最强信号）
		if ev["event_id"] in entity_hit_ids:
			score += 0.3
		_add(ev, score)

	# episodic 兜底（只在前两路都很空时给小分）
	if len(scores) < 4:
		for ev in cand_episodic:
			_add(ev, 0.05 + _bonus(ev))

	ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
	return [store[eid] for eid, _ in ordered]


async def load_events_for_recall(
	sender_entity: str = "",
	context_scope: str = "",
	limit: int = 6,
	query: str = "",
) -> list[dict]:
	"""三路融合召回：meta 必入 + 实体边 + FTS5 + episodic 兜底。

	`query`：用户当前消息文本（被动召回）或显式关键词（主动召回）。
	空 query 时降级为旧行为（仅实体边 + meta + episodic）。
	"""
	from memory.tokenizer import build_fts_query as _build_q

	related_entities: list[str] = ["Bot:self"]
	if sender_entity:
		related_entities.append(sender_entity)
	related_entities_set = set(related_entities)

	fts_q = _build_q(query) if query else ""

	now = _ms()
	async with _connect() as db:
		db.row_factory = aiosqlite.Row

		# 三路并联（顺序串行，单连接复用）
		cand_meta = await _query_meta(db)
		cand_entity = await _query_by_entity(
			db, related_entities, context_scope, limit * 2,
		)
		cand_fts = await _query_by_fts(
			db, fts_q, context_scope, limit * 2,
		) if fts_q else []
		cand_episodic = await _query_episodic_recent(
			db, context_scope, limit,
		)

		ordered = _fuse(
			cand_meta, cand_entity, cand_fts, cand_episodic,
			related_entities_set, now,
		)
		top = ordered[:limit]
		if not top:
			return []

		ids = [ev["event_id"] for ev in top]
		placeholders = ",".join("?" * len(ids))
		async with db.execute(
			f"SELECT * FROM MemoryRoles WHERE event_id IN ({placeholders})",
			ids,
		) as cur:
			role_rows = [dict(r) for r in await cur.fetchall()]

		roles_by_event: dict[int, list[dict]] = {}
		for role in role_rows:
			roles_by_event.setdefault(role["event_id"], []).append(role)
		for ev in top:
			ev["roles"] = roles_by_event.get(ev["event_id"], [])

		await db.execute(
			f"UPDATE MemoryEvents SET last_accessed=? WHERE event_id IN ({placeholders})",
			[now, *ids],
		)
		await db.commit()

	return top


async def prefetch_candidates_for_archiver(
	sender_entity: str,
	context_scope: str,
	dialogue_text: str,
	limit: int = 8,
) -> list[dict]:
	"""Read-Before-Write：给 archiver 预取「可能与本轮重复」的候选事件。

	策略与 load_events_for_recall 一致，但 query 用对话原文而非用户单句，
	且不刷新 last_accessed（archiver 只是参考，不算真实读取）。
	"""
	from memory.tokenizer import build_fts_query as _build_q

	related_entities: list[str] = ["Bot:self"]
	if sender_entity:
		related_entities.append(sender_entity)

	fts_q = _build_q(dialogue_text) if dialogue_text else ""
	now = _ms()

	async with _connect() as db:
		db.row_factory = aiosqlite.Row

		cand_meta: list[dict] = []  # archiver 不需要 meta（它写不进 meta）
		cand_entity = await _query_by_entity(
			db, related_entities, context_scope, limit * 2,
		)
		cand_fts = await _query_by_fts(
			db, fts_q, context_scope, limit * 2,
		) if fts_q else []
		cand_episodic = await _query_episodic_recent(
			db, context_scope, limit,
		)

		ordered = _fuse(
			cand_meta, cand_entity, cand_fts, cand_episodic,
			set(related_entities), now,
		)
		top = ordered[:limit]
		if not top:
			return []

		ids = [ev["event_id"] for ev in top]
		placeholders = ",".join("?" * len(ids))
		async with db.execute(
			f"SELECT * FROM MemoryRoles WHERE event_id IN ({placeholders})",
			ids,
		) as cur:
			role_rows = [dict(r) for r in await cur.fetchall()]

		roles_by_event: dict[int, list[dict]] = {}
		for role in role_rows:
			roles_by_event.setdefault(role["event_id"], []).append(role)
		for ev in top:
			ev["roles"] = roles_by_event.get(ev["event_id"], [])

	return top


async def soft_delete_event(event_id: int) -> bool:
	"""Soft-delete an event row (role edges retained for audit)."""
	async with _connect() as db:
		cur = await db.execute(
			"UPDATE MemoryEvents SET is_deleted=1 WHERE event_id=? AND is_deleted=0",
			(int(event_id),),
		)
		await db.commit()
		return cur.rowcount > 0
