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
	"episodic", "hypothetical", "evidence",
})

VALID_MODALITY: frozenset[str] = frozenset({"actual", "hypothetical", "possible"})


async def write_event(
	event_type: str,
	summary: str,
	summary_tok: str = "",
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
	if modality not in VALID_MODALITY:
		modality = "actual"
	confidence = max(0.0, min(1.0, float(confidence)))

	now = _ms()
	async with _connect() as db:
		cur = await db.execute(
			"""INSERT INTO MemoryEvents
			   (event_type, summary, summary_tok, modality,
				confidence, context_type, recall_scope, occurred_at, last_accessed,
				last_seen_at, occurrences, supersedes,
				source, reason, conv_type, conv_id, conv_name, is_deleted)
			   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
			(
				event_type,
				summary,
				summary_tok,
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

		_roles_tok_parts: list[str] = []
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
				if value_tok:
					_roles_tok_parts.append(value_tok)

		# 聚合角色 value_tok → roles_tok，供 FTS5 主题词索引（me_fts_roles_update 触发器同步）
		if _roles_tok_parts:
			await db.execute(
				"UPDATE MemoryEvents SET roles_tok=? WHERE event_id=?",
				(" ".join(_roles_tok_parts), event_id),
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
_W_BM25      = 0.5
_W_ENTITY    = 0.3
_W_RECENCY   = 0.1
_W_CONF      = 0.1
_W_SPREADING = 0.25   # 泼溅激活权重（联想召回第一跳折扣）
_RECENT_WINDOW_MS = 30 * 24 * 3600 * 1000  # 30 天
# 实体出现在 >= 此数量的事件中视为「枢纽节点」，不作为扩散种子（防止过度泛化召回）
# 参考 entitySystem: log10(degree)×0.3 的枢纽惩罚思路，此处改为硬阈值过滤
_HUB_ENTITY_THRESHOLD = 25

# 显著事件类型：承诺/拒绝/情感/纠错 在召回时额外加权
# 对应用户问题：「对于印象深刻的内容加强召回权重」
_SALIENT_EVENT_TYPES: frozenset[str] = frozenset({
	"promise", "refuse", "dislike", "correct", "teach", "feel",
})


def _event_bonus(ev: dict, now: int) -> float:
	"""事件额外得分（时近性 + 置信度 + 重复次数 + 显著性）。

	从 _fuse 内部 _bonus 提取为模块级函数，供泼溅激活等外部逻辑复用。
	"""
	base = 0.0
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
	# 显著事件加成：承诺/拒绝/情感类更值得记住
	if str(ev.get("event_type") or "") in _SALIENT_EVENT_TYPES:
		base += 0.08
	return base


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
		WHERE e.is_deleted=0 AND e.context_type IN ('episodic','evidence')
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


async def _query_spreading_activation(
	db,
	seed_entities: list[str],
	exclude_ids: set[int],
	context_scope: str,
	limit: int,
) -> list[dict]:
	"""1-hop 泼溅激活：找到与 seed_entities 共享实体的事件（不含 exclude_ids 已有的）。

	对应用户问题：「黑裙子→黑衣服→黑寡妇→超级英雄」式联想链路。
	当 top-k 结果涉及某实体 X，系统会查找所有同样涉及 X 的其他事件，
	以折扣分加入召回候选，让相关记忆自然"浮现"。
	"""
	if not seed_entities:
		return []
	ent_placeholders = ",".join("?" * len(seed_entities))
	excl_clause = ""
	excl_params: list = []
	if exclude_ids:
		excl_placeholders = ",".join("?" * len(exclude_ids))
		excl_clause = f"AND e.event_id NOT IN ({excl_placeholders})"
		excl_params = list(exclude_ids)
	scope_clause = (
		"AND (e.recall_scope='global' OR e.recall_scope=?)"
		if context_scope else ""
	)
	sql = f"""
		SELECT DISTINCT e.* FROM MemoryEvents e
		WHERE e.is_deleted=0
		  {excl_clause}
		  AND e.event_id IN (
			  SELECT event_id FROM MemoryRoles
			  WHERE entity IN ({ent_placeholders})
		  )
		  {scope_clause}
		ORDER BY e.occurred_at DESC
		LIMIT ?
	"""
	params: list = excl_params + list(seed_entities)
	if context_scope:
		params.append(context_scope)
	params.append(limit)
	async with db.execute(sql, params) as cur:
		return [dict(r) for r in await cur.fetchall()]


def _fuse(
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

	# entity 命中
	for ev in cand_entity:
		_add(ev, _W_ENTITY + _event_bonus(ev, now))

	# FTS5 命中
	for ev, bm in cand_fts:
		score = _W_BM25 * bm + _event_bonus(ev, now)
		# entity + FTS 双命中：再加成（这是最强信号）
		if ev["event_id"] in entity_hit_ids:
			score += 0.3
		_add(ev, score)

	# episodic 兜底（只在前两路都很空时给小分）
	if len(scores) < 4:
		for ev in cand_episodic:
			_add(ev, 0.05 + _event_bonus(ev, now))

	ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
	return [store[eid] for eid, _ in ordered]


async def load_events_for_recall(
	sender_entity: str = "",
	context_scope: str = "",
	limit: int = 6,
	query: str = "",
) -> list[dict]:
	"""融合召回：实体边 + FTS5 + episodic 兜底。

	`query`：用户当前消息文本（被动召回）或显式关键词（主动召回）。
	空 query 时降级为旧行为（仅实体边 + episodic）。
	"""
	from memory.tokenizer import build_fts_query as _build_q

	related_entities: list[str] = ["self"]
	if sender_entity:
		related_entities.append(sender_entity)
	related_entities_set = set(related_entities)

	fts_q = _build_q(query) if query else ""

	now = _ms()
	async with _connect() as db:
		db.row_factory = aiosqlite.Row

		# 多路候选并联（顺序串行，单连接复用）
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
			cand_entity, cand_fts, cand_episodic,
			related_entities_set, now,
		)
		top = ordered[:limit]
		if not top:
			return []

		# ── 多跳泼溅激活（联想召回）────────────────────────────────────
		# 参考 entitySystem v2 的三项设计移植到 SQLite：
		#   1. Modality 屏障：hypothetical 事件的实体永不作为扩散种子
		#   2. 枢纽惩罚：出现在 >=_HUB_ENTITY_THRESHOLD 条事件的实体跳过
		#   3. 2-hop：hop-1 新实体再扩散一跳，折扣平方（_W_SPREADING²）
		top_ids = [ev["event_id"] for ev in top]

		# Step 1: 收集 top 事件的实体（跳过 modality=hypothetical 的事件）
		non_hyp_ids = [ev["event_id"] for ev in top if ev.get("modality") != "hypothetical"]
		if non_hyp_ids:
			ph_non_hyp = ",".join("?" * len(non_hyp_ids))
			async with db.execute(
				f"SELECT DISTINCT entity FROM MemoryRoles "
				f"WHERE event_id IN ({ph_non_hyp}) AND entity IS NOT NULL",
				non_hyp_ids,
			) as sp_cur:
				spreading_ents = [row["entity"] for row in await sp_cur.fetchall()]
		else:
			spreading_ents = []

		# Step 2: 过滤 novel 实体 + 枢纽惩罚
		novel_ents = [e for e in spreading_ents if e not in related_entities_set]
		if novel_ents:
			ent_ph = ",".join("?" * len(novel_ents))
			async with db.execute(
				f"SELECT entity, COUNT(DISTINCT event_id) AS cnt "
				f"FROM MemoryRoles WHERE entity IN ({ent_ph}) GROUP BY entity",
				novel_ents,
			) as hub_cur:
				hub_counts = {row["entity"]: int(row["cnt"]) for row in await hub_cur.fetchall()}
			novel_ents = [e for e in novel_ents if hub_counts.get(e, 0) < _HUB_ENTITY_THRESHOLD]

		if novel_ents:
			# Hop-1
			sp_cands = await _query_spreading_activation(
				db, novel_ents, set(top_ids), context_scope, limit,
			)
			if sp_cands:
				sp_store: dict[int, dict] = {ev["event_id"]: ev for ev in top}
				n_top = len(top)
				sp_scores: dict[int, float] = {
					ev["event_id"]: 1.0 - (i / max(n_top - 1, 1)) * 0.9
					for i, ev in enumerate(top)
				}
				for ev in sp_cands:
					eid = ev["event_id"]
					sp_scores[eid] = _W_SPREADING + _event_bonus(ev, now)
					sp_store[eid] = ev

				# Hop-2: 从 hop-1 结果的非假设实体再扩散，折扣平方
				all_seen_ids = set(top_ids) | {ev["event_id"] for ev in sp_cands}
				hop1_non_hyp_ids = [
					ev["event_id"] for ev in sp_cands
					if ev.get("modality") != "hypothetical"
				]
				if hop1_non_hyp_ids:
					known_ents = related_entities_set | set(novel_ents)
					ph_hop1 = ",".join("?" * len(hop1_non_hyp_ids))
					async with db.execute(
						f"SELECT DISTINCT entity FROM MemoryRoles "
						f"WHERE event_id IN ({ph_hop1}) AND entity IS NOT NULL",
						hop1_non_hyp_ids,
					) as h2_cur:
						hop2_raw = [row["entity"] for row in await h2_cur.fetchall()]
					hop2_novel = [e for e in hop2_raw if e not in known_ents]
					if hop2_novel:
						ent_ph2 = ",".join("?" * len(hop2_novel))
						async with db.execute(
							f"SELECT entity, COUNT(DISTINCT event_id) AS cnt "
							f"FROM MemoryRoles WHERE entity IN ({ent_ph2}) GROUP BY entity",
							hop2_novel,
						) as hub2_cur:
							hub2_counts = {row["entity"]: int(row["cnt"]) for row in await hub2_cur.fetchall()}
						hop2_novel = [e for e in hop2_novel if hub2_counts.get(e, 0) < _HUB_ENTITY_THRESHOLD]
						if hop2_novel:
							sp_cands2 = await _query_spreading_activation(
								db, hop2_novel, all_seen_ids, context_scope, limit,
							)
							for ev in sp_cands2:
								eid = ev["event_id"]
								sp_scores[eid] = _W_SPREADING * _W_SPREADING + _event_bonus(ev, now)
								sp_store[eid] = ev

				ordered_sp = sorted(
					sp_scores.items(), key=lambda kv: kv[1], reverse=True,
				)
				top = [sp_store[eid] for eid, _ in ordered_sp][:limit]
		# ────────────────────────────────────────────────────────────────

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

	related_entities: list[str] = ["self"]
	if sender_entity:
		related_entities.append(sender_entity)

	fts_q = _build_q(dialogue_text) if dialogue_text else ""
	now = _ms()

	async with _connect() as db:
		db.row_factory = aiosqlite.Row

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
			cand_entity, cand_fts, cand_episodic,
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
