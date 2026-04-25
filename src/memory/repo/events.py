"""Memory event repository implementation."""

from ._common import _connect, _ms, aiosqlite, logger

__all__ = ["load_events_for_recall", "write_event"]


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
) -> int:
	"""Write an event with its role edges and return event_id."""
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
				source, reason, conv_type, conv_id, conv_name, is_deleted)
			   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
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
		await db.commit()
	logger.debug("已写入 MemoryEvent id=%d type=%s context=%s", event_id, event_type, context_type)
	return event_id


async def load_events_for_recall(
	sender_entity: str = "",
	context_scope: str = "",
	limit: int = 6,
) -> list[dict]:
	"""Load recalled events and hydrate their role edges."""
	async with _connect() as db:
		db.row_factory = aiosqlite.Row

		scope_clause = (
			"AND (e.recall_scope='global' OR e.recall_scope=?)"
			if context_scope else ""
		)
		related_entities: list[str] = ["Bot:self"]
		if sender_entity:
			related_entities.append(sender_entity)
		ent_placeholders = ",".join("?" * len(related_entities))

		sql = f"""
			SELECT e.* FROM MemoryEvents e
			WHERE e.is_deleted=0
			  AND (
				  e.context_type='meta'
				  OR e.event_id IN (
					  SELECT DISTINCT event_id FROM MemoryRoles
					  WHERE entity IN ({ent_placeholders})
				  )
				  OR e.context_type='episodic'
			  )
			  {scope_clause}
			ORDER BY
				CASE e.context_type
					WHEN 'meta'     THEN 0
					WHEN 'contract' THEN 1
					ELSE 2
				END,
				e.occurred_at DESC
			LIMIT ?
		"""
		params: list = list(related_entities)
		if context_scope:
			params.append(context_scope)
		params.append(limit)

		async with db.execute(sql, params) as cur:
			events = [dict(row) for row in await cur.fetchall()]

		if not events:
			return []

		ids = [event["event_id"] for event in events]
		placeholders = ",".join("?" * len(ids))
		async with db.execute(
			f"SELECT * FROM MemoryRoles WHERE event_id IN ({placeholders})",
			ids,
		) as cur:
			role_rows = [dict(row) for row in await cur.fetchall()]

		roles_by_event: dict[int, list[dict]] = {}
		for role in role_rows:
			roles_by_event.setdefault(role["event_id"], []).append(role)
		for event in events:
			event["roles"] = roles_by_event.get(event["event_id"], [])

		now = _ms()
		await db.execute(
			f"UPDATE MemoryEvents SET last_accessed=? WHERE event_id IN ({placeholders})",
			[now, *ids],
		)
		await db.commit()

		return events