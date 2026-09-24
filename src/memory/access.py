"""Explicit text search/read across memory stores; no automatic recall policy."""
from __future__ import annotations

from memory.active import store
from memory.tokenizer import tokenize


def memory_id(item: dict) -> str:
    """Use the same identity in tool results and the automatic memory block."""
    if item.get("memory_kind") == "active":
        return str(item.get("memory_id") or "")
    if item.get("memory_kind") == "summary":
        return str(item.get("summary_id") or item.get("event_id") or "")
    event_id = item.get("event_id")
    return f"E{event_id:06d}" if isinstance(event_id, int) and event_id > 0 else ""


def _sources(con) -> list[str]:
    tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    sources = []
    if "ActiveMemoryEntries" in tables:
        sources.append("SELECT printf('M%06d', id) AS id, 'active' AS kind, content, "
                       "revision, created_at, updated_at FROM ActiveMemoryEntries WHERE status='active'")
    if "MemoryEvents" in tables:
        sources.append("SELECT printf('E%06d', event_id) AS id, 'event' AS kind, summary AS content, "
                       "NULL AS revision, created_at, occurred_at AS updated_at "
                       "FROM MemoryEvents WHERE is_deleted=0")
    if "MemorySummaryCache" in tables:
        sources.append("SELECT summary_id AS id, 'summary' AS kind, summary AS content, "
                       "NULL AS revision, created_at_ms AS created_at, updated_at_ms AS updated_at "
                       "FROM MemorySummaryCache WHERE status='ready' AND summary<>''")
    return sources


def search(path: str, query: str, *, limit: int = 10, offset: int = 0, literal: bool = False) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    terms = [query] if literal else list(dict.fromkeys([query, *tokenize(query).split()]))[:32]
    terms = [term.casefold() for term in terms if term.strip()]
    matches = " + ".join("(instr(lower(content), ?) > 0)" for _ in terms)
    id_match = "0" if literal else "100 * (id = ?)"
    params = [*terms, *([] if literal else [query]), limit, offset]
    with store.connection(path) as con:
        sources = _sources(con)
        if not sources:
            return []
        rows = con.execute(
            "WITH memories AS (" + " UNION ALL ".join(sources) + ") "
            f"SELECT *, ({matches}) + {id_match} AS matches FROM memories "
            "WHERE matches>0 ORDER BY matches DESC, updated_at DESC, id ASC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [dict(row) for row in rows]


def read(path: str, item_id: str) -> dict | None:
    """Read the complete text of a currently available memory."""
    with store.connection(path) as con:
        sources = _sources(con)
        if not sources:
            return None
        row = con.execute(
            "WITH memories AS (" + " UNION ALL ".join(sources) + ") SELECT * FROM memories WHERE id=?",
            (item_id,),
        ).fetchone()
        return dict(row) if row else None
