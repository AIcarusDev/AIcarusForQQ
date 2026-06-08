"""search_session.py - Search accessible QQ sessions by name."""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

from pypinyin import Style, lazy_pinyin

from qq_adapter.access_control import is_session_allowed_by_config
from qq_adapter.conversation import TEMP_CONV_TYPE
from tools._async_bridge import run_coroutine_sync

logger = logging.getLogger("AICQ.tools")

ALWAYS_AVAILABLE: bool = True

DECLARATION: dict = {
    "name": "search_session",
    "description": "按名称搜索当前平台中可访问的好友、群聊或已登记临时会话。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "名称关键词。",
            },
            "type": {
                "type": "string",
                "enum": ["any", "private", "group", "temp"],
                "description": "搜索范围，默认 any。",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回数，默认 5。",
            },
        },
        "required": ["query"],
    },
}

REQUIRES_CONTEXT: list[str] = ["config"]

_VALID_TYPES = {"any", "private", "group", "temp"}
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_KEEP_CHARS_RE = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)


@dataclass
class _Candidate:
    conv_type: str
    conv_id: str
    name: str
    aliases: set[str] = field(default_factory=set)
    source_rank: int = 0

    def merge(self, name: str, aliases: Iterable[str], source_rank: int) -> None:
        display_name = str(name or "")
        if display_name.strip() and (not self.name or source_rank >= self.source_rank):
            self.name = display_name
            self.source_rank = source_rank
        for alias in aliases:
            alias = str(alias or "").strip()
            if alias:
                self.aliases.add(alias)


def _main_loop_fallback() -> Any:
    try:
        import app_state

        return getattr(app_state, "main_loop", None)
    except Exception:
        return None


def _qq_adapter_client_fallback() -> Any:
    try:
        import app_state

        return getattr(app_state, "qq_adapter_client", None)
    except Exception:
        return None


def _valid_platform_id(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.isdigit()


def _session_allowed(config: dict[str, Any], conv_type: str, conv_id: str) -> bool:
    qq_adapter_cfg = config.get("qq_adapter", {}) if isinstance(config, dict) else {}
    return is_session_allowed_by_config(qq_adapter_cfg, conv_type, conv_id)


def _result_type(conv_type: str) -> str:
    return "temp" if conv_type == TEMP_CONV_TYPE else conv_type


def _normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(_KEEP_CHARS_RE.findall(text))


def _symbol_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    chars: list[str] = []
    for char in text:
        if char.isspace() or char == "_":
            continue
        if char.isalnum() or _CJK_RE.fullmatch(char):
            continue
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            chars.append(char)
    return "".join(chars)


def _collapse_repeats(value: str) -> str:
    result: list[str] = []
    last = ""
    for char in value:
        if char != last:
            result.append(char)
            last = char
    return "".join(result)


def _pinyin_text(value: str) -> str:
    if not value or not _CJK_RE.search(value):
        return ""
    return "".join(lazy_pinyin(value, style=Style.NORMAL, errors="ignore"))


def _pinyin_initials(value: str) -> str:
    if not value or not _CJK_RE.search(value):
        return ""
    return "".join(lazy_pinyin(value, style=Style.FIRST_LETTER, errors="ignore"))


def _is_single_cjk(value: str) -> bool:
    return len(value) == 1 and _CJK_RE.fullmatch(value) is not None


def _ngrams(value: str, sizes: tuple[int, ...]) -> Counter[str]:
    vector: Counter[str] = Counter()
    if not value:
        return vector
    for size in sizes:
        if len(value) < size:
            continue
        for index in range(len(value) - size + 1):
            vector[f"{size}:{value[index:index + size]}"] += 1
    if not vector:
        vector[f"1:{value}"] += 1
    return vector


def _sparse_vector(value: str) -> Counter[str]:
    norm = _normalize_text(value)
    vector = _ngrams(norm, (1, 2, 3))
    collapsed = _collapse_repeats(norm)
    if collapsed and collapsed != norm:
        vector.update({f"r:{k}": v for k, v in _ngrams(collapsed, (1, 2)).items()})
    if not _is_single_cjk(norm):
        pinyin = _pinyin_text(norm)
        if pinyin:
            vector.update({f"p:{k}": v for k, v in _ngrams(pinyin, (2, 3, 4)).items()})
        initials = _pinyin_initials(norm)
        if initials:
            vector.update({f"i:{k}": v for k, v in _ngrams(initials, (1, 2)).items()})
    return vector


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in shared)
    if dot <= 0:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _field_score(query: str, alias: str) -> float:
    q = _normalize_text(query)
    a = _normalize_text(alias)
    q_symbols = _symbol_text(query)
    if not q and q_symbols:
        a_symbols = _symbol_text(alias)
        if not a_symbols:
            return 0.0
        if q_symbols == a_symbols:
            return 1.0
        if q_symbols in a_symbols:
            coverage = len(q_symbols) / max(len(a_symbols), 1)
            return min(0.9, 0.78 + coverage * 0.12)
        q_collapsed_symbols = _collapse_repeats(q_symbols)
        if q_collapsed_symbols and q_collapsed_symbols in a_symbols:
            coverage = len(q_collapsed_symbols) / max(len(a_symbols), 1)
            return min(0.84, 0.7 + coverage * 0.1)
        return SequenceMatcher(None, q_symbols, a_symbols).ratio() * 0.72

    if not q or not a:
        return 0.0
    if q == a:
        return 1.0

    q_collapsed = _collapse_repeats(q)
    a_collapsed = _collapse_repeats(a)
    if q_collapsed and q_collapsed == a_collapsed:
        return 0.96
    if q_collapsed and q_collapsed == a:
        return 0.94

    single_cjk_query = _is_single_cjk(q)
    q_py = _pinyin_text(q)
    a_py = _pinyin_text(a)
    if not single_cjk_query and q_py and a_py and q_py == a_py:
        return 0.93
    q_initials = _pinyin_initials(q)
    a_initials = _pinyin_initials(a)
    if not single_cjk_query and q_initials and a_initials and q_initials == a_initials:
        return 0.89

    if a.startswith(q):
        return 0.9 if len(q) > 1 else 0.82
    if q_collapsed and a.startswith(q_collapsed):
        return 0.76 if len(q) > len(q_collapsed) else 0.8
    if q in a:
        coverage = len(q) / max(len(a), 1)
        return min(0.86, 0.72 + coverage * 0.16)
    if q_collapsed and q_collapsed in a:
        coverage = len(q_collapsed) / max(len(a), 1)
        return min(0.72, 0.56 + coverage * 0.12)
    if not single_cjk_query and q_py and a_py:
        if a_py.startswith(q_py):
            return 0.84
        if q_py in a_py:
            return 0.78
    if not single_cjk_query and not _CJK_RE.search(q) and a_py:
        if q == a_py:
            return 0.91
        if a_py.startswith(q):
            return 0.82
        if q in a_py:
            return 0.74
    if not single_cjk_query and q_initials and a_initials:
        if a_initials.startswith(q_initials):
            return 0.78
        if q_initials in a_initials:
            return 0.7

    return SequenceMatcher(None, q, a).ratio() * 0.68


def _score_alias(query: str, query_vector: Counter[str], alias: str) -> float:
    field_score = _field_score(query, alias)
    vector_score = _cosine(query_vector, _sparse_vector(alias))
    blended = field_score * 0.82 + vector_score * 0.18
    return max(field_score, blended, vector_score * 0.72)


def _candidate_score(query: str, query_vector: Counter[str], candidate: _Candidate) -> float:
    aliases = {candidate.name, *candidate.aliases}
    return max((_score_alias(query, query_vector, alias) for alias in aliases if alias), default=0.0)


def _candidate_payload(candidate: _Candidate, score: float) -> dict[str, Any]:
    return {
        "type": _result_type(candidate.conv_type),
        "id": candidate.conv_id,
        "name": candidate.name or candidate.conv_id,
        "score": round(score, 3),
    }


def _add_candidate(
    candidates: dict[tuple[str, str], _Candidate],
    *,
    conv_type: str,
    conv_id: Any,
    name: Any,
    aliases: Iterable[Any] = (),
    source_rank: int = 0,
) -> None:
    conv_id = str(conv_id or "").strip()
    if conv_type not in {"private", "group", TEMP_CONV_TYPE} or not _valid_platform_id(conv_id):
        return

    display_name = str(name or "")
    alias_values = [str(alias or "").strip() for alias in aliases]
    alias_values = [alias for alias in alias_values if alias]
    if display_name.strip():
        alias_values.append(display_name.strip())
    if not alias_values:
        return

    key = (conv_type, conv_id)
    existing = candidates.get(key)
    if existing is None:
        candidates[key] = _Candidate(
            conv_type=conv_type,
            conv_id=conv_id,
            name=display_name if display_name.strip() else alias_values[0],
            aliases=set(alias_values),
            source_rank=source_rank,
        )
        return
    existing.merge(display_name, alias_values, source_rank)


def _read_db_candidates(config: dict[str, Any], requested_type: str) -> dict[tuple[str, str], _Candidate]:
    from database import DB_PATH

    candidates: dict[tuple[str, str], _Candidate] = {}
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT session_key, conv_type, conv_id, conv_name, temp_source_group_name "
            "FROM chat_sessions"
        ):
            conv_type = str(row["conv_type"] or "").strip()
            conv_id = str(row["conv_id"] or "").strip()
            if requested_type != "any" and conv_type != requested_type:
                continue
            if not _session_allowed(config, conv_type, conv_id):
                continue
            _add_candidate(
                candidates,
                conv_type=conv_type,
                conv_id=conv_id,
                name=row["conv_name"],
                aliases=[row["temp_source_group_name"]],
                source_rank=20,
            )

        if requested_type in {"any", "group"}:
            for row in conn.execute("SELECT group_id, group_name FROM groups WHERE platform='qq'"):
                group_id = str(row["group_id"] or "").strip()
                if not _session_allowed(config, "group", group_id):
                    continue
                _add_candidate(
                    candidates,
                    conv_type="group",
                    conv_id=group_id,
                    name=row["group_name"],
                    source_rank=10,
                )

        if requested_type in {"any", "private"}:
            for row in conn.execute(
                "SELECT platform_id, nickname FROM entities "
                "WHERE platform='qq' AND COALESCE(is_bot, 0)=0"
            ):
                user_id = str(row["platform_id"] or "").strip()
                if not _session_allowed(config, "private", user_id):
                    continue
                _add_candidate(
                    candidates,
                    conv_type="private",
                    conv_id=user_id,
                    name=row["nickname"],
                    source_rank=10,
                )
    except Exception as exc:
        logger.warning("[tools] search_session: 读取本地会话候选失败: %s", exc)
    finally:
        if conn is not None:
            conn.close()
    return candidates


def _merge_live_temp_sessions(
    candidates: dict[tuple[str, str], _Candidate],
    config: dict[str, Any],
    requested_type: str,
) -> None:
    if requested_type not in {"any", "temp"}:
        return
    try:
        from llm.session import sessions

        for session in sessions.values():
            if getattr(session, "conv_type", "") != TEMP_CONV_TYPE:
                continue
            user_id = str(getattr(session, "conv_id", "") or "").strip()
            if not _session_allowed(config, TEMP_CONV_TYPE, user_id):
                continue
            _add_candidate(
                candidates,
                conv_type=TEMP_CONV_TYPE,
                conv_id=user_id,
                name=getattr(session, "conv_name", ""),
                aliases=[getattr(session, "temp_source_group_name", "")],
                source_rank=30,
            )
    except Exception:
        logger.debug("[tools] search_session: live temp session merge failed", exc_info=True)


def _adapter_loop(client: Any) -> Any:
    return getattr(client, "_loop", None) or _main_loop_fallback()


def _adapter_list(client: Any, action: str) -> list[dict] | None:
    if not client or not getattr(client, "connected", False):
        return None
    loop = _adapter_loop(client)
    if loop is None or not loop.is_running():
        return None
    try:
        data = run_coroutine_sync(client.send_api(action, {}), loop, timeout=15)
    except Exception as exc:
        logger.warning("[tools] search_session: adapter %s failed: %s", action, exc)
        return None
    return data if isinstance(data, list) else None


def _collect_candidates(config: dict[str, Any], requested_type: str) -> dict[tuple[str, str], _Candidate]:
    candidates = _read_db_candidates(config, requested_type)
    client = _qq_adapter_client_fallback()

    live_private_ids: set[str] | None = None
    if requested_type in {"any", "private"}:
        friends = _adapter_list(client, "get_friend_list")
        if friends is not None:
            live_private_ids = set()
            for friend in friends:
                if not isinstance(friend, dict):
                    continue
                user_id = str(friend.get("user_id", "") or "").strip()
                if not _session_allowed(config, "private", user_id):
                    continue
                live_private_ids.add(user_id)
                _add_candidate(
                    candidates,
                    conv_type="private",
                    conv_id=user_id,
                    name=friend.get("remark") or friend.get("nickname"),
                    aliases=[friend.get("remark"), friend.get("nickname")],
                    source_rank=40,
                )

    live_group_ids: set[str] | None = None
    if requested_type in {"any", "group"}:
        groups = _adapter_list(client, "get_group_list")
        if groups is not None:
            live_group_ids = set()
            for group in groups:
                if not isinstance(group, dict):
                    continue
                group_id = str(group.get("group_id", "") or "").strip()
                if not _session_allowed(config, "group", group_id):
                    continue
                live_group_ids.add(group_id)
                _add_candidate(
                    candidates,
                    conv_type="group",
                    conv_id=group_id,
                    name=group.get("group_name"),
                    source_rank=40,
                )

    if live_private_ids is not None:
        for key in list(candidates):
            conv_type, conv_id = key
            if conv_type == "private" and conv_id not in live_private_ids:
                candidates.pop(key, None)

    if live_group_ids is not None:
        for key in list(candidates):
            conv_type, conv_id = key
            if conv_type == "group" and conv_id not in live_group_ids:
                candidates.pop(key, None)

    _merge_live_temp_sessions(candidates, config, requested_type)
    return candidates


def _search(query: str, config: dict[str, Any], requested_type: str, limit: int) -> dict[str, Any]:
    query = str(query or "").strip()
    if not query:
        return {"ok": False, "status": "invalid"}

    requested_type = requested_type if requested_type in _VALID_TYPES else "any"
    limit = max(1, min(int(limit or 5), 8))
    candidates = _collect_candidates(config, requested_type)
    query_vector = _sparse_vector(query)

    scored: list[tuple[float, _Candidate]] = []
    for candidate in candidates.values():
        score = _candidate_score(query, query_vector, candidate)
        if score >= 0.34:
            scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], item[1].conv_type, item[1].name, item[1].conv_id))
    if not scored:
        return {"ok": True, "status": "not_found"}

    top_score = scored[0][0]
    if top_score < 0.55:
        return {"ok": True, "status": "not_found"}

    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if top_score >= 0.72 and top_score - second_score >= 0.08:
        return {
            "ok": True,
            "status": "found",
            "target": _candidate_payload(scored[0][1], top_score),
        }

    return {
        "ok": True,
        "status": "ambiguous",
        "candidates": [
            _candidate_payload(candidate, score)
            for score, candidate in scored[:limit]
        ],
    }


def make_handler(config: dict) -> Callable:
    def execute(
        query: str,
        type: str = "any",
        limit: int = 5,
        **kwargs,
    ) -> dict:
        return _search(query, config, str(type or "any"), limit)

    return execute


def sanitize_semantic_args(args: dict[str, Any]) -> tuple[dict[str, Any], list[str], str | None]:
    changes: list[str] = []
    repaired = dict(args)

    query = str(repaired.get("query", "") or "").strip()
    if query != repaired.get("query"):
        repaired["query"] = query
        changes.append("query: trimmed surrounding whitespace")
    if not query:
        return repaired, changes, "query is empty"

    requested_type = str(repaired.get("type", "any") or "any").strip()
    if requested_type not in _VALID_TYPES:
        requested_type = "any"
        changes.append("type: reset to any")
    if requested_type != repaired.get("type"):
        repaired["type"] = requested_type

    try:
        limit = int(repaired.get("limit", 5) or 5)
    except (TypeError, ValueError):
        limit = 5
        changes.append("limit: reset to 5")
    clamped_limit = max(1, min(limit, 8))
    if clamped_limit != repaired.get("limit"):
        repaired["limit"] = clamped_limit
        if clamped_limit != limit:
            changes.append("limit: clamped to 1..8")

    return repaired, changes, None
