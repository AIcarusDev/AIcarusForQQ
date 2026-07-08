"""Recall query construction and multi-source result fusion."""

from __future__ import annotations

import html
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class RecallQueryFacet:
    source: str
    query: str
    weight: float


RecallFn = Callable[..., Awaitable[list[dict[str, Any]]]]

logger = logging.getLogger("AICQ.memory.recall")

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|[\r\n]+")
_TIMESTAMP_RE = re.compile(
    r"^\s*(?:\d{1,4}[-/年])?\d{1,2}[-/:月]\d{1,2}(?:[日\sT]+)?(?:\d{1,2}:\d{2}(?::\d{2})?)?\s*$"
)
_PURE_SYMBOL_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_MIN_QUERY_CHARS = 4
_MAX_QUERY_CHARS = 180
_DEFAULT_WORLD_CHUNKS = 6
_DEFAULT_COGNITION_CHUNKS = 3


def build_recall_query_facets(
    *,
    latest_user_text: str = "",
    chat_world_content: str | list | None = None,
    browser_world_content: str | list | None = None,
    recent_cognitions: list[str] | tuple[str, ...] | None = None,
    max_world_chunks: int = _DEFAULT_WORLD_CHUNKS,
    max_cognition_chunks: int = _DEFAULT_COGNITION_CHUNKS,
    min_query_chars: int = _MIN_QUERY_CHARS,
) -> list[RecallQueryFacet]:
    """Build bounded, weighted recall queries from current input surfaces."""

    facets: list[RecallQueryFacet] = []
    seen: set[str] = set()
    min_chars = max(1, int(min_query_chars or _MIN_QUERY_CHARS))

    def add(source: str, query: str, weight: float) -> None:
        cleaned = _clean_query_text(query)
        if not _is_useful_query(cleaned, min_chars=min_chars):
            return
        key = _dedupe_key(cleaned)
        if key in seen:
            return
        seen.add(key)
        facets.append(RecallQueryFacet(source=source, query=cleaned, weight=float(weight)))

    add("latest_user", latest_user_text, 1.0)

    chat_text = extract_visible_text(chat_world_content)
    for chunk in _select_chunks(chat_text, int(max_world_chunks)):
        add("world.chat", chunk, 0.72)

    browser_text = extract_visible_text(browser_world_content)
    for chunk in _select_chunks(browser_text, max(0, int(max_world_chunks) // 2)):
        add("world.browser", chunk, 0.56)

    cognition_items = list(recent_cognitions or ())[-max(0, int(max_cognition_chunks)) :]
    for cognition in cognition_items:
        for chunk in _select_chunks(cognition, 1):
            add("recent_cognition", chunk, 0.64)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[recall] facets built count=%d latest_len=%d chat_len=%d browser_len=%d cognition_count=%d\n%s",
            len(facets),
            len(_clean_query_text(latest_user_text)),
            len(chat_text),
            len(browser_text),
            len(cognition_items),
            _format_facets_for_log(facets),
        )
    return facets


async def recall_events_from_facets(
    *,
    sender_entity: str,
    context_scope: str,
    limit: int,
    facets: list[RecallQueryFacet],
    recall_fn: RecallFn | None = None,
) -> list[dict[str, Any]]:
    """Run bounded recall for each facet and fuse candidates by weighted average."""

    uses_default_recall = recall_fn is None
    if recall_fn is None:
        from memory.repo.events import load_events_for_recall as recall_fn

    limit = max(1, int(limit or 1))
    if not facets:
        events = await recall_fn(
            sender_entity=sender_entity,
            context_scope=context_scope,
            limit=limit,
            query="",
        )
        logger.debug(
            "[recall] no facets; fallback recent/entity recall scope=%s sender=%s results=%d\n%s",
            context_scope or "<global>",
            sender_entity or "<none>",
            len(events),
            _format_events_for_log(events),
        )
        if uses_default_recall:
            return await _augment_with_ready_summaries(
                events,
                sender_entity=sender_entity,
                context_scope=context_scope,
                limit=limit,
                query="",
            )
        return events

    merged: dict[int, dict[str, Any]] = {}
    weighted_scores: dict[int, float] = {}
    weights: dict[int, float] = {}
    facet_sources: dict[int, set[str]] = {}
    facet_hits: dict[int, list[dict[str, Any]]] = {}

    per_facet_limit = max(limit, min(16, limit * 2))
    for facet in facets:
        events = await recall_fn(
            sender_entity=sender_entity,
            context_scope=context_scope,
            limit=per_facet_limit,
            query=facet.query,
        )
        for event, facet_score in _normalized_facet_scores(events):
            try:
                event_id = int(event.get("event_id", 0))
            except (TypeError, ValueError):
                continue
            if event_id not in merged:
                merged[event_id] = dict(event)
            weighted_scores[event_id] = weighted_scores.get(event_id, 0.0) + facet.weight * facet_score
            weights[event_id] = weights.get(event_id, 0.0) + facet.weight
            facet_sources.setdefault(event_id, set()).add(facet.source)
            facet_hits.setdefault(event_id, []).append({
                "source": facet.source,
                "query": facet.query,
                "weight": round(facet.weight, 6),
                "facet_score": round(facet_score, 6),
            })
        logger.debug(
            "[recall] facet source=%s weight=%.2f query=%r results=%d\n%s",
            facet.source,
            facet.weight,
            _preview(facet.query),
            len(events),
            _format_events_for_log(events),
        )

    ranked: list[tuple[float, dict[str, Any]]] = []
    for event_id, event in merged.items():
        denom = weights.get(event_id, 0.0) or 1.0
        score = weighted_scores.get(event_id, 0.0) / denom
        source_count = len(facet_sources.get(event_id, ()))
        score += min(0.16, max(0, source_count - 1) * 0.08)
        reasons = set(str(x) for x in event.get("recall_reasons", []) or [])
        reasons.update(f"facet:{source}" for source in sorted(facet_sources.get(event_id, ())))
        event["recall_reasons"] = sorted(reasons)
        event["recall_facets"] = _dedupe_facet_hits(facet_hits.get(event_id, []))
        event["recall_score"] = round(score, 6)
        ranked.append((score, event))

    ranked.sort(
        key=lambda item: (
            item[0],
            int(item[1].get("occurred_at") or 0),
            int(item[1].get("event_id") or 0),
        ),
        reverse=True,
    )
    fused_candidates = [event for _, event in ranked]
    top = fused_candidates[:limit]
    logger.debug(
        "[recall] fused scope=%s sender=%s facets=%d candidates=%d top=%d\n%s",
        context_scope or "<global>",
        sender_entity or "<none>",
        len(facets),
        len(merged),
        len(top),
        _format_events_for_log(top, include_reasons=True, include_facets=True),
    )
    if uses_default_recall:
        return await _augment_with_ready_summaries(
            fused_candidates,
            sender_entity=sender_entity,
            context_scope=context_scope,
            limit=limit,
            query="\n".join(facet.query for facet in facets),
        )
    return top


async def _augment_with_ready_summaries(
    events: list[dict[str, Any]],
    *,
    sender_entity: str,
    context_scope: str,
    limit: int,
    query: str,
) -> list[dict[str, Any]]:
    event_ids = _event_int_ids(events)
    replacement_summaries: list[dict[str, Any]] = []
    try:
        from memory.summary_recall import load_ready_summaries_covering_events

        replacement_summaries = await load_ready_summaries_covering_events(
            event_ids=event_ids,
            sender_entity=sender_entity,
            context_scope=context_scope,
            query=query,
            limit=max(max(1, int(limit or 1)) * 8, len(event_ids) * 4, 16),
        )
    except Exception:
        logger.debug("[recall] ready summary augmentation failed", exc_info=True)
        return events
    if not replacement_summaries:
        return events

    summary_items, covered_event_ids = _cluster_summaries_with_inherited_scores(
        replacement_summaries,
        events,
    )
    if not summary_items:
        return events

    combined: list[dict[str, Any]] = list(summary_items)
    for event in events:
        try:
            event_id = int(event.get("event_id", 0))
        except (TypeError, ValueError):
            combined.append(event)
            continue
        if event_id not in covered_event_ids:
            combined.append(event)
    combined.sort(key=_recall_item_sort_key, reverse=True)
    return combined[: max(1, int(limit or 1))]


def _event_int_ids(events: list[dict[str, Any]]) -> list[int]:
    ids: list[int] = []
    for event in events:
        try:
            event_id = int(event.get("event_id", 0))
        except (TypeError, ValueError):
            continue
        if event_id > 0:
            ids.append(event_id)
    return list(dict.fromkeys(ids))


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cluster_summaries_with_inherited_scores(
    summaries: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[int]]:
    events_by_id: dict[int, dict[str, Any]] = {}
    for event in events:
        try:
            event_id = int(event.get("event_id", 0))
        except (TypeError, ValueError):
            continue
        if event_id > 0:
            events_by_id[event_id] = event

    out_by_summary_id: dict[str, dict[str, Any]] = {}
    covered_event_ids: set[int] = set()
    for summary in summaries:
        summary_id = str(summary.get("summary_id") or "").strip()
        if not summary_id:
            continue
        contributing_ids = sorted(_source_event_ids(summary) & set(events_by_id))
        if not contributing_ids:
            continue
        item = out_by_summary_id.get(summary_id)
        if item is None:
            item = dict(summary)
            item["recall_score"] = 0.0
            out_by_summary_id[summary_id] = item
        existing_contributors = set(item.get("_contributing_event_ids") or ())
        new_contributors = [event_id for event_id in contributing_ids if event_id not in existing_contributors]
        if not new_contributors:
            continue
        inherited_score = sum(_float_value(events_by_id[event_id].get("recall_score"), 0.0) for event_id in new_contributors)
        item["_contributing_event_ids"] = sorted(existing_contributors | set(new_contributors))
        item["recall_score"] = round(_float_value(item.get("recall_score"), 0.0) + inherited_score, 6)
        reasons = set(str(x) for x in item.get("recall_reasons", []) or [])
        reasons.update(str(x) for x in summary.get("recall_reasons", []) or [])
        for event_id in new_contributors:
            reasons.update(str(x) for x in events_by_id[event_id].get("recall_reasons", []) or [])
        reasons.add("summary:replaced_event")
        reasons.add("summary:score_inherited_from_atoms")
        if len(item.get("_contributing_event_ids") or ()) > 1:
            reasons.add("summary:score_summed_from_atoms")
        item["recall_reasons"] = sorted(reasons)
        covered_event_ids.update(new_contributors)
    out = []
    for item in out_by_summary_id.values():
        item["contributing_event_ids"] = list(item.pop("_contributing_event_ids", ()))
        out.append(item)
    return out, covered_event_ids


def _source_event_ids(summary: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for value in summary.get("source_event_ids") or ():
        try:
            event_id = int(value)
        except (TypeError, ValueError):
            continue
        if event_id > 0:
            ids.add(event_id)
    return ids


def _recall_item_sort_key(item: dict[str, Any]) -> tuple[float, int, int, str]:
    return (
        _float_value(item.get("recall_score"), 0.0),
        int(item.get("occurred_at") or item.get("created_at") or 0),
        1 if item.get("memory_kind") == "summary" else 0,
        str(item.get("event_id") or ""),
    )


def extract_visible_text(content: str | list | None) -> str:
    """Extract visible text from prompt text or multimodal message parts."""

    if content is None:
        return ""
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content)
    text = html.unescape(text)
    parsed = _extract_xml_text(text)
    return _clean_query_text(parsed)


def _extract_xml_text(text: str) -> str:
    if not text.strip():
        return ""
    try:
        root = ET.fromstring(f"<root>{text}</root>")
        return "\n".join(part.strip() for part in root.itertext() if part and part.strip())
    except ET.ParseError:
        return _TAG_RE.sub("\n", text)


def _select_chunks(text: str, limit: int) -> list[str]:
    if limit <= 0:
        return []
    chunks: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text or ""):
        cleaned = _clean_query_text(part)
        if not _is_useful_query(cleaned):
            continue
        chunks.append(cleaned[:_MAX_QUERY_CHARS])
    chunks.sort(key=lambda value: (len(value), value), reverse=True)
    return chunks[:limit]


def _normalized_facet_scores(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
    if not events:
        return []
    out: list[tuple[dict[str, Any], float]] = []
    count = max(1, len(events))
    for index, event in enumerate(events):
        rank_score = 1.0 - (index / count)
        try:
            score_score = float(event.get("recall_score", 0.0))
        except (TypeError, ValueError):
            score_score = rank_score
        score_score = max(0.0, min(1.0, score_score))
        combined = score_score * 0.85 + rank_score * 0.15
        out.append((event, max(0.0, min(1.0, combined))))
    return out


def _clean_query_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.replace("\u200b", " ")
    return _SPACE_RE.sub(" ", text).strip()


def _is_useful_query(text: str, *, min_chars: int = _MIN_QUERY_CHARS) -> bool:
    if len(text) < max(1, int(min_chars or _MIN_QUERY_CHARS)):
        return False
    if _TIMESTAMP_RE.match(text):
        return False
    if _PURE_SYMBOL_RE.match(text):
        return False
    return True


def _dedupe_key(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _preview(text: str, limit: int = 120) -> str:
    cleaned = _clean_query_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "..."


def _format_facets_for_log(facets: list[RecallQueryFacet]) -> str:
    if not facets:
        return "  <no facets>"
    return "\n".join(
        f"  - #{index} source={facet.source} weight={facet.weight:.2f} query={_preview(facet.query)!r}"
        for index, facet in enumerate(facets, start=1)
    )


def _dedupe_facet_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for hit in hits:
        source = str(hit.get("source") or "")
        query = str(hit.get("query") or "")
        key = (source, _dedupe_key(query))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": source,
            "query": query,
            "weight": hit.get("weight", 0.0),
            "facet_score": hit.get("facet_score", 0.0),
        })
    return out


def _format_events_for_log(
    events: list[dict[str, Any]],
    *,
    include_reasons: bool = False,
    include_facets: bool = False,
) -> str:
    if not events:
        return "  <no results>"
    lines: list[str] = []
    for index, event in enumerate(events, start=1):
        event_id = event.get("event_id", event.get("id", ""))
        score = event.get("recall_score", "")
        summary = _preview(str(event.get("summary") or ""), 140)
        reason_text = ""
        if include_reasons:
            reasons = ",".join(str(x) for x in event.get("recall_reasons", []) or [])
            reason_text = f" reasons={reasons}" if reasons else ""
        lines.append(f"  - #{index} id={event_id} score={score}{reason_text} summary={summary!r}")
        if include_facets:
            for hit in event.get("recall_facets", []) or []:
                lines.append(
                    "      via "
                    f"source={hit.get('source', '')} "
                    f"weight={hit.get('weight', '')} "
                    f"facet_score={hit.get('facet_score', '')} "
                    f"query={_preview(str(hit.get('query') or ''), 120)!r}"
                )
    return "\n".join(lines)
