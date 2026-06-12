"""Tool usage statistics built from persisted bot turn logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import aiosqlite

from database import DB_PATH
from llm.core.tool_calling.xml_protocol import XML_TOOL_CALL_ERROR_NAME


def _utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _is_failed_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    return bool(result.get("error")) or result.get("ok") is False


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _normalize_granularity(value: str | None) -> str:
    if value in {"hour", "day", "month"}:
        return value
    return "day"


def _bucket_start_ms(created_at: int, granularity: str, tz_offset_minutes: int) -> int:
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    dt = datetime.fromtimestamp(created_at / 1000, tz)
    if granularity == "hour":
        start = dt.replace(minute=0, second=0, microsecond=0)
    elif granularity == "month":
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.astimezone(timezone.utc).timestamp() * 1000)


def _next_bucket_start_ms(bucket_start: int, granularity: str, tz_offset_minutes: int) -> int:
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    dt = datetime.fromtimestamp(bucket_start / 1000, tz)
    if granularity == "hour":
        nxt = dt + timedelta(hours=1)
    elif granularity == "month":
        year = dt.year + (1 if dt.month == 12 else 0)
        month = 1 if dt.month == 12 else dt.month + 1
        nxt = dt.replace(year=year, month=month)
    else:
        nxt = dt + timedelta(days=1)
    return int(nxt.astimezone(timezone.utc).timestamp() * 1000)


def _apply_range_preset(
    *,
    start_ms: int | None,
    end_ms: int | None,
    range_preset: str | None,
    latest_created_at: int | None,
) -> tuple[int | None, int | None, str]:
    preset = range_preset if range_preset in {"24h", "7d", "30d", "90d", "all"} else "all"
    if start_ms is not None or end_ms is not None or preset == "all":
        return start_ms, end_ms, preset
    if latest_created_at is None:
        return start_ms, end_ms, preset
    span_ms = {
        "24h": 24 * 60 * 60 * 1000,
        "7d": 7 * 24 * 60 * 60 * 1000,
        "30d": 30 * 24 * 60 * 60 * 1000,
        "90d": 90 * 24 * 60 * 60 * 1000,
    }[preset]
    return latest_created_at - span_ms, latest_created_at, preset


@dataclass(slots=True)
class ToolUsageBucket:
    name: str
    total: int = 0
    success: int = 0
    failed: int = 0
    first_seen_at: int | None = None
    last_seen_at: int | None = None

    def add(self, *, created_at: int, failed: bool) -> None:
        self.total += 1
        if failed:
            self.failed += 1
        else:
            self.success += 1
        if self.first_seen_at is None or created_at < self.first_seen_at:
            self.first_seen_at = created_at
        if self.last_seen_at is None or created_at > self.last_seen_at:
            self.last_seen_at = created_at


@dataclass(slots=True)
class ToolUsageEvent:
    turn_id: str
    created_at: int
    conv_type: str
    conv_id: str
    name: str
    failed: bool
    turn_tools: tuple[str, ...]
    cognition: str


class ToolUsageStatsService:
    """Aggregate function-tool calls from ``bot_turns.tool_calls``.

    The service intentionally reads the durable round log instead of instrumenting
    each tool handler. That keeps failed calls, unknown tools, and argument errors
    inside the same counting path as successful executions. XML protocol parse
    failures are counted separately because they are not real tool handler calls.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    async def timeline(
        self,
        *,
        granularity: str = "day",
        range_preset: str = "all",
        tool_names: list[str] | None = None,
        limit: int = 6,
        start_ms: int | None = None,
        end_ms: int | None = None,
        tz_offset_minutes: int = 480,
    ) -> dict:
        """Return time-series call counts for selected tools."""
        granularity = _normalize_granularity(granularity)
        limit = max(1, min(12, int(limit or 6)))
        tz_offset_minutes = _clamp_int(
            tz_offset_minutes,
            default=480,
            minimum=-14 * 60,
            maximum=14 * 60,
        )
        requested_names = [
            str(name).strip()
            for name in (tool_names or [])
            if str(name).strip()
        ]
        requested_set = set(requested_names)

        events, meta = await self._load_events()
        latest_created_at = max((event.created_at for event in events), default=None)
        start_ms, end_ms, range_preset = _apply_range_preset(
            start_ms=start_ms,
            end_ms=end_ms,
            range_preset=range_preset,
            latest_created_at=latest_created_at,
        )
        filtered = [
            event for event in events
            if (start_ms is None or event.created_at >= start_ms)
            and (end_ms is None or event.created_at <= end_ms)
        ]

        totals: dict[str, ToolUsageBucket] = {}
        for event in filtered:
            bucket = totals.get(event.name)
            if bucket is None:
                bucket = ToolUsageBucket(name=event.name)
                totals[event.name] = bucket
            bucket.add(created_at=event.created_at, failed=event.failed)

        ranked_names = [
            bucket.name
            for bucket in sorted(totals.values(), key=lambda item: (-item.total, item.name))
        ]
        selected_names = [
            name for name in requested_names
            if name in totals or requested_set
        ]
        if not selected_names:
            selected_names = ranked_names[:limit]
        else:
            selected_names = selected_names[:limit]
        selected_set = set(selected_names)

        bucket_starts = self._timeline_bucket_starts(
            filtered,
            granularity=granularity,
            tz_offset_minutes=tz_offset_minutes,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        bucket_totals: dict[int, int] = {bucket_start: 0 for bucket_start in bucket_starts}
        series_counts: dict[str, dict[int, ToolUsageBucket]] = {
            name: {
                bucket_start: ToolUsageBucket(name=name)
                for bucket_start in bucket_starts
            }
            for name in selected_names
        }
        bucket_turns: dict[tuple[str, int], set[str]] = {
            (name, bucket_start): set()
            for name in selected_names
            for bucket_start in bucket_starts
        }
        co_tools: dict[str, dict[str, int]] = {}

        for event in filtered:
            bucket_start = _bucket_start_ms(event.created_at, granularity, tz_offset_minutes)
            if bucket_start in bucket_totals:
                bucket_totals[bucket_start] += 1
            if event.name not in selected_set:
                continue
            if bucket_start not in series_counts[event.name]:
                continue
            series_counts[event.name][bucket_start].add(
                created_at=event.created_at,
                failed=event.failed,
            )
            bucket_turns[(event.name, bucket_start)].add(event.turn_id)
            for co_name in event.turn_tools:
                if co_name == event.name or co_name == XML_TOOL_CALL_ERROR_NAME:
                    continue
                tool_co = co_tools.setdefault(co_name, {"calls": 0, "turns": 0})
                tool_co["calls"] += 1

        selected_turns_by_co: dict[str, set[str]] = {}
        for event in filtered:
            if event.name not in selected_set:
                continue
            for co_name in event.turn_tools:
                if co_name == event.name or co_name == XML_TOOL_CALL_ERROR_NAME:
                    continue
                selected_turns_by_co.setdefault(co_name, set()).add(event.turn_id)
        for co_name, turns in selected_turns_by_co.items():
            co_tools.setdefault(co_name, {"calls": 0, "turns": 0})["turns"] = len(turns)

        total_selected_calls = sum(totals[name].total for name in selected_names if name in totals)
        peak: dict[str, Any] | None = None
        tools = []
        for name in selected_names:
            aggregate = totals.get(name, ToolUsageBucket(name=name))
            points = []
            for bucket_start in bucket_starts:
                point_bucket = series_counts[name][bucket_start]
                bucket_end = _next_bucket_start_ms(bucket_start, granularity, tz_offset_minutes)
                point = {
                    "bucket_start": bucket_start,
                    "bucket_end": bucket_end,
                    "total": point_bucket.total,
                    "success": point_bucket.success,
                    "failed": point_bucket.failed,
                    "turn_count": len(bucket_turns[(name, bucket_start)]),
                }
                points.append(point)
                if point_bucket.total and (
                    peak is None
                    or point_bucket.total > peak["total"]
                    or (
                        point_bucket.total == peak["total"]
                        and bucket_start < peak["bucket_start"]
                    )
                ):
                    peak = {
                        "tool": name,
                        "bucket_start": bucket_start,
                        "bucket_end": bucket_end,
                        "total": point_bucket.total,
                        "success": point_bucket.success,
                        "failed": point_bucket.failed,
                    }
            tools.append({
                "name": name,
                "total": aggregate.total,
                "success": aggregate.success,
                "failed": aggregate.failed,
                "share": (
                    aggregate.total / total_selected_calls
                    if total_selected_calls else 0.0
                ),
                "points": points,
            })

        return {
            "generated_at": _utc_ms(),
            "granularity": granularity,
            "range": range_preset,
            "start_at": bucket_starts[0] if bucket_starts else None,
            "end_at": (
                _next_bucket_start_ms(bucket_starts[-1], granularity, tz_offset_minutes)
                if bucket_starts else None
            ),
            "timezone_offset_minutes": tz_offset_minutes,
            "summary": {
                "bucket_count": len(bucket_starts),
                "total_calls": total_selected_calls,
                "selected_tool_count": len(selected_names),
                "available_tool_count": len(ranked_names),
                "protocol_error_calls": meta["protocol_error_calls"],
                "peak": peak,
            },
            "buckets": [
                {
                    "bucket_start": bucket_start,
                    "bucket_end": _next_bucket_start_ms(bucket_start, granularity, tz_offset_minutes),
                    "total": bucket_totals.get(bucket_start, 0),
                }
                for bucket_start in bucket_starts
            ],
            "tools": tools,
            "available_tools": [
                {
                    "name": name,
                    "total": totals[name].total,
                    "success": totals[name].success,
                    "failed": totals[name].failed,
                }
                for name in ranked_names[:40]
            ],
            "co_tools": [
                {"name": name, "calls": stats["calls"], "turns": stats["turns"]}
                for name, stats in sorted(
                    co_tools.items(),
                    key=lambda item: (-item[1]["calls"], item[0]),
                )[:12]
            ],
        }

    async def bucket_detail(
        self,
        *,
        granularity: str,
        bucket_start: int,
        tool_name: str | None = None,
        tz_offset_minutes: int = 480,
        limit: int = 30,
    ) -> dict:
        """Return turn-level details for a single time bucket."""
        granularity = _normalize_granularity(granularity)
        tz_offset_minutes = _clamp_int(
            tz_offset_minutes,
            default=480,
            minimum=-14 * 60,
            maximum=14 * 60,
        )
        limit = max(1, min(80, int(limit or 30)))
        bucket_start = int(bucket_start)
        bucket_end = _next_bucket_start_ms(bucket_start, granularity, tz_offset_minutes)
        selected_tool = (tool_name or "").strip()

        events, meta = await self._load_events()
        matching = [
            event for event in events
            if bucket_start <= event.created_at < bucket_end
            and (not selected_tool or event.name == selected_tool)
        ]
        total_calls = len(matching)
        success_calls = sum(1 for event in matching if not event.failed)
        failed_calls = total_calls - success_calls

        selected_turn_ids = {event.turn_id for event in matching}
        co_tool_calls: dict[str, int] = {}
        co_tool_turns: dict[str, set[str]] = {}
        for event in events:
            if event.turn_id not in selected_turn_ids:
                continue
            if selected_tool and event.name == selected_tool:
                continue
            if event.name == XML_TOOL_CALL_ERROR_NAME:
                continue
            co_tool_calls[event.name] = co_tool_calls.get(event.name, 0) + 1
            co_tool_turns.setdefault(event.name, set()).add(event.turn_id)

        turns: dict[str, dict[str, Any]] = {}
        for event in events:
            if event.turn_id not in selected_turn_ids:
                continue
            turn = turns.setdefault(
                event.turn_id,
                {
                    "turn_id": event.turn_id,
                    "created_at": event.created_at,
                    "conv_type": event.conv_type,
                    "conv_id": event.conv_id,
                    "tools": [],
                    "selected_calls": 0,
                    "failed_calls": 0,
                    "cognition": event.cognition[:240],
                },
            )
            turn["tools"].append(event.name)
            if not selected_tool or event.name == selected_tool:
                turn["selected_calls"] += 1
                if event.failed:
                    turn["failed_calls"] += 1

        ordered_turns = sorted(
            turns.values(),
            key=lambda item: (item["created_at"], item["turn_id"]),
        )[:limit]
        for turn in ordered_turns:
            turn["tools"] = self._compact_tool_names(turn["tools"])

        return {
            "generated_at": _utc_ms(),
            "granularity": granularity,
            "timezone_offset_minutes": tz_offset_minutes,
            "bucket": {
                "bucket_start": bucket_start,
                "bucket_end": bucket_end,
            },
            "tool": selected_tool or None,
            "summary": {
                "total_calls": total_calls,
                "success_calls": success_calls,
                "failed_calls": failed_calls,
                "turn_count": len(selected_turn_ids),
                "returned_turn_count": len(ordered_turns),
                "protocol_error_calls": meta["protocol_error_calls"],
            },
            "co_tools": [
                {
                    "name": name,
                    "calls": calls,
                    "turns": len(co_tool_turns.get(name, set())),
                }
                for name, calls in sorted(
                    co_tool_calls.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:12]
            ],
            "turns": ordered_turns,
        }

    async def snapshot(self) -> dict:
        buckets: dict[str, ToolUsageBucket] = {}
        total_turns = 0
        malformed_rows = 0
        malformed_calls = 0
        protocol_error_calls = 0
        protocol_error_first_seen_at: int | None = None
        protocol_error_last_seen_at: int | None = None

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT created_at, tool_calls FROM bot_turns ORDER BY created_at ASC"
            ) as cur:
                async for row in cur:
                    total_turns += 1
                    created_at = int(row[0] or 0)
                    try:
                        tool_calls = json.loads(row[1] or "[]")
                    except Exception:
                        malformed_rows += 1
                        continue
                    if not isinstance(tool_calls, list):
                        malformed_rows += 1
                        continue
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            malformed_calls += 1
                            name = "<malformed>"
                            result = None
                        else:
                            name = str(call.get("function") or "<unknown>")
                            result = call.get("result")
                            if name == XML_TOOL_CALL_ERROR_NAME:
                                protocol_error_calls += 1
                                if (
                                    protocol_error_first_seen_at is None
                                    or created_at < protocol_error_first_seen_at
                                ):
                                    protocol_error_first_seen_at = created_at
                                if (
                                    protocol_error_last_seen_at is None
                                    or created_at > protocol_error_last_seen_at
                                ):
                                    protocol_error_last_seen_at = created_at
                                continue
                        bucket = buckets.get(name)
                        if bucket is None:
                            bucket = ToolUsageBucket(name=name)
                            buckets[name] = bucket
                        bucket.add(
                            created_at=created_at,
                            failed=_is_failed_result(result),
                        )

        total_calls = sum(bucket.total for bucket in buckets.values())
        successful_calls = sum(bucket.success for bucket in buckets.values())
        failed_calls = sum(bucket.failed for bucket in buckets.values())
        ranked = sorted(
            buckets.values(),
            key=lambda bucket: (-bucket.total, bucket.name),
        )

        tools = []
        for rank, bucket in enumerate(ranked, start=1):
            share = (bucket.total / total_calls) if total_calls else 0.0
            tools.append({
                "rank": rank,
                "name": bucket.name,
                "total": bucket.total,
                "success": bucket.success,
                "failed": bucket.failed,
                "share": share,
                "first_seen_at": bucket.first_seen_at,
                "last_seen_at": bucket.last_seen_at,
            })

        return {
            "generated_at": _utc_ms(),
            "summary": {
                "total_turns": total_turns,
                "total_calls": total_calls,
                "tool_count": len(buckets),
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "malformed_rows": malformed_rows,
                "malformed_calls": malformed_calls,
                "protocol_error_calls": protocol_error_calls,
                "protocol_error_first_seen_at": protocol_error_first_seen_at,
                "protocol_error_last_seen_at": protocol_error_last_seen_at,
            },
            "tools": tools,
        }

    async def _load_events(self) -> tuple[list[ToolUsageEvent], dict[str, int]]:
        events: list[ToolUsageEvent] = []
        malformed_rows = 0
        malformed_calls = 0
        protocol_error_calls = 0

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT turn_id, created_at, conv_type, conv_id, result_json, tool_calls
                   FROM bot_turns
                   ORDER BY created_at ASC"""
            ) as cur:
                async for row in cur:
                    turn_id = str(row[0] or "")
                    created_at = int(row[1] or 0)
                    conv_type = str(row[2] or "")
                    conv_id = str(row[3] or "")
                    cognition = ""
                    try:
                        result_json = json.loads(row[4] or "{}")
                        if isinstance(result_json, dict):
                            cognition = str(result_json.get("cognition") or "")
                    except Exception:
                        cognition = ""
                    try:
                        tool_calls = json.loads(row[5] or "[]")
                    except Exception:
                        malformed_rows += 1
                        continue
                    if not isinstance(tool_calls, list):
                        malformed_rows += 1
                        continue
                    turn_tool_names = self._turn_tool_names(tool_calls)
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            malformed_calls += 1
                            events.append(ToolUsageEvent(
                                turn_id=turn_id,
                                created_at=created_at,
                                conv_type=conv_type,
                                conv_id=conv_id,
                                name="<malformed>",
                                failed=False,
                                turn_tools=turn_tool_names,
                                cognition=cognition,
                            ))
                            continue
                        name = str(call.get("function") or "<unknown>")
                        if name == XML_TOOL_CALL_ERROR_NAME:
                            protocol_error_calls += 1
                            continue
                        events.append(ToolUsageEvent(
                            turn_id=turn_id,
                            created_at=created_at,
                            conv_type=conv_type,
                            conv_id=conv_id,
                            name=name,
                            failed=_is_failed_result(call.get("result")),
                            turn_tools=turn_tool_names,
                            cognition=cognition,
                        ))

        return events, {
            "malformed_rows": malformed_rows,
            "malformed_calls": malformed_calls,
            "protocol_error_calls": protocol_error_calls,
        }

    def _timeline_bucket_starts(
        self,
        events: list[ToolUsageEvent],
        *,
        granularity: str,
        tz_offset_minutes: int,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[int]:
        if events:
            first_event_at = min(event.created_at for event in events)
            last_event_at = max(event.created_at for event in events)
        else:
            first_event_at = start_ms
            last_event_at = end_ms
        if first_event_at is None or last_event_at is None:
            return []
        first_bucket = _bucket_start_ms(
            start_ms if start_ms is not None else first_event_at,
            granularity,
            tz_offset_minutes,
        )
        last_bucket = _bucket_start_ms(
            end_ms if end_ms is not None else last_event_at,
            granularity,
            tz_offset_minutes,
        )
        buckets = []
        bucket_start = first_bucket
        max_buckets = {"hour": 1500, "day": 730, "month": 120}[granularity]
        while bucket_start <= last_bucket and len(buckets) < max_buckets:
            buckets.append(bucket_start)
            bucket_start = _next_bucket_start_ms(
                bucket_start,
                granularity,
                tz_offset_minutes,
            )
        return buckets

    def _turn_tool_names(self, tool_calls: list[Any]) -> tuple[str, ...]:
        names = []
        for call in tool_calls:
            if not isinstance(call, dict):
                names.append("<malformed>")
                continue
            name = str(call.get("function") or "<unknown>")
            if name != XML_TOOL_CALL_ERROR_NAME:
                names.append(name)
        return tuple(names)

    def _compact_tool_names(self, names: list[str]) -> list[str]:
        counts: dict[str, int] = {}
        compact = []
        for name in names:
            counts[name] = counts.get(name, 0) + 1
            if counts[name] == 1:
                compact.append(name)
        return [
            f"{name}x{counts[name]}" if counts[name] > 1 else name
            for name in compact
        ]
