"""Token usage statistics built from persisted LLM usage events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import aiosqlite

from database import DB_PATH


def _utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _int(value) -> int:
    return int(value or 0)


def _clamp_int(value, *, default: int, minimum: int, maximum: int) -> int:
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


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


class TokenUsageStatsService:
    """Aggregate LLM usage by model and by feature."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    async def snapshot(self) -> dict:
        async with aiosqlite.connect(self.db_path) as db:
            table_exists = await self._table_exists(db)
            if not table_exists:
                return self._empty_snapshot()

            summary = await self._summary(db)
            by_model = await self._grouped(
                db,
                key_cols=("provider", "model"),
                label_cols=("provider", "model"),
            )
            by_feature = await self._grouped(
                db,
                key_cols=("feature",),
                label_cols=("feature",),
            )

        total_tokens = summary["total_tokens"]
        for rows in (by_model, by_feature):
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
                row["share"] = (row["total_tokens"] / total_tokens) if total_tokens else 0.0

        summary["model_count"] = len(by_model)
        summary["feature_count"] = len(by_feature)
        return {
            "generated_at": _utc_ms(),
            "summary": summary,
            "by_model": by_model,
            "by_feature": by_feature,
        }

    async def timeline(
        self,
        *,
        granularity: str = "day",
        range_preset: str = "all",
        start_ms: int | None = None,
        end_ms: int | None = None,
        tz_offset_minutes: int = 480,
    ) -> dict:
        """Return time-bucketed token totals."""
        granularity = _normalize_granularity(granularity)
        tz_offset_minutes = _clamp_int(
            tz_offset_minutes,
            default=480,
            minimum=-14 * 60,
            maximum=14 * 60,
        )

        async with aiosqlite.connect(self.db_path) as db:
            table_exists = await self._table_exists(db)
            if not table_exists:
                return self._empty_timeline(
                    granularity=granularity,
                    range_preset=range_preset,
                    tz_offset_minutes=tz_offset_minutes,
                )

            latest_created_at = await self._latest_created_at(db)
            start_ms, end_ms, range_preset = _apply_range_preset(
                start_ms=start_ms,
                end_ms=end_ms,
                range_preset=range_preset,
                latest_created_at=latest_created_at,
            )
            events = await self._timeline_events(db, start_ms=start_ms, end_ms=end_ms)

        bucket_starts = self._timeline_bucket_starts(
            events,
            granularity=granularity,
            tz_offset_minutes=tz_offset_minutes,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        buckets = {
            bucket_start: {
                "bucket_start": bucket_start,
                "bucket_end": _next_bucket_start_ms(
                    bucket_start,
                    granularity,
                    tz_offset_minutes,
                ),
                "requests": 0,
                "known_requests": 0,
                "unknown_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_output_tokens": 0,
            }
            for bucket_start in bucket_starts
        }

        for event in events:
            bucket_start = _bucket_start_ms(event["created_at"], granularity, tz_offset_minutes)
            bucket = buckets.get(bucket_start)
            if bucket is None:
                continue
            bucket["requests"] += 1
            if event["usage_available"]:
                bucket["known_requests"] += 1
                bucket["input_tokens"] += event["input_tokens"]
                bucket["output_tokens"] += event["output_tokens"]
                bucket["total_tokens"] += event["total_tokens"]
                bucket["cached_input_tokens"] += event["cached_input_tokens"]
                bucket["reasoning_output_tokens"] += event["reasoning_output_tokens"]
            else:
                bucket["unknown_requests"] += 1

        bucket_list = list(buckets.values())
        active_buckets = [bucket for bucket in bucket_list if bucket["requests"] > 0]
        active_totals = [bucket["total_tokens"] for bucket in active_buckets]
        total_requests = sum(bucket["requests"] for bucket in bucket_list)
        known_requests = sum(bucket["known_requests"] for bucket in bucket_list)
        unknown_requests = sum(bucket["unknown_requests"] for bucket in bucket_list)
        total_tokens = sum(bucket["total_tokens"] for bucket in bucket_list)
        peak = self._timeline_peak(active_buckets)

        return {
            "generated_at": _utc_ms(),
            "granularity": granularity,
            "range": range_preset,
            "start_at": bucket_list[0]["bucket_start"] if bucket_list else None,
            "end_at": bucket_list[-1]["bucket_end"] if bucket_list else None,
            "timezone_offset_minutes": tz_offset_minutes,
            "summary": {
                "bucket_count": len(bucket_list),
                "active_bucket_count": len(active_buckets),
                "total_requests": total_requests,
                "known_requests": known_requests,
                "unknown_requests": unknown_requests,
                "total_tokens": total_tokens,
                "avg_bucket_total_tokens": (
                    total_tokens / len(active_buckets)
                    if active_buckets else 0.0
                ),
                "min_bucket_total_tokens": min(active_totals) if active_totals else 0,
                "max_bucket_total_tokens": max(active_totals) if active_totals else 0,
                "median_bucket_total_tokens": _median(active_totals),
                "peak": peak,
            },
            "buckets": bucket_list,
        }

    async def _table_exists(self, db) -> bool:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_usage_events'"
        ) as cur:
            return await cur.fetchone() is not None

    def _empty_snapshot(self) -> dict:
        return {
            "generated_at": _utc_ms(),
            "summary": {
                "total_requests": 0,
                "known_requests": 0,
                "unknown_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_output_tokens": 0,
                "model_count": 0,
                "feature_count": 0,
            },
            "by_model": [],
            "by_feature": [],
        }

    def _empty_timeline(
        self,
        *,
        granularity: str,
        range_preset: str,
        tz_offset_minutes: int,
    ) -> dict:
        return {
            "generated_at": _utc_ms(),
            "granularity": granularity,
            "range": range_preset if range_preset in {"24h", "7d", "30d", "90d", "all"} else "all",
            "start_at": None,
            "end_at": None,
            "timezone_offset_minutes": tz_offset_minutes,
            "summary": {
                "bucket_count": 0,
                "active_bucket_count": 0,
                "total_requests": 0,
                "known_requests": 0,
                "unknown_requests": 0,
                "total_tokens": 0,
                "avg_bucket_total_tokens": 0.0,
                "min_bucket_total_tokens": 0,
                "max_bucket_total_tokens": 0,
                "median_bucket_total_tokens": 0.0,
                "peak": None,
            },
            "buckets": [],
        }

    async def _summary(self, db) -> dict:
        async with db.execute(
            """SELECT
                   COUNT(*) AS total_requests,
                   SUM(CASE WHEN usage_available=1 THEN 1 ELSE 0 END) AS known_requests,
                   SUM(CASE WHEN usage_available=0 THEN 1 ELSE 0 END) AS unknown_requests,
                   SUM(CASE WHEN usage_available=1 THEN input_tokens ELSE 0 END) AS input_tokens,
                   SUM(CASE WHEN usage_available=1 THEN output_tokens ELSE 0 END) AS output_tokens,
                   SUM(CASE WHEN usage_available=1 THEN total_tokens ELSE 0 END) AS total_tokens,
                   SUM(CASE WHEN usage_available=1 THEN cached_input_tokens ELSE 0 END) AS cached_input_tokens,
                   SUM(CASE WHEN usage_available=1 THEN reasoning_output_tokens ELSE 0 END) AS reasoning_output_tokens
               FROM llm_usage_events"""
        ) as cur:
            row = await cur.fetchone()
        return {
            "total_requests": _int(row[0]),
            "known_requests": _int(row[1]),
            "unknown_requests": _int(row[2]),
            "input_tokens": _int(row[3]),
            "output_tokens": _int(row[4]),
            "total_tokens": _int(row[5]),
            "cached_input_tokens": _int(row[6]),
            "reasoning_output_tokens": _int(row[7]),
            "model_count": 0,
            "feature_count": 0,
        }

    async def _grouped(
        self,
        db,
        *,
        key_cols: tuple[str, ...],
        label_cols: tuple[str, ...],
    ) -> list[dict]:
        select_cols = ", ".join(key_cols)
        group_cols = ", ".join(key_cols)
        async with db.execute(
            f"""SELECT
                    {select_cols},
                    COUNT(*) AS requests,
                    SUM(CASE WHEN usage_available=1 THEN 1 ELSE 0 END) AS known_requests,
                    SUM(CASE WHEN usage_available=0 THEN 1 ELSE 0 END) AS unknown_requests,
                    SUM(CASE WHEN usage_available=1 THEN input_tokens ELSE 0 END) AS input_tokens,
                    SUM(CASE WHEN usage_available=1 THEN output_tokens ELSE 0 END) AS output_tokens,
                    SUM(CASE WHEN usage_available=1 THEN total_tokens ELSE 0 END) AS total_tokens,
                    SUM(CASE WHEN usage_available=1 THEN cached_input_tokens ELSE 0 END) AS cached_input_tokens,
                    SUM(CASE WHEN usage_available=1 THEN reasoning_output_tokens ELSE 0 END) AS reasoning_output_tokens
                FROM llm_usage_events
                GROUP BY {group_cols}
                ORDER BY total_tokens DESC, requests DESC, {group_cols} ASC"""
        ) as cur:
            rows = await cur.fetchall()

        result = []
        for row in rows:
            labels = {
                col: str(row[idx] or "")
                for idx, col in enumerate(label_cols)
            }
            offset = len(key_cols)
            result.append({
                **labels,
                "requests": _int(row[offset]),
                "known_requests": _int(row[offset + 1]),
                "unknown_requests": _int(row[offset + 2]),
                "input_tokens": _int(row[offset + 3]),
                "output_tokens": _int(row[offset + 4]),
                "total_tokens": _int(row[offset + 5]),
                "cached_input_tokens": _int(row[offset + 6]),
                "reasoning_output_tokens": _int(row[offset + 7]),
            })
        return result

    async def _latest_created_at(self, db) -> int | None:
        async with db.execute("SELECT MAX(created_at) FROM llm_usage_events") as cur:
            row = await cur.fetchone()
        if not row or row[0] is None:
            return None
        return int(row[0])

    async def _timeline_events(
        self,
        db,
        *,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[dict]:
        where = []
        params = []
        if start_ms is not None:
            where.append("created_at >= ?")
            params.append(int(start_ms))
        if end_ms is not None:
            where.append("created_at <= ?")
            params.append(int(end_ms))
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        async with db.execute(
            f"""SELECT
                    created_at,
                    usage_available,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_input_tokens,
                    reasoning_output_tokens
                FROM llm_usage_events
                {where_sql}
                ORDER BY created_at ASC""",
            params,
        ) as cur:
            rows = await cur.fetchall()
        return [
            {
                "created_at": _int(row[0]),
                "usage_available": bool(row[1]),
                "input_tokens": _int(row[2]),
                "output_tokens": _int(row[3]),
                "total_tokens": _int(row[4]),
                "cached_input_tokens": _int(row[5]),
                "reasoning_output_tokens": _int(row[6]),
            }
            for row in rows
        ]

    def _timeline_bucket_starts(
        self,
        events: list[dict],
        *,
        granularity: str,
        tz_offset_minutes: int,
        start_ms: int | None,
        end_ms: int | None,
    ) -> list[int]:
        if events:
            first_event_at = min(event["created_at"] for event in events)
            last_event_at = max(event["created_at"] for event in events)
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

    def _timeline_peak(self, active_buckets: list[dict]) -> dict | None:
        peak = None
        for bucket in active_buckets:
            if peak is None or bucket["total_tokens"] > peak["total_tokens"] or (
                bucket["total_tokens"] == peak["total_tokens"]
                and bucket["bucket_start"] < peak["bucket_start"]
            ):
                peak = bucket
        if peak is None:
            return None
        return {
            "bucket_start": peak["bucket_start"],
            "bucket_end": peak["bucket_end"],
            "requests": peak["requests"],
            "known_requests": peak["known_requests"],
            "unknown_requests": peak["unknown_requests"],
            "input_tokens": peak["input_tokens"],
            "output_tokens": peak["output_tokens"],
            "total_tokens": peak["total_tokens"],
            "cached_input_tokens": peak["cached_input_tokens"],
            "reasoning_output_tokens": peak["reasoning_output_tokens"],
        }
