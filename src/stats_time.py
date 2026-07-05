"""Shared time helpers for usage statistics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

_BASE_RANGE_PRESETS = {"24h", "7d", "30d", "90d", "all"}
_CUSTOM_RANGE_PRESETS = _BASE_RANGE_PRESETS | {"custom"}


def bucket_start_ms(created_at: int, granularity: str, tz_offset_minutes: int) -> int:
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    dt = datetime.fromtimestamp(created_at / 1000, tz)
    if granularity == "hour":
        start = dt.replace(minute=0, second=0, microsecond=0)
    elif granularity == "month":
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.astimezone(timezone.utc).timestamp() * 1000)


def next_bucket_start_ms(bucket_start: int, granularity: str, tz_offset_minutes: int) -> int:
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


def apply_range_preset(
    *,
    start_ms: int | None,
    end_ms: int | None,
    range_preset: str | None,
    latest_created_at: int | None,
    allow_custom: bool = False,
) -> tuple[int | None, int | None, str]:
    allowed = _CUSTOM_RANGE_PRESETS if allow_custom else _BASE_RANGE_PRESETS
    preset = range_preset if range_preset in allowed else "all"
    if start_ms is not None or end_ms is not None or preset in {"all", "custom"}:
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
