"""Tool usage statistics built from persisted bot turn logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
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


class ToolUsageStatsService:
    """Aggregate function-tool calls from ``bot_turns.tool_calls``.

    The service intentionally reads the durable round log instead of instrumenting
    each tool handler. That keeps failed calls, unknown tools, and argument errors
    inside the same counting path as successful executions. XML protocol parse
    failures are counted separately because they are not real tool handler calls.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

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
