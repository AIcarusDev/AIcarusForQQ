"""Token usage statistics built from persisted LLM usage events."""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite

from database import DB_PATH


def _utc_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _int(value) -> int:
    return int(value or 0)


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
