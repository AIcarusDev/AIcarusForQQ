from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from token_usage_stats import TokenUsageStatsService
from tool_usage_stats import ToolUsageStatsService


BASE_TIME_MS = 1_700_000_000_000


def _create_tool_db(path) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE bot_turns (
                turn_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                conv_type TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                tool_calls TEXT NOT NULL
            )"""
        )
        rows = [
            (
                "turn-1",
                BASE_TIME_MS,
                "group",
                "1",
                json.dumps({"cognition": "first"}),
                json.dumps([
                    {
                        "namespace": "core",
                        "function": "web_search",
                        "result": {"ok": True},
                        "elapsed_ms": 100,
                    }
                ]),
            ),
            (
                "turn-2",
                BASE_TIME_MS + 1_000,
                "group",
                "1",
                json.dumps({"cognition": "second"}),
                json.dumps([
                    {
                        "namespace": "core",
                        "function": "web_search",
                        "result": {"ok": False, "error": "timeout"},
                        "elapsed_ms": 300,
                    },
                    {
                        "namespace": "core",
                        "function": "calculator",
                        "result": {"ok": True},
                        "elapsed_ms": "invalid",
                    },
                ]),
            ),
        ]
        db.executemany(
            "INSERT INTO bot_turns VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()


def _create_token_db(path) -> None:
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE llm_usage_events (
                event_id TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                feature TEXT NOT NULL,
                subfeature TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cached_input_tokens INTEGER NOT NULL,
                reasoning_output_tokens INTEGER NOT NULL,
                usage_available INTEGER NOT NULL,
                status TEXT NOT NULL,
                raw_usage_json TEXT NOT NULL,
                legacy_turn_id TEXT NOT NULL
            )"""
        )
        rows = [
            ("evt-1", BASE_TIME_MS, "openai", "gpt-main", "main_round", "", 100, 50, 150, 20, 10, 1, "success", "{}", ""),
            ("evt-2", BASE_TIME_MS + 1_000, "openai", "gpt-main", "memory_event_extraction", "", 200, 100, 300, 0, 0, 1, "success", "{}", ""),
            ("evt-3", BASE_TIME_MS + 2_000, "openai", "gpt-small", "main_round", "retry", 40, 10, 50, 0, 0, 1, "success", "{}", ""),
            ("evt-4", BASE_TIME_MS + 3_000, "openai", "gpt-main", "slow_thinking", "", 0, 0, 0, 0, 0, 0, "success", "{}", ""),
        ]
        db.executemany(
            "INSERT INTO llm_usage_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        db.commit()


def test_tool_timeline_aggregates_persisted_latency(tmp_path) -> None:
    db_path = tmp_path / "tools.db"
    _create_tool_db(db_path)
    service = ToolUsageStatsService(str(db_path))

    payload = asyncio.run(
        service.timeline(granularity="day", range_preset="all", tz_offset_minutes=0)
    )

    search = next(item for item in payload["tools"] if item["name"] == "core.web_search")
    assert search["total"] == 2
    assert search["success"] == 1
    assert search["failed"] == 1
    assert search["timed_calls"] == 2
    assert search["avg_elapsed_ms"] == pytest.approx(200.0)
    assert search["p50_elapsed_ms"] == pytest.approx(200.0)
    assert search["p95_elapsed_ms"] == pytest.approx(290.0)
    assert search["max_elapsed_ms"] == pytest.approx(300.0)
    assert search["points"][0]["timed_calls"] == 2
    assert search["points"][0]["avg_elapsed_ms"] == pytest.approx(200.0)

    calculator = next(item for item in payload["tools"] if item["name"] == "core.calculator")
    assert calculator["total"] == 1
    assert calculator["timed_calls"] == 0
    assert calculator["avg_elapsed_ms"] == 0.0


def test_tool_snapshot_and_bucket_detail_share_latency_semantics(tmp_path) -> None:
    db_path = tmp_path / "tools.db"
    _create_tool_db(db_path)
    service = ToolUsageStatsService(str(db_path))

    snapshot = asyncio.run(service.snapshot())
    search = next(item for item in snapshot["tools"] if item["name"] == "core.web_search")
    assert search["timed_calls"] == 2
    assert search["p95_elapsed_ms"] == pytest.approx(290.0)

    timeline = asyncio.run(
        service.timeline(granularity="day", range_preset="all", tz_offset_minutes=0)
    )
    bucket_start = timeline["tools"][0]["points"][0]["bucket_start"]
    detail = asyncio.run(
        service.bucket_detail(
            granularity="day",
            bucket_start=bucket_start,
            tool_name="core.web_search",
            tz_offset_minutes=0,
        )
    )
    assert detail["summary"]["timed_calls"] == 2
    assert detail["summary"]["avg_elapsed_ms"] == pytest.approx(200.0)


def test_token_timeline_can_group_series_by_feature(tmp_path) -> None:
    db_path = tmp_path / "tokens.db"
    _create_token_db(db_path)
    service = TokenUsageStatsService(str(db_path))

    payload = asyncio.run(
        service.timeline(
            granularity="day",
            range_preset="all",
            tz_offset_minutes=0,
            group_by="feature",
        )
    )

    assert payload["group_by"] == "feature"
    main = next(item for item in payload["series"] if item["feature"] == "main_round")
    assert main["requests"] == 2
    assert main["known_requests"] == 2
    assert main["input_tokens"] == 140
    assert main["output_tokens"] == 60
    assert main["total_tokens"] == 200
    assert main["cached_input_tokens"] == 20
    assert main["reasoning_output_tokens"] == 10
    assert main["points"][0]["input_tokens"] == 140

    unknown = next(item for item in payload["series"] if item["feature"] == "slow_thinking")
    assert unknown["requests"] == 1
    assert unknown["known_requests"] == 0
    assert unknown["unknown_requests"] == 1
    assert unknown["total_tokens"] == 0


def test_token_timeline_default_model_group_remains_compatible(tmp_path) -> None:
    db_path = tmp_path / "tokens.db"
    _create_token_db(db_path)
    service = TokenUsageStatsService(str(db_path))

    payload = asyncio.run(
        service.timeline(granularity="day", range_preset="all", tz_offset_minutes=0)
    )

    assert payload["group_by"] == "model"
    assert {(item["provider"], item["model"]) for item in payload["series"]} == {
        ("openai", "gpt-main"),
        ("openai", "gpt-small"),
    }


def test_token_feature_timeline_is_empty_when_usage_table_is_missing(tmp_path) -> None:
    service = TokenUsageStatsService(str(tmp_path / "empty.db"))

    payload = asyncio.run(service.timeline(group_by="feature"))

    assert payload["group_by"] == "feature"
    assert payload["summary"]["total_requests"] == 0
    assert payload["series"] == []
