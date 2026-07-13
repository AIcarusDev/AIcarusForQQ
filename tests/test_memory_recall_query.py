import asyncio
import logging

from memory.recall.recall_query import (
    build_recall_query_facets,
    extract_visible_text,
    recall_events_from_facets,
)


def test_extract_visible_text_filters_xml_tags_and_keeps_text():
    xml = '<world><qq><msg sender="Alice">今天讨论 memory recall。</msg></qq><browser><title>Memory Graph</title></browser></world>'

    text = extract_visible_text(xml)

    assert "今天讨论 memory recall" in text
    assert "Memory Graph" in text
    assert "<msg" not in text
    assert "sender=" not in text


def test_build_recall_query_facets_uses_world_and_recent_cognition():
    facets = build_recall_query_facets(
        latest_user_text="world 里的内容吗",
        chat_world_content="<world><qq>引用消息里提到了 jasmine tea。</qq></world>",
        browser_world_content="<browser><page>OpenAI compatible embeddings 文档</page></browser>",
        recent_cognitions=["我刚才判断用户在追问记忆召回范围。"],
    )

    sources = {facet.source for facet in facets}
    queries = [facet.query for facet in facets]
    assert "latest_user" in sources
    assert "world.chat" in sources
    assert "world.browser" in sources
    assert "recent_cognition" in sources
    assert any("jasmine tea" in query for query in queries)


def test_build_recall_query_facets_keeps_short_keyword_opt_in_only():
    assert build_recall_query_facets(latest_user_text="以撒") == []

    facets = build_recall_query_facets(latest_user_text="以撒", min_query_chars=2)

    assert len(facets) == 1
    assert facets[0].query == "以撒"


def test_recall_events_from_facets_fuses_by_weighted_average():
    async def fake_recall(**kwargs):
        query = kwargs["query"]
        if "primary" in query:
            return [
                {"event_id": 1, "summary": "A", "recall_score": 0.9, "occurred_at": 1},
                {"event_id": 2, "summary": "B", "recall_score": 0.8, "occurred_at": 2},
            ]
        return [
            {"event_id": 2, "summary": "B", "recall_score": 0.95, "occurred_at": 2},
            {"event_id": 3, "summary": "C", "recall_score": 0.7, "occurred_at": 3},
        ]

    facets = build_recall_query_facets(
        latest_user_text="primary recall signal",
        chat_world_content="<world>secondary recall signal</world>",
    )

    recalled = asyncio.run(
        recall_events_from_facets(
            sender_entity="self",
            context_scope="group:qq_1",
            limit=3,
            facets=facets,
            recall_fn=fake_recall,
        )
    )

    assert [event["event_id"] for event in recalled][:2] == [2, 1]
    assert "facet:latest_user" in recalled[0]["recall_reasons"]
    assert "facet:world.chat" in recalled[0]["recall_reasons"]
    assert {hit["source"] for hit in recalled[0]["recall_facets"]} == {"latest_user", "world.chat"}
    assert any("primary recall signal" in hit["query"] for hit in recalled[0]["recall_facets"])
    assert any("secondary recall signal" in hit["query"] for hit in recalled[0]["recall_facets"])


def test_recall_events_from_facets_logs_facets_and_fused_results(caplog):
    async def fake_recall(**kwargs):
        return [
            {
                "event_id": 42,
                "summary": f"memory for {kwargs['query']}",
                "recall_score": 0.88,
                "occurred_at": 1,
            }
        ]

    facets = build_recall_query_facets(latest_user_text="logging recall signal")

    with caplog.at_level(logging.DEBUG, logger="AICQ.memory.recall"):
        recalled = asyncio.run(
            recall_events_from_facets(
                sender_entity="self",
                context_scope="group:qq_1",
                limit=1,
                facets=facets,
                recall_fn=fake_recall,
            )
        )

    assert recalled[0]["event_id"] == 42
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "[recall] facet source=latest_user" in messages
    assert "[recall] fused" in messages
    assert "via source=latest_user" in messages
    assert "logging recall signal" in messages


def test_storyline_summary_inherits_and_sums_atom_recall_strength(monkeypatch):
    from memory.recall import recall_query
    from memory.recall.items import RecallItem
    import memory.recall.summary_recall as summary_recall

    events = [
        {"event_id": 1, "summary": "atom one", "recall_score": 0.7, "occurred_at": 10, "recall_reasons": ["atom:one"]},
        {"event_id": 2, "summary": "atom two", "recall_score": 0.4, "occurred_at": 20, "recall_reasons": ["atom:two"]},
        {"event_id": 3, "summary": "uncovered raw", "recall_score": 0.9, "occurred_at": 30},
    ]
    summaries = [
        {
            "memory_kind": "summary",
            "event_id": "summary:storyline-a",
            "summary_id": "storyline-a",
            "summary": "storyline A",
            "recall_score": 0.05,
            "occurred_at": 15,
            "source_event_ids": [1, 2, 99],
            "recall_reasons": ["summary:base"],
        },
        {
            "memory_kind": "summary",
            "event_id": "summary:storyline-b",
            "summary_id": "storyline-b",
            "summary": "storyline B",
            "recall_score": 0.05,
            "occurred_at": 12,
            "source_event_ids": [1],
        },
    ]

    async def fake_covering(**_kwargs):
        return [RecallItem.from_mapping(item) for item in summaries]

    monkeypatch.setattr(summary_recall, "load_ready_summaries_covering_events", fake_covering)

    recalled = asyncio.run(
        recall_query._augment_with_ready_summaries(
            events,
            context_scope="group:qq_1",
            limit=3,
            query="direct",
        )
    )

    assert [item["event_id"] for item in recalled] == ["summary:storyline-a", 3, "summary:storyline-b"]
    by_id = {item["event_id"]: item for item in recalled}
    assert by_id["summary:storyline-a"]["recall_score"] == 1.1
    assert by_id["summary:storyline-a"]["contributing_event_ids"] == [1, 2]
    assert "summary:score_summed_from_atoms" in by_id["summary:storyline-a"]["recall_reasons"]
    assert by_id["summary:storyline-b"]["recall_score"] == 0.7
    assert by_id["summary:storyline-b"]["contributing_event_ids"] == [1]
    assert 1 not in by_id
    assert 2 not in by_id


def test_default_recall_without_ready_summaries_respects_limit(monkeypatch):
    from memory.recall import recall_query
    import memory.recall.summary_recall as summary_recall
    import memory.repo.events as events_repo

    async def fake_recall(**_kwargs):
        return [
            {"event_id": 1, "summary": "A", "recall_score": 0.9, "occurred_at": 1},
            {"event_id": 2, "summary": "B", "recall_score": 0.8, "occurred_at": 2},
            {"event_id": 3, "summary": "C", "recall_score": 0.7, "occurred_at": 3},
            {"event_id": 4, "summary": "D", "recall_score": 0.6, "occurred_at": 4},
        ]

    async def no_ready_summaries(**_kwargs):
        return []

    monkeypatch.setattr(events_repo, "load_events_for_recall", fake_recall)
    monkeypatch.setattr(summary_recall, "load_ready_summaries_covering_events", no_ready_summaries)

    recalled = asyncio.run(
        recall_query.recall_events_from_facets(
            sender_entity="",
            context_scope="group:qq_1",
            limit=2,
            facets=[recall_query.RecallQueryFacet("latest_user", "bounded recall", 1.0)],
        )
    )

    assert [item["event_id"] for item in recalled] == [1, 2]


def test_summary_augmentation_failure_still_respects_limit(monkeypatch):
    from memory.recall import recall_query
    import memory.recall.summary_recall as summary_recall

    events = [
        {"event_id": 1, "summary": "A", "recall_score": 0.9, "occurred_at": 1},
        {"event_id": 2, "summary": "B", "recall_score": 0.8, "occurred_at": 2},
        {"event_id": 3, "summary": "C", "recall_score": 0.7, "occurred_at": 3},
    ]

    async def fail_summary_lookup(**_kwargs):
        raise RuntimeError("summary store unavailable")

    monkeypatch.setattr(summary_recall, "load_ready_summaries_covering_events", fail_summary_lookup)

    recalled = asyncio.run(
        recall_query._augment_with_ready_summaries(
            events,
            context_scope="group:qq_1",
            limit=2,
            query="bounded recall",
        )
    )

    assert [item["event_id"] for item in recalled] == [1, 2]


def test_storyline_summary_recall_item_contains_only_runtime_fields():
    from memory.recall.summary_recall import _summary_recall_item

    item = _summary_recall_item(
        row={"created_at_ms": 10, "updated_at_ms": 20},
        summary_id="summary:storyline:minimal",
        summary="只保留故事线文本。",
        event_ids={1, 2},
        relation_count=0,
        recall_score=0.8,
        recall_reasons=["summary:source_event_overlap"],
        occurred_at=30,
    ).to_dict()

    assert item["summary"] == "只保留故事线文本。"
    assert item["source_event_ids"] == [1, 2]
    assert "roles" not in item
    assert "storyline_summary" not in item
    assert "source_kind" not in item
    assert "source_id" not in item


def test_storyline_summary_sums_atoms_before_final_limit(monkeypatch):
    from memory.recall import recall_query
    from memory.recall.items import RecallItem
    import memory.recall.summary_recall as summary_recall
    import memory.repo.events as legacy_events_repo

    async def fake_recall(**kwargs):
        assert kwargs["limit"] >= 2
        return [
            {"event_id": 1, "summary": "atom one", "recall_score": 0.9, "occurred_at": 10},
            {"event_id": 2, "summary": "atom two", "recall_score": 0.8, "occurred_at": 20},
        ]

    async def fake_covering(**_kwargs):
        return [
            RecallItem.from_mapping({
                "memory_kind": "summary",
                "event_id": "summary:storyline-a",
                "summary_id": "storyline-a",
                "summary": "storyline A",
                "recall_score": 0.01,
                "occurred_at": 20,
                "source_event_ids": [1, 2],
            })
        ]

    monkeypatch.setattr(summary_recall, "load_ready_summaries_covering_events", fake_covering)
    monkeypatch.setattr(legacy_events_repo, "load_events_for_recall", fake_recall)

    recalled = asyncio.run(
        recall_query.recall_events_from_facets(
            sender_entity="",
            context_scope="group:qq_1",
            limit=1,
            facets=[
                recall_query.RecallQueryFacet("latest_user", "atom", 1.0),
            ],
        )
    )

    assert recalled[0]["event_id"] == "summary:storyline-a"
    assert recalled[0]["contributing_event_ids"] == [1, 2]
    assert recalled[0]["recall_score"] > 1.6
