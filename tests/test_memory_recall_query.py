import asyncio
import logging

from memory.recall_query import (
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
