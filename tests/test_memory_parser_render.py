from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memory.event_extraction.parser import EventExtractionParseFatalError, parse_event_extraction_output
from memory.recall.render import build_memory_debug_xml, build_memory_xml


def test_parse_event_extraction_output_accepts_valid_events_and_reports_bad_siblings():
    raw = """
    <extract>
      <event>{"summary":"prefers concise replies","source_id":"1","event_type":"preference","roles":[]}</event>
      <event>```{"summary":"bad","event_type":"format","roles":[]}```</event>
    </extract>
    """

    result = parse_event_extraction_output(raw)

    assert [item.event["summary"] for item in result.events] == ["prefers concise replies"]
    assert len(result.errors) == 1


def test_parse_event_extraction_output_falls_back_to_complete_event_json():
    raw = 'noise {"summary":"uses sandbox data","source_id":"1","event_type":"fact","roles":[]}'

    result = parse_event_extraction_output(raw)

    assert len(result.events) == 1
    assert result.events[0].event["event_type"] == "fact"
    assert result.errors


def test_parse_event_extraction_output_repairs_unescaped_quotes_inside_event_string():
    raw = (
        '<extract><event>{"summary":"我在Pixiv搜索画师"Sacrai"，但未找到对应作品",'
        '"source_id":"1","event_type":"search","roles":[]}</event></extract>'
    )

    result = parse_event_extraction_output(raw)

    assert len(result.events) == 1
    assert result.events[0].event["summary"] == '我在Pixiv搜索画师"Sacrai"，但未找到对应作品'
    assert '\\"Sacrai\\"' in result.events[0].raw_json
    assert result.errors


def test_parse_event_extraction_output_raises_when_no_extract_or_recoverable_event_exists():
    with pytest.raises(EventExtractionParseFatalError):
        parse_event_extraction_output("plain text only")


def test_build_memory_xml_is_minimal_escaped_and_relative():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event_time = int((now - timedelta(hours=2)).timestamp() * 1000)
    event = {
        "event_id": "evt-1",
        "summary": "Alice < Bob",
        "created_at": event_time,
        "confidence": 1.5,
        "recall_score": 9,
    }

    xml = build_memory_xml(now=now, recalled_events=[event])

    assert "Alice &lt; Bob" in xml
    assert 'when="2' in xml
    assert 'confidence="1.00"' in xml
    assert "evt-1" not in xml
    assert "recall_score" not in xml


def test_build_memory_debug_xml_keeps_diagnostics_outside_normal_memory():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event_time = int((now - timedelta(minutes=3)).timestamp() * 1000)
    event = {
        "event_id": "evt-1",
        "summary": "debug summary",
        "created_at": event_time,
        "event_type": "fact",
        "recall_score": 0.7,
        "recall_path": ["seed", "evt-1"],
    }

    debug_xml = build_memory_debug_xml(now=now, recalled_events=[event])

    assert debug_xml.startswith('<memory_debug items="1">')
    assert 'id="evt-1"' in debug_xml
    assert "seed -&gt; evt-1" in debug_xml
