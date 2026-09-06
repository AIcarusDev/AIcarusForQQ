from __future__ import annotations

import asyncio
import base64
import io
from datetime import timezone

import pytest
from PIL import Image

from llm.forward_browser import _normalize_forward_node
from llm.media.image_resolver import image_bytes, image_unavailable_status
from platforms.qq.adapter import events as qq_events
from platforms.qq.adapter.segments import build_content_segments


def _png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("segment_type, kind", [("image", "image"), ("mface", "sticker")])
def test_segment_only_conversion_preserves_refs_without_image_data(segment_type, kind):
    parts = build_content_segments([{"type": segment_type, "data": None}])
    assert parts[0]["type"] == kind
    assert parts[0]["image_ref"]


@pytest.fixture(params=["event", "forward_message", "forward_content"])
def parse_entry(request):
    def parse(message: list[dict]) -> dict:
        node = {"sender": {"user_id": "fixture", "nickname": "Fixture"}, "time": 1}
        if request.param == "event":
            entry = asyncio.run(qq_events.qq_adapter_event_to_context({
                **node,
                "post_type": "message",
                "message_type": "private",
                "message_id": "fixture-message",
                "message": message,
            }))
            assert entry is not None
            return entry
        field = "content" if request.param == "forward_content" else "message"
        return _normalize_forward_node(
            {**node, field: message},
            bot_id=None,
            bot_display_name="",
            timezone=timezone.utc,
        )

    return parse


@pytest.mark.parametrize("missing_position", [0, 2, 4], ids=["first", "middle", "last"])
@pytest.mark.parametrize("missing_segment, missing_kind", [
    ({"type": "mface", "data": {}}, "sticker"),
    ({"type": "image", "data": {}}, "image"),
    ({"type": "image", "data": {"sub_type": 1}}, "sticker"),
    ({"type": "image", "data": {"subType": 1}}, "sticker"),
], ids=["mface", "image", "snake_sticker", "camel_sticker"])
def test_each_image_keeps_its_own_payload_across_source_gaps(
    parse_entry, monkeypatch, missing_position, missing_segment, missing_kind,
):
    red, blue, green, yellow = [_png(color) for color in ("red", "blue", "green", "yellow")]
    sources = [
        ({"type": "image", "data": {"base64": base64.b64encode(red).decode()}}, "image", red),
        ({"type": "mface", "data": {"url": "https://example.invalid/blue"}}, "sticker", blue),
        ({"type": "image", "data": {
            "subType": 1, "base64": base64.b64encode(green).decode(),
            "url": "https://example.invalid/must-not-fetch",
        }}, "sticker", green),
        ({"type": "image", "data": {"url": "https://example.invalid/yellow"}}, "image", yellow),
    ]
    # Compare category labels to valid peers without freezing editable label wording.
    labels = {
        kind: next(iter(parse_entry([segment])["images"].values()))["label"]
        for segment, kind, _ in sources[:2]
    }
    sources.insert(missing_position, (missing_segment, missing_kind, None))
    message = [{"type": "reply", "data": {"id": "fixture-quote"}}]
    for segment, _, _ in sources:
        message.extend([
            {"type": "text", "data": {"text": ""}},
            segment,
            {"type": "text", "data": {"text": "separator"}},
        ])
    fetched = []

    async def fetch(url):
        fetched.append(url)
        raw = {"https://example.invalid/blue": blue, "https://example.invalid/yellow": yellow}[url]
        return base64.b64encode(raw).decode(), "image/png"

    monkeypatch.setattr(qq_events, "_fetch_image_b64", fetch)
    entry = parse_entry(message)
    parts = [part for part in entry["content_segments"] if part["type"] in ("image", "sticker")]
    assert len(parts) == len(sources)
    assert fetched == []
    for part, (segment, kind, expected) in zip(parts, sources, strict=True):
        assert part["type"] == kind
        info = entry["images"].get(part["image_ref"])
        assert info is not None
        assert info["label"] == labels[kind]
        if expected is None:
            assert image_bytes(info) is None
            assert image_unavailable_status(info) == "failed"
        elif segment["data"].get("base64"):
            assert image_bytes(info)[0] == expected
        else:
            assert image_bytes(info) is None
            assert image_unavailable_status(info) == "pending"

    assert asyncio.run(qq_events.download_pending_images(entry)) is True
    assert fetched == ["https://example.invalid/blue", "https://example.invalid/yellow"]
    for part, (_, kind, expected) in zip(parts, sources, strict=True):
        info = entry["images"][part["image_ref"]]
        assert info["label"] == labels[kind]
        if expected is None:
            assert image_bytes(info) is None
            assert image_unavailable_status(info) == "failed"
        else:
            assert image_bytes(info)[0] == expected
    assert asyncio.run(qq_events.download_pending_images(entry)) is False


def test_download_failures_stay_on_their_own_refs(parse_entry, monkeypatch):
    raw = _png("blue")
    message = [
        {"type": "mface", "data": {}},
        {"type": "image", "data": {"url": "https://example.invalid/failed"}},
        {"type": "mface", "data": {"url": "https://example.invalid/expired"}},
        {"type": "image", "data": {"url": "https://example.invalid/ok"}},
    ]

    async def fetch(url):
        return {
            "https://example.invalid/failed": None,
            "https://example.invalid/expired": qq_events._EXPIRED_SENTINEL,
            "https://example.invalid/ok": (base64.b64encode(raw).decode(), "image/png"),
        }[url]

    monkeypatch.setattr(qq_events, "_fetch_image_b64", fetch)
    entry = parse_entry(message)
    assert asyncio.run(qq_events.download_pending_images(entry)) is True
    images = [entry["images"].get(part["image_ref"]) for part in entry["content_segments"]]
    assert all(image is not None for image in images)
    assert [image_unavailable_status(image) for image in images[:3]] == ["failed", "failed", "expired"]
    assert all(image_bytes(image) is None for image in images[:3])
    assert image_bytes(images[3]) == (raw, "image/png")


def test_sourceless_images_do_not_borrow_unrelated_urls(parse_entry, monkeypatch):
    async def unexpected_fetch(url):
        pytest.fail(f"unexpected image fetch: {url}")

    monkeypatch.setattr(qq_events, "_fetch_image_b64", unexpected_fetch)
    entry = parse_entry([
        {"type": "mface", "data": {}},
        {"type": "image", "data": {"base64": "", "url": ""}},
        {"type": "file", "data": {"name": "fixture.txt", "url": "https://example.invalid/file"}},
    ])
    for part in entry["content_segments"][:2]:
        info = entry.get("images", {}).get(part["image_ref"])
        assert info is not None
        assert image_bytes(info) is None
        assert image_unavailable_status(info) == "failed"
    assert asyncio.run(qq_events.download_pending_images(entry)) is False
