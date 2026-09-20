from __future__ import annotations

import asyncio
import base64
import copy
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from llm.media import image_resolver, media_cache, vision_bridge
from platforms.qq.adapter import events
from platforms.qq.adapter.segments import build_message_content
from tools.core import examine_image, view_image


def _png():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("window", ["chat", "history", "forward"])
@pytest.mark.parametrize("warm_cache", [True, False])
def test_examination_preserves_visible_metadata_and_persists_results(window, warm_cache, monkeypatch, tmp_path):
    raw = _png()
    entry = build_message_content([{
        "type": "image", "data": {"base64": base64.b64encode(raw).decode("ascii")},
    }])
    ref = entry["content_segments"][0]["image_ref"]
    bridge = vision_bridge.VisionBridge({"enabled": False})
    bridge._enabled = True
    bridge._client = object()
    responses = iter(["description", "first-result", "second-result"])
    payloads = []

    def call_vlm(encoded, mime, *args):
        payloads.append(base64.b64decode(encoded))
        return next(responses)

    monkeypatch.setattr(bridge, "_call_vlm", call_vlm)
    bridge.process_entry(entry)
    session = SimpleNamespace(
        context_messages=[entry] if window == "chat" else [],
        is_browsing_history=lambda: window == "history",
        chat_window_view={"top_db_id": 1, "page_size": 1},
        forward_browser_stack=[{"nodes": [entry], "page_size": 1}] if window == "forward" else [],
    )
    # History loads fresh objects; its existing sidecar persistence must survive.
    monkeypatch.setattr(image_resolver, "load_history_window", lambda *_: [copy.deepcopy(entry)])
    handler = examine_image.make_handler(session, bridge)
    if not warm_cache:
        media_cache.clear_recent_media_cache()
    assert handler(ref, "first-focus")["result"] == "first-result"
    media_cache.clear_recent_media_cache()
    assert handler(ref, "second-focus")["result"] == "second-result"
    expected = [("first-focus", "first-result"), ("second-focus", "second-result")]
    from llm.media.image_store import read_image
    saved = read_image(ref)["examinations"]
    assert [(item["focus"], item["result"]) for item in saved] == expected
    assert image_resolver.ImageResolver(session).resolve(ref)[0]["examinations"] == saved
    assert payloads == [raw, raw, raw]


@pytest.mark.parametrize("outcome", ["success", "failed", "expired"])
def test_view_image_tracks_download_completion(outcome, monkeypatch):
    entry = build_message_content([{"type": "image", "data": {"url": "https://example.invalid/image"}}])
    ref = entry["content_segments"][0]["image_ref"]
    handler = view_image.make_handler(SimpleNamespace(context_messages=[entry]))
    assert handler(ref)["status"] == "pending"
    raw = _png()

    async def download(_):
        if outcome == "success":
            return base64.b64encode(raw).decode("ascii"), "image/png"
        return events._EXPIRED_SENTINEL if outcome == "expired" else None

    monkeypatch.setattr(events, "_fetch_image_b64", download)
    asyncio.run(events.download_pending_images(entry))
    result = handler(ref)
    if outcome == "success":
        assert result["ok"] is True
        assert result["_multimodal_parts"][0]["data"] == raw
    else:
        assert result["status"] == outcome


@pytest.mark.parametrize("status", ["pending", "failed", "expired", "unavailable_status"])
def test_unavailable_media_invalidates_cached_payload(status):
    ref = "state-fixture"
    media_cache.cache_recent_image(ref, {"data": _png(), "mime": "image/png"})
    media_cache.cache_recent_image(ref, {status: "missing_image" if status == "unavailable_status" else True})
    assert media_cache.get_recent_image(ref) is None
