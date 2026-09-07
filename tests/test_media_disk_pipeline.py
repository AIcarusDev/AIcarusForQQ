from __future__ import annotations

import asyncio
import base64
import io
import sqlite3

import pytest
from PIL import Image

import database
from llm.media import media_storage, vision_bridge
from llm.media.image_resolver import ImageResolver, image_bytes
from platforms.chat.xml_builder import build_multimodal_content
from platforms.qq.adapter.segments import build_message_content


def png():
    stream = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(stream, format="PNG")
    return stream.getvalue()


@pytest.mark.parametrize("source", ["inline", "download", "legacy_migration"])
def test_disk_images_reach_model_and_vision_bridge(source, monkeypatch):
    raw = png()
    b64 = base64.b64encode(raw).decode("ascii")
    content = build_message_content([{
        "type": "image", "data": {"base64": b64} if source == "inline" else {"url": "https://example.invalid/image"},
    }])
    ref = content["content_segments"][0]["image_ref"]
    entry = {
        "role": "user", "message_id": "fixture", "content": "", "content_type": "image",
        "timestamp": "2026-09-07T00:00:00+00:00", "sender_name": "Fixture", **content,
    }
    if source == "download":
        from platforms.qq.adapter import events

        async def download(_url):
            return b64, "image/png"

        monkeypatch.setattr(events, "_fetch_image_b64", download)
        asyncio.run(events.download_pending_images(entry))
    elif source == "legacy_migration":
        async def migrate():
            await database.init_db()
            entry["images"] = {ref: {"base64": b64, "mime": "image/png"}}
            await database.save_chat_message("qq:private:fixture", entry)
            with sqlite3.connect(database.DB_PATH) as conn:
                conn.execute("DELETE FROM _migrations WHERE name='chat_images_slimming_to_disk_v1'")
            await database.init_db()

        asyncio.run(migrate())
        entry["images"] = {ref: database.load_chat_image_payload_sync("qq:private:fixture", "fixture", ref)}

    assert "base64" not in entry["images"][ref]
    assert image_bytes(entry["images"][ref])[0] == raw
    parts = build_multimodal_content([entry])
    urls = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    assert len(urls) == 1
    assert base64.b64decode(urls[0].split(",", 1)[1]) == raw

    calls = []
    bridge = vision_bridge.VisionBridge({"enabled": False})
    bridge._enabled = True
    bridge._client = object()
    monkeypatch.setattr(bridge, "_call_vlm", lambda encoded, mime, *args: calls.append(base64.b64decode(encoded)) or "fixture-description")
    bridge.process_entry(entry)
    assert calls == [raw]
    assert entry["images"][ref]["description"] == "fixture-description"
    assert "base64" not in entry["images"][ref]


@pytest.mark.parametrize("ref", ["../outside", "..\\outside", "*", "ab??", "ab[cd]", "/outside", "C:\\outside", "aaaa:bbbb"])
def test_media_refs_cannot_address_paths_or_globs(ref, tmp_path):
    media_storage.MEDIA_ROOT.mkdir()
    (tmp_path / "outside.png").write_bytes(png())
    assert media_storage.read_media_bytes(ref) is None
    assert ImageResolver(browser_image_reader=lambda _: None).resolve(ref) is None
    with pytest.raises(ValueError):
        media_storage.save_media_bytes(b"replacement", image_ref=ref)
    assert (tmp_path / "outside.png").read_bytes() == png()


def test_media_store_rejects_symlinks_outside_root(tmp_path):
    media_storage.MEDIA_ROOT.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(png())
    link = media_storage.MEDIA_ROOT / "legacyref.png"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Host cannot create symlinks")
    assert media_storage.read_media_bytes("legacyref") is None
    folder = media_storage.MEDIA_ROOT / "2026"
    folder.symlink_to(tmp_path, target_is_directory=True)
    assert media_storage.read_media_bytes("260907_abcde") is None
    with pytest.raises(ValueError):
        media_storage.save_media_bytes(b"replacement", image_ref="260907_abcde")


def test_legacy_and_time_refs_remain_readable():
    for ref in ("a" * 12, "260907_abcde"):
        media_storage.save_media_bytes(png(), mime="image/png", image_ref=ref)
        assert media_storage.read_media_bytes(ref) == (png(), "image/png")
