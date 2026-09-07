from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from database import (
    init_db,
    load_chat_image_payload_sync,
    lookup_media_ref,
    lookup_media_ref_sync,
    save_chat_message,
    update_chat_message_recalled,
)
from llm.media.image_resolver import ImageResolver, image_bytes
from llm.media.media_cache import (
    RecentMediaCache,
    cache_recent_image,
    clear_recent_media_cache,
    get_recent_image,
)
from workspace.media import register_workspace_image
from test_sticker_collection import png
from llm.media.image_store import register_image


def test_recent_media_cache_lru() -> None:
    cache = RecentMediaCache(capacity=3)
    cache.put("ref1", {"data": b"1"}, source="chat")
    cache.put("ref2", {"data": b"2"}, source="chat")
    cache.put("ref3", {"data": b"3"}, source="chat")

    assert len(cache) == 3
    assert cache.get("ref1") == ({"data": b"1"}, "chat")

    # Adding 4th should evict ref2 because ref1 was accessed recently
    cache.put("ref4", {"data": b"4"}, source="chat")
    assert len(cache) == 3
    assert cache.get("ref2") is None
    assert cache.get("ref1") == ({"data": b"1"}, "chat")
    assert cache.get("ref3") == ({"data": b"3"}, "chat")
    assert cache.get("ref4") == ({"data": b"4"}, "chat")


def test_media_registry_crud_and_chat_indexing() -> None:
    async def _run() -> None:
        await init_db()

        test_ref = "test_manual_ref"
        register_image(png("blue"), "chat", test_ref)
        record_sync = lookup_media_ref_sync(test_ref)
        assert record_sync["image_ref"] == test_ref
        assert await lookup_media_ref(test_ref) == record_sync

        # Test save_chat_message auto-indexing into media_registry
        session_key = f"qq:group:{uuid.uuid4().hex[:6]}"
        message_id = f"msg_{uuid.uuid4().hex[:6]}"
        chat_ref = f"chat_{uuid.uuid4().hex[:6]}"

        entry = {
            "role": "user",
            "message_id": message_id,
            "sender_id": "12345",
            "content": "[图片]",
            "images": {
                chat_ref: {
                    "base64": base64.b64encode(png()).decode(),
                    "mime": "image/jpeg",
                    "label": "图片",
                }
            },
        }

        await save_chat_message(session_key, entry)

        # Verify that media_registry recorded chat_ref
        auto_record = lookup_media_ref_sync(chat_ref)
        assert auto_record is not None
        assert auto_record["source_type"] == "chat"

        # Verify payload lookup
        payload = load_chat_image_payload_sync(session_key, message_id, chat_ref)
        assert payload is not None
        res_bytes = image_bytes(payload)
        assert res_bytes is not None
        assert res_bytes[0] == png()

    asyncio.run(_run())


def test_image_resolver_with_l1_cache() -> None:
    clear_recent_media_cache()
    ref = "l1_test_ref"
    session = SimpleNamespace(
        context_messages=[],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )
    resolver = ImageResolver(session, browser_image_reader=lambda _: None)

    # Before caching, should not be found
    assert resolver.resolve(ref) is None

    # Cache into L1
    cache_recent_image(ref, {"base64": "bG1fY2FjaGVk"}, source="chat")

    resolved = resolver.resolve(ref)
    assert resolved is None  # LRU entries cannot invent an unregistered identity.


def test_image_resolver_with_l2_sqlite_fallback() -> None:
    async def _run() -> None:
        await init_db()
        clear_recent_media_cache()

        session_key = f"qq:private:{uuid.uuid4().hex[:6]}"
        message_id = f"msg_{uuid.uuid4().hex[:6]}"
        cold_ref = f"cold_{uuid.uuid4().hex[:6]}"

        entry = {
            "role": "user",
            "message_id": message_id,
            "sender_id": "11111",
            "content": "[图片]",
            "images": {
                cold_ref: {
                    "base64": base64.b64encode(png()).decode(),
                    "mime": "image/png",
                    "label": "图片",
                }
            },
        }
        await save_chat_message(session_key, entry)
        clear_recent_media_cache()  # Clear L1 to force L2 fallback

        # A completely unrelated session with empty context
        other_session = SimpleNamespace(
            context_messages=[],
            is_browsing_history=lambda: False,
            forward_browser_stack=[],
        )
        resolver = ImageResolver(other_session, browser_image_reader=lambda _: None)

        resolved = resolver.resolve(cold_ref)
        assert resolved is not None
        img_data, source = resolved
        assert image_bytes(img_data)[0] == png()
        assert source == "chat"

        # Verify that it backfilled L1 cache
        cached = get_recent_image(cold_ref)
        assert cached is not None
        assert image_bytes(cached[0])[0] == png()

    asyncio.run(_run())


def test_workspace_media_registration(tmp_path: Path) -> None:
    async def _run() -> None:
        await init_db()
        clear_recent_media_cache()

        test_file = tmp_path / "plot.png"
        test_file.write_bytes(png())

        ws_ref = await register_workspace_image(test_file)
        assert ws_ref is not None

        session = SimpleNamespace(
            context_messages=[],
            is_browsing_history=lambda: False,
            forward_browser_stack=[],
        )
        resolver = ImageResolver(session, browser_image_reader=lambda _: None)

        # 1. Resolves via L1
        resolved = resolver.resolve(ws_ref)
        assert resolved is not None
        assert resolved[0]["data"] == png()
        assert resolved[0]["mime"] == "image/png"
        assert resolved[1] == "workspace"

        # 2. Resolves via L2 after L1 clear
        clear_recent_media_cache()
        resolved_l2 = resolver.resolve(ws_ref)
        assert resolved_l2 is not None
        assert resolved_l2[0]["data"] == png()
        assert resolved_l2[1] == "workspace"

    asyncio.run(_run())


@pytest.mark.parametrize("recalled", [False, True])
def test_workspace_registration_reuses_only_readable_chat_refs(tmp_path: Path, recalled: bool) -> None:
    async def _run() -> None:
        await init_db()
        raw = png()
        old_ref = "legacy_chat_image"
        session_key = "qq:group:test"
        await save_chat_message(session_key, {
            "role": "user",
            "message_id": "1",
            "images": {old_ref: {
                "base64": base64.b64encode(raw).decode("ascii"),
                "mime": "image/png",
            }},
        })
        if recalled:
            await update_chat_message_recalled("1", "recalled", "", session_key=session_key)

        original = tmp_path / "local.png"
        original.write_bytes(raw)
        ref = await register_workspace_image(original)
        if not recalled:
            assert ref == old_ref
        original.unlink()
        clear_recent_media_cache()

        resolved = ImageResolver(browser_image_reader=lambda _: None).resolve(ref)
        assert resolved is not None
        assert image_bytes(resolved[0]) == (raw, "image/png")

    asyncio.run(_run())


def test_time_ref_generation_and_storage(tmp_path: Path) -> None:
    from datetime import datetime, timezone
    from llm.media.media_storage import generate_time_ref, parse_time_ref_date, save_media_bytes, read_media_bytes

    dt = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    ref = generate_time_ref(dt)
    assert len(ref) == 12
    assert ref.startswith("260907_")

    parsed = parse_time_ref_date(ref)
    assert parsed == (2026, 9, 7)

    # Test saving & reading
    assigned_ref, disk_path = save_media_bytes(png(), mime="image/png", image_ref=ref, dt=dt)
    assert assigned_ref == ref
    assert disk_path.exists()
    assert "2026" in str(disk_path)
    assert "09" in str(disk_path)

    read_res = read_media_bytes(ref)
    assert read_res is not None
    data, mime = read_res
    assert data == png()
    assert mime == "image/png"
