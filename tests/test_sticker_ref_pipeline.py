from __future__ import annotations

import asyncio
import base64
import json
import threading
from types import SimpleNamespace

import pytest
from quart import Quart
from werkzeug.datastructures import FileStorage

import app_state
import database
from llm.core.tool_calling.schema import validate_arguments_by_declaration
from llm.media import image_cache, image_resolver, sticker_collection as stickers
from llm.media.image_importer import ImageImportError, ImageImporter
from llm.media.image_resolver import ImageResolver
from llm.media.media_cache import cache_recent_image, clear_recent_media_cache
from platforms.chat.xml_builder import build_chat_log_xml
from platforms.core.prompt import _segment_text
from platforms.qq.adapter.segments import ImageLoadError, llm_segments_to_qq_adapter
from platforms.qq.tools.qq_social.send_message import send_message as send
from platforms.qq.tools.qq_stickers import delete_sticker, list_stickers, save_sticker, update_sticker
from tools.core import examine_image, view_image
from web import routes_dashboard, routes_settings

from test_sticker_collection import gif, png


def session_with_ref(ref, raw):
    return SimpleNamespace(
        context_messages=[{"images": {ref: {"base64": base64.b64encode(raw).decode(), "mime": "image/gif"}}}],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )


def register_chat_image_lookup(monkeypatch, ref, entry):
    payload = entry["images"][ref]
    monkeypatch.setattr(
        database,
        "lookup_media_ref_sync",
        lambda image_ref: {
            "source_type": "chat",
            "locator": "qq:group:history::history-message",
            "mime": payload["mime"],
        }
        if image_ref == ref
        else None,
    )
    monkeypatch.setattr(
        database,
        "load_chat_image_payload_sync",
        lambda _session_key, _message_id, image_ref: payload if image_ref == ref else None,
    )


class Workspace:
    def __init__(self, destination, *, fail=False):
        self.destination = destination
        self.data = bytearray()
        self.fail = fail
        self.aborted = False

    async def begin_file_import(self, path, size):
        self.data = bytearray()
        return self

    async def write(self, chunk):
        self.data.extend(chunk)
        if self.fail:
            raise OSError("isolated partial import failure")

    async def finish(self):
        if self.destination.exists():
            return {"ok": False, "code": "already_exists"}
        self.destination.write_bytes(self.data)
        return {"ok": True, "size_bytes": len(self.data)}

    async def abort(self):
        self.aborted = True
        self.data.clear()


def test_saved_gif_works_after_original_context_disappears(monkeypatch, tmp_path):
    main, alias = "a" * 12, "b" * 12
    raw = gif()
    session = session_with_ref(main, raw)
    saved = save_sticker.make_handler(lambda: session)(main, "first")
    assert saved["image_ref"] == main
    assert saved["duplicate"] is False
    session.context_messages = session_with_ref(alias, raw).context_messages
    duplicate = save_sticker.make_handler(lambda: session)(alias, "ignored")
    assert duplicate["image_ref"] == main
    session.context_messages = []
    clear_recent_media_cache()
    monkeypatch.setattr(image_cache, "read_image_b64", lambda *_: pytest.fail("must use durable image bytes"))
    assert ImageResolver(session).resolve(alias)[0]["data"] == raw
    viewed = view_image.make_handler(session)(image_ref=alias)
    assert viewed["image_ref"] == main
    assert viewed["_multimodal_parts"][0]["data"] == raw
    calls = []
    bridge = SimpleNamespace(enabled=True, examine=lambda *args: calls.append(args) or "observed")
    examined = examine_image.make_handler(session, bridge)(alias, "movement")
    assert examined["image_ref"] == main
    assert base64.b64decode(calls[0][1]) == raw
    workspace = Workspace(tmp_path / "copy.gif")
    importer = ImageImporter(session, workspace)
    asyncio.run(importer.save(image_ref=alias, path="/home/agent/copy.gif"))
    assert workspace.destination.read_bytes() == raw
    with pytest.raises(ImageImportError) as error:
        asyncio.run(importer.save(image_ref=main, path="/home/agent/copy.gif"))
    assert error.value.code == "already_exists"
    assert delete_sticker.execute(alias)["image_ref"] == main
    assert workspace.destination.read_bytes() == raw


def test_failed_workspace_import_of_collection_aborts_without_changing_collection(tmp_path):
    ref, raw = "a" * 12, gif()
    stickers.save_sticker(raw, "image/gif", "first", image_ref=ref)
    workspace = Workspace(tmp_path / "copy.gif", fail=True)
    with pytest.raises(OSError):
        asyncio.run(ImageImporter(SimpleNamespace(context_messages=[]), workspace).save(
            image_ref=ref, path="/home/agent/copy.gif",
        ))
    assert workspace.aborted
    assert not workspace.destination.exists()
    assert stickers.load_sticker_bytes(ref) == (raw, "image/gif")


def test_registered_but_changed_image_does_not_fall_back_to_context():
    ref = "a" * 12
    stickers.save_sticker(png(), "image/png", "first", image_ref=ref)
    (stickers._IMAGES_DIR / f"{ref}.png").write_bytes(png("blue"))
    session = session_with_ref(ref, png("green"))
    image, source = ImageResolver(session).resolve(ref)
    assert source == "sticker"
    assert ImageResolver.payload(image) is None
    assert ImageResolver.unavailable_status(image) == "image_changed"
    prepared, error, _ = send._prepare_sendable_segments([{"command": "sticker", "image_ref": ref}], session)
    assert prepared is None
    assert error


def test_invalid_chat_image_bytes_cannot_be_sent_as_a_sticker():
    ref = "a" * 12
    prepared, error, _ = send._prepare_sendable_segments(
        [{"command": "text", "content": "must not leak"}, {"command": "sticker", "image_ref": ref}],
        session_with_ref(ref, b"not an image"),
    )
    assert prepared is None
    assert error


@pytest.mark.parametrize("source", ["chat", "history", "forward"])
def test_uncollected_visible_refs_can_be_sent_and_collected(source, monkeypatch):
    ref, raw = "a" * 12, gif()
    session = session_with_ref(ref, raw)
    entry = session.context_messages[0]
    if source != "chat":
        session.context_messages = []
    if source == "history":
        session.is_browsing_history = lambda: True
        session.chat_window_view = {"top_db_id": 7}
        register_chat_image_lookup(monkeypatch, ref, entry)
    if source == "forward":
        session.forward_browser_stack = [{"nodes": [entry], "page_offset": 0, "page_size": 1}]
        cache_recent_image(ref, entry["images"][ref], source="chat")
    prepared, error, warnings = send._prepare_sendable_segments([{"command": "sticker", "image_ref": ref}], session)
    assert error is None
    assert warnings == []
    assert prepared[0]["_image_bytes"] == raw
    assert stickers.list_all() == []
    assert save_sticker.make_handler(lambda: session)(ref, "first")["image_ref"] == ref


def test_hidden_forward_ref_falls_back_to_browser(monkeypatch):
    ref = "a" * 12
    entry = session_with_ref(ref, gif()).context_messages[0]
    session = SimpleNamespace(
        context_messages=[],
        forward_browser_stack=[{"nodes": [entry, {}], "page_offset": 1, "page_size": 1}],
    )
    monkeypatch.setattr(image_resolver, "read_browser_image_file", lambda *_: (gif(), "image/gif"))
    image, source = ImageResolver(session).resolve(ref)
    assert source == "browser"
    assert image["data"] == gif()


@pytest.mark.parametrize("adapter,flag", [("napcat", "sub_type"), ("llonebot", "subType")])
def test_prepared_sticker_keeps_gif_bytes_after_collection_is_deleted(adapter, flag):
    main, alias, raw = "a" * 12, "b" * 12, gif()
    stickers.save_sticker(raw, "image/gif", "first", image_ref=main)
    stickers.save_sticker(raw, "image/gif", "ignored", image_ref=alias)
    prepared, error, _ = send._prepare_sendable_segments(
        [{"command": "sticker", "image_ref": alias}], SimpleNamespace(context_messages=[]),
    )
    assert error is None
    stickers.delete_sticker(main)
    converted = llm_segments_to_qq_adapter(prepared, adapter=adapter)
    assert converted[0]["data"][flag] == 1
    assert base64.b64decode(converted[0]["data"]["file"].removeprefix("base64://")) == raw
    assert send._extract_message_text(prepared)[1] == [{"type": "sticker", "image_ref": main}]


def test_send_schema_accepts_image_ref_and_rejects_old_or_ambiguous_fields():
    declaration = send.SEND_MESSAGE_ARRAY_CONTRACT.declaration()

    def arguments(segment):
        return {"messages": [{"segments": [segment]}]}
    assert validate_arguments_by_declaration(arguments({"command": "sticker", "image_ref": "a" * 12}), declaration)[0]
    for segment in (
        {"command": "sticker", "sticker_id": "000"},
        {"command": "sticker", "image_ref": "000"},
        {"command": "sticker", "sticker_id": "000", "image_ref": "a" * 12},
    ):
        assert not validate_arguments_by_declaration(arguments(segment), declaration)[0]
        assert send._prepare_sendable_segments([segment], SimpleNamespace(context_messages=[]))[0] is None
        with pytest.raises(ImageLoadError):
            llm_segments_to_qq_adapter([segment])


@pytest.mark.parametrize("module", [save_sticker, update_sticker, delete_sticker])
def test_collection_tool_schemas_have_only_image_refs(module):
    extra = {"description": "fixture impression"} if module is not delete_sticker else {}
    declaration = module.TOOL_CONTRACT.declaration()
    assert validate_arguments_by_declaration({"image_ref": "a" * 12, **extra}, declaration)[0]
    assert not validate_arguments_by_declaration({"sticker_id": "000", **extra}, declaration)[0]
    assert not validate_arguments_by_declaration({"image_ref": "000", **extra}, declaration)[0]


def test_list_and_edit_results_return_primary_ref():
    main, alias = "a" * 12, "b" * 12
    stickers.save_sticker(png(), "image/png", "first", image_ref=main)
    stickers.save_sticker(png(), "image/png", "ignored", image_ref=alias)
    assert update_sticker.execute(alias, "revised")["image_ref"] == main
    result = list_stickers.make_handler({"vision": True})()
    assert result["stickers"] == [{"image_ref": main, "description": "revised"}]
    assert result["_multimodal_parts"][0]["data"].startswith(b"\xff\xd8")


@pytest.fixture
def fake_sender(fake_session, monkeypatch):
    import database
    import web.debug_server

    stored, broadcasts = [], []

    async def persist(_key, entry):
        stored.append(entry)

    async def broadcast(event):
        broadcasts.append(event)

    class Client:
        connected = True
        adapter = "napcat"
        last_api_error = None

        def __init__(self):
            self.calls = []

        async def send_message(self, **kwargs):
            self.calls.append(kwargs)
            return {"message_id": f"sent-{len(self.calls)}"}

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    fake_session.key = "qq:group:1234"
    monkeypatch.setattr(database, "save_chat_message", persist)
    monkeypatch.setattr(web.debug_server, "broadcast_chat_event", broadcast)
    monkeypatch.setattr(app_state, "main_loop", loop)
    client = Client()
    yield fake_session, client, send.make_handler(lambda: fake_session, client), stored
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=2)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def test_invalid_sticker_fails_its_whole_message_and_batch_continues(fake_sender):
    session, client, handler, stored = fake_sender
    main = "a" * 12
    stickers.save_sticker(png(), "image/png", "first", image_ref=main)
    result = handler(messages=[
        {"segments": [{"command": "text", "content": "must not leak"}, {"command": "sticker", "image_ref": "missing"}]},
        {"segments": [{"command": "sticker", "image_ref": main}]},
    ])
    assert result["sent_count"] == 1
    assert result["failed_count"] == 1
    assert len(client.calls) == 1
    assert session.context_messages[0]["content_segments"] == [{"type": "sticker", "image_ref": main}]
    assert "_image_bytes" not in json.dumps(session.context_messages)
    assert "must not leak" not in json.dumps(client.calls)


def test_history_refs_remain_sendable_when_send_snaps_window_to_latest(fake_sender, monkeypatch):
    session, client, handler, _ = fake_sender
    ref, raw = "a" * 12, gif()
    entry = session_with_ref(ref, raw).context_messages[0]
    session.chat_window_view = {"mode": "history", "top_db_id": 7}
    # Use class methods so the copied image session reads its own snapshotted window.
    monkeypatch.setattr(type(session), "is_browsing_history", lambda self: self.chat_window_view["mode"] == "history", raising=False)
    monkeypatch.setattr(type(session), "reset_chat_window_view", lambda self: setattr(self, "chat_window_view", {"mode": "live"}), raising=False)
    register_chat_image_lookup(monkeypatch, ref, entry)
    result = handler(messages=[{"segments": [{"command": "sticker", "image_ref": ref}]}] * 2)
    assert result["sent_count"] == 2
    assert session.chat_window_view["mode"] == "live"
    assert all(base64.b64decode(call["message"][0]["data"]["file"].removeprefix("base64://")) == raw for call in client.calls)
    assert session.context_messages[0]["images"][ref]["base64"] == base64.b64encode(raw).decode()


def test_history_rendering_uses_refs_and_never_guesses_legacy_number_identity():
    ref = "a" * 12
    for segment in ({"type": "sticker", "image_ref": ref}, {"type": "sticker", "sticker_id": "003"}):
        message = {
            "role": "bot", "message_id": "sent-one", "sender_id": "bot",
            "timestamp": "2026-09-06T00:00:00+00:00", "content_segments": [segment],
        }
        rendered = build_chat_log_xml([message], {"type": "group", "id": "fixture", "name": "Fixture"})
        core_text = _segment_text(segment)
        if "image_ref" in segment:
            assert ref in rendered and ref in core_text
        else:
            assert "003" not in rendered and "003" not in core_text
        assert "sticker_id" not in rendered and "sticker_id" not in core_text


def test_web_upload_list_read_edit_delete_use_refs_and_reject_numbers():
    import io

    async def exercise():
        app = Quart(__name__)
        app.register_blueprint(routes_settings.settings_bp)
        app.register_blueprint(routes_dashboard.dashboard_bp)
        client = app.test_client()
        raw = gif()
        uploaded = await client.post(
            "/api/stickers/upload", form={"description": "first"},
            files={"file": FileStorage(stream=io.BytesIO(raw), filename="sample.gif", content_type="image/gif")},
        )
        assert uploaded.status_code == 200
        result = await uploaded.get_json()
        ref = result["image_ref"]
        assert "id" not in result
        alias = "b" * 12
        stickers.save_sticker(raw, "image/gif", "ignored", image_ref=alias)
        listed = await (await client.get("/api/stickers/list")).get_json()
        assert listed["stickers"][0]["image_ref"] == ref
        image = await client.get(f"/api/sticker/{alias}")
        assert await image.get_data() == raw
        changed = await client.patch(f"/api/stickers/{alias}", json={"description": "changed"})
        assert (await changed.get_json())["image_ref"] == ref
        assert (await client.delete("/api/stickers/000")).status_code == 400
        assert (await client.get("/api/sticker/000")).status_code == 400
        deleted = await client.delete(f"/api/stickers/{alias}")
        assert (await deleted.get_json())["image_ref"] == ref
        assert (await client.get(f"/api/sticker/{ref}")).status_code == 404
        assert (await client.patch(f"/api/stickers/{alias}", json={"description": "gone"})).status_code == 404

    asyncio.run(exercise())


def test_web_corrupt_collection_returns_explicit_failure():
    stickers._STICKER_DIR.mkdir()
    stickers._INDEX_PATH.write_bytes(b"{")

    async def exercise():
        app = Quart(__name__)
        app.register_blueprint(routes_settings.settings_bp)
        response = await app.test_client().get("/api/stickers/list")
        assert response.status_code == 503
        assert (await response.get_json())["code"] == "invalid_index"

    asyncio.run(exercise())
