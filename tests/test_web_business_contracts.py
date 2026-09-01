from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from quart import Quart
from PIL import Image

import app_state
from web import routes_dashboard, routes_settings


class _FakeCoreSession:
    def __init__(self) -> None:
        self.conv_type = ""
        self.entries: list[dict] = []
        self.unread: list[str] = []

    def set_conversation_meta(self, conv_type, conv_id, conv_name, *, platform) -> None:
        self.conv_type = conv_type

    def add_to_context(self, entry: dict) -> None:
        self.entries.append(entry)

    def mark_unread_message(self, message_id: str) -> None:
        self.unread.append(message_id)


def test_core_chat_client_id_is_idempotent(monkeypatch) -> None:
    async def scenario() -> None:
        persisted: dict[str, dict] = {}
        session = _FakeCoreSession()
        wake_count = 0

        async def existing_ids(_session_key: str, message_ids: list[str]) -> set[str]:
            return {message_id for message_id in message_ids if message_id in persisted}

        async def save_message(_session_key: str, entry: dict) -> None:
            persisted[entry["message_id"]] = dict(entry)

        async def load_message(message_id: str) -> dict | None:
            return persisted.get(message_id)

        async def upsert_session(*_args, **_kwargs) -> None:
            return None

        def wake() -> None:
            nonlocal wake_count
            wake_count += 1

        monkeypatch.setattr(routes_dashboard, "get_existing_chat_message_ids", existing_ids)
        monkeypatch.setattr(routes_dashboard, "get_chat_message_by_id", load_message)
        monkeypatch.setattr(routes_dashboard, "save_chat_message", save_message)
        monkeypatch.setattr(routes_dashboard, "upsert_chat_session", upsert_session)
        monkeypatch.setattr(routes_dashboard, "get_or_create_session", lambda _focus: session)
        monkeypatch.setattr(routes_dashboard, "_guardian_meta", lambda: ("guardian", "监护人"))
        monkeypatch.setattr(routes_dashboard, "_wake_for_core_message", wake)
        monkeypatch.setattr(app_state, "current_focus", object())
        monkeypatch.setattr(app_state, "first_input_event", SimpleNamespace(set=lambda: None))
        monkeypatch.setattr(app_state, "TIMEZONE", None)

        app = Quart(__name__)
        app.register_blueprint(routes_dashboard.dashboard_bp)
        client = app.test_client()

        first = await client.post(
            "/api/core/chat",
            json={"content": "同一条消息", "client_id": "client-request-1"},
        )
        retry = await client.post(
            "/api/core/chat",
            json={"content": "同一条消息", "client_id": "client-request-1"},
        )
        first_payload = await first.get_json()
        retry_payload = await retry.get_json()

        assert first.status_code == 200
        assert retry.status_code == 200
        assert first_payload["duplicate"] is False
        assert retry_payload["duplicate"] is True
        assert first_payload["message"]["message_id"] == retry_payload["message"]["message_id"]
        assert len(persisted) == 1
        assert len(session.entries) == 1
        assert len(session.unread) == 1
        assert wake_count == 1

    asyncio.run(scenario())


def test_core_chat_rejects_invalid_client_id_without_writing(monkeypatch) -> None:
    async def scenario() -> None:
        app = Quart(__name__)
        app.register_blueprint(routes_dashboard.dashboard_bp)
        client = app.test_client()

        blank = await client.post(
            "/api/core/chat",
            json={"content": "hello", "client_id": "   "},
        )
        too_long = await client.post(
            "/api/core/chat",
            json={"content": "hello", "client_id": "x" * 129},
        )

        assert blank.status_code == 400
        assert too_long.status_code == 400
        assert "client_id" in (await blank.get_json())["error"]
        assert "128" in (await too_long.get_json())["error"]

    asyncio.run(scenario())


def test_core_chat_without_client_id_preserves_legacy_non_idempotent_behavior(monkeypatch) -> None:
    async def scenario() -> None:
        persisted: list[dict] = []
        session = _FakeCoreSession()

        async def save_message(_session_key: str, entry: dict) -> None:
            persisted.append(dict(entry))

        async def upsert_session(*_args, **_kwargs) -> None:
            return None

        monkeypatch.setattr(routes_dashboard, "save_chat_message", save_message)
        monkeypatch.setattr(routes_dashboard, "upsert_chat_session", upsert_session)
        monkeypatch.setattr(routes_dashboard, "get_or_create_session", lambda _focus: session)
        monkeypatch.setattr(routes_dashboard, "_guardian_meta", lambda: ("guardian", "监护人"))
        monkeypatch.setattr(routes_dashboard, "_wake_for_core_message", lambda: None)
        monkeypatch.setattr(app_state, "current_focus", object())
        monkeypatch.setattr(app_state, "first_input_event", SimpleNamespace(set=lambda: None))
        monkeypatch.setattr(app_state, "TIMEZONE", None)

        app = Quart(__name__)
        app.register_blueprint(routes_dashboard.dashboard_bp)
        client = app.test_client()

        first = await client.post("/api/core/chat", json={"content": "旧版消息"})
        second = await client.post("/api/core/chat", json={"content": "旧版消息"})
        first_payload = await first.get_json()
        second_payload = await second.get_json()

        assert first_payload["duplicate"] is False
        assert second_payload["duplicate"] is False
        assert first_payload["client_id"] is None
        assert second_payload["client_id"] is None
        assert len(persisted) == 2
        assert persisted[0]["message_id"].startswith("core_")
        assert persisted[1]["message_id"].startswith("core_")
        assert persisted[0]["message_id"] != persisted[1]["message_id"]

    asyncio.run(scenario())


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "#246b64").save(output, format="PNG")
    return output.getvalue()


def test_image_upload_validation_uses_file_content_not_claimed_mime() -> None:
    png = _png_bytes()

    assert routes_settings._inspect_image_upload(
        png,
        allowed_mimes={"image/png"},
    ) == "image/png"

    try:
        routes_settings._inspect_image_upload(b"not-an-image", allowed_mimes={"image/png"})
    except routes_settings._ImageUploadError as exc:
        assert exc.status_code == 400
        assert "有效图片" in str(exc)
    else:
        raise AssertionError("invalid image bytes must be rejected")


def test_self_image_paths_cannot_escape_or_overwrite_existing_file(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "self_image"
    image_dir.mkdir()
    monkeypatch.setattr(routes_settings, "_SELF_IMAGE_DIR", image_dir)
    existing = image_dir / "avatar.png"
    existing.write_bytes(b"existing")

    assert routes_settings._self_image_target("../self_image_backup/secret.png") is None
    duplicate_path, duplicate = routes_settings._available_self_image_path("avatar.png", b"existing")
    replacement_path, replacement_duplicate = routes_settings._available_self_image_path("avatar.png", b"new")

    assert duplicate_path == existing
    assert duplicate is True
    assert replacement_path.name == "avatar_2.png"
    assert replacement_duplicate is False
    assert existing.read_bytes() == b"existing"


def test_self_image_filename_rejects_windows_reserved_and_invalid_names() -> None:
    assert routes_settings._safe_image_filename("头像.png") == "头像.png"
    for candidate in ("CON.png", "bad:name.png", "x" * 181 + ".png"):
        try:
            routes_settings._safe_image_filename(candidate)
        except routes_settings._ImageUploadError:
            pass
        else:
            raise AssertionError(f"unsafe filename accepted: {candidate}")
