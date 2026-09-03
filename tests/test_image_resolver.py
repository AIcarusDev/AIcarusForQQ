from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import app_state
from llm.core.tool_calling.schema import validate_arguments_by_declaration
from llm.media.image_resolver import ImageResolver
from PIL import Image
from tools.core import view_image
from tools.core.view_image import ViewImageArgs, make_handler


def test_image_resolver_preserves_world_visibility_order() -> None:
    image_ref = "image-1"
    session = SimpleNamespace(
        context_messages=[{"images": {image_ref: {"data": b"chat"}}}],
        is_browsing_history=lambda: True,
        chat_window_view={"top_db_id": 12, "page_size": 5},
        forward_browser_stack=[
            {
                "page_offset": 1,
                "page_size": 1,
                "nodes": [
                    {"images": {image_ref: {"data": b"hidden"}}},
                    {"images": {image_ref: {"data": b"forward"}}},
                ],
            }
        ],
    )
    resolver = ImageResolver(
        session,
        history_loader=lambda *_args: [{"images": {image_ref: {"data": b"history"}}}],
        browser_image_reader=lambda _ref: (b"browser", "image/png"),
    )

    assert resolver.resolve(image_ref) == ({"data": b"chat"}, "chat")
    session.context_messages = []
    assert resolver.resolve(image_ref) == ({"data": b"history"}, "history")
    session.is_browsing_history = lambda: False
    assert resolver.resolve(image_ref) == ({"data": b"forward"}, "forward")
    session.forward_browser_stack = []
    assert resolver.resolve(image_ref) == ({"data": b"browser", "mime": "image/png"}, "browser")


def test_view_image_ref_keeps_existing_multimodal_result_shape() -> None:
    session = SimpleNamespace(
        context_messages=[
            {
                "images": {
                    "image-2": {
                        "data": b"payload",
                        "mime_type": "image/webp",
                    }
                }
            }
        ],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )

    result = make_handler(session)("image_ref='image-2'")

    assert result == {
        "ok": True,
        "image_ref": "image-2",
        "source": "chat",
        "mime_type": "image/webp",
        "_multimodal_parts": [
            {
                "data": b"payload",
                "mime_type": "image/webp",
                "display_name": "chat:image-2",
            }
        ],
    }


def test_view_image_requires_exactly_one_source() -> None:
    declaration = view_image.TOOL_CONTRACT.declaration()

    assert ViewImageArgs.model_validate({"image_ref": "image-1"}).root.image_ref == "image-1"
    assert ViewImageArgs.model_validate({"path": "/home/agent/image.png"}).root.path == "/home/agent/image.png"
    assert validate_arguments_by_declaration({"image_ref": "image-1"}, declaration)[0]
    assert validate_arguments_by_declaration({"path": "/home/agent/image.png"}, declaration)[0]
    assert not validate_arguments_by_declaration({}, declaration)[0]
    assert not validate_arguments_by_declaration(
        {"image_ref": "image-1", "path": "/home/agent/image.png"},
        declaration,
    )[0]


def test_view_image_reads_a_valid_linux_path(monkeypatch, tmp_path) -> None:
    host_image = tmp_path / "staged-image.bin"
    Image.new("RGB", (3, 2), "red").save(host_image, format="PNG")
    raw = host_image.read_bytes()

    class WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path):
            assert path == "/home/agent/media/sample.jpg"
            yield SimpleNamespace(
                workspace_path=path,
                host_path=str(host_image),
                name="sample.jpg",
                size=len(raw),
            )

    monkeypatch.setattr(app_state, "workspace_service", WorkspaceService())
    monkeypatch.setattr(app_state, "main_loop", object())
    monkeypatch.setattr(view_image, "run_on_main_loop", lambda coro, _loop: coro)
    session = SimpleNamespace(context_messages=[])

    result = asyncio.run(make_handler(session)(path="/home/agent/media/sample.jpg"))

    assert result["ok"] is True
    assert result["path"] == "/home/agent/media/sample.jpg"
    assert result["source"] == "path"
    assert result["mime_type"] == "image/png"
    assert result["_multimodal_parts"] == [
        {"data": raw, "mime_type": "image/png", "display_name": "sample.jpg"}
    ]


def test_view_image_rejects_non_image_path_content(monkeypatch, tmp_path) -> None:
    staged_file = tmp_path / "payload.bin"
    staged_file.write_bytes(b"not an image")

    class WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path):
            yield SimpleNamespace(
                workspace_path=path,
                host_path=str(staged_file),
                name="fake.png",
                size=staged_file.stat().st_size,
            )

    monkeypatch.setattr(app_state, "workspace_service", WorkspaceService())
    monkeypatch.setattr(app_state, "main_loop", object())
    monkeypatch.setattr(view_image, "run_on_main_loop", lambda coro, _loop: coro)

    result = asyncio.run(make_handler(SimpleNamespace(context_messages=[]))(
        path="/home/agent/media/fake.png"
    ))

    assert result == {
        "ok": False,
        "status": "invalid_image",
        "path": "/home/agent/media/fake.png",
    }


def test_view_image_rejects_oversized_path_before_host_read(monkeypatch, tmp_path) -> None:
    missing_staged_file = tmp_path / "must-not-be-read.bin"

    class WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path):
            yield SimpleNamespace(
                workspace_path=path,
                host_path=str(missing_staged_file),
                name="huge.png",
                size=view_image.MAX_VIEW_IMAGE_BYTES + 1,
            )

    monkeypatch.setattr(app_state, "workspace_service", WorkspaceService())
    monkeypatch.setattr(app_state, "main_loop", object())
    monkeypatch.setattr(view_image, "run_on_main_loop", lambda coro, _loop: coro)

    result = asyncio.run(make_handler(SimpleNamespace(context_messages=[]))(
        path="/home/agent/media/huge.png"
    ))

    assert result["ok"] is False
    assert result["status"] == "image_too_large"
    assert result["limit_bytes"] == view_image.MAX_VIEW_IMAGE_BYTES
    assert not missing_staged_file.exists()
