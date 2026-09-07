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


def test_image_resolver_uses_registered_identity_over_context_and_cache():
    from llm.media.image_store import register_image
    from llm.media.media_cache import cache_recent_image
    from test_sticker_collection import png
    ref = register_image(png(), "chat")["image_ref"]
    session = SimpleNamespace(context_messages=[{"images": {ref: {"data": png("blue")}}}])
    cache_recent_image(ref, {"data": png("blue")})
    assert ImageResolver(session).resolve(ref)[0]["data"] == png()


def test_view_image_ref_keeps_existing_multimodal_result_shape():
    from llm.media.image_store import register_image
    from test_sticker_collection import png
    register_image(png(), "chat", "image-2")
    result = make_handler(SimpleNamespace(context_messages=[]))("image_ref='image-2'")
    assert result == {
        "ok": True, "image_ref": "image-2", "source": "chat", "mime_type": "image/png",
        "_multimodal_parts": [{"data": png(), "mime_type": "image/png", "display_name": "chat:image-2"}],
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
    from database import init_db
    asyncio.run(init_db())
    session = SimpleNamespace(context_messages=[])

    result = asyncio.run(make_handler(session)(path="/home/agent/media/sample.jpg"))

    assert result["ok"] is True
    assert result["path"] == "/home/agent/media/sample.jpg"
    assert result["source"] == "path"
    assert result["mime_type"] == "image/png"
    assert result["_multimodal_parts"] == [
        {"data": raw, "mime_type": "image/png", "display_name": "sample.jpg"}
    ]
    assert "image_ref" in result and bool(result["image_ref"])
    assigned_ref = result["image_ref"]

    # 验证分配的 image_ref 能被 ImageResolver 正确解析为 workspace 来源
    resolver = ImageResolver(session)
    resolved = resolver.resolve(assigned_ref)
    assert resolved is not None
    assert resolved[0]["data"] == raw
    assert resolved[1] == "workspace"

    # 验证同一图片再次查看时命中去重，返回相同的 image_ref
    result_again = asyncio.run(make_handler(session)(path="/home/agent/media/sample.jpg"))
    assert result_again["ok"] is True
    assert result_again["image_ref"] == assigned_ref


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


def test_workspace_image_ref_integration_with_examine_and_sticker(monkeypatch, tmp_path) -> None:
    from database import init_db
    from tools.core import examine_image
    from platforms.qq.tools.qq_stickers import save_sticker

    asyncio.run(init_db())

    host_image = tmp_path / "chart.png"
    Image.new("RGB", (10, 10), "blue").save(host_image, format="PNG")
    raw = host_image.read_bytes()

    class WorkspaceService:
        @asynccontextmanager
        async def stage_host_file(self, path):
            yield SimpleNamespace(
                workspace_path=path,
                host_path=str(host_image),
                name="chart.png",
                size=len(raw),
            )

    monkeypatch.setattr(app_state, "workspace_service", WorkspaceService())
    monkeypatch.setattr(app_state, "main_loop", object())
    monkeypatch.setattr(view_image, "run_on_main_loop", lambda coro, _loop: coro)

    session = SimpleNamespace(context_messages=[])

    # 1. 通过 view_image 查看 Linux 本地图片，自动获得 image_ref
    view_res = asyncio.run(make_handler(session)(path="/home/agent/chart.png"))
    assert view_res["ok"] is True
    ref = view_res.get("image_ref")
    assert ref and isinstance(ref, str)

    # 2. 将该 ref 送入 examine_image 进行精查
    fake_vision_bridge = SimpleNamespace(
        enabled=True,
        examine=lambda phash, b64, mime, focus: f"Examined: {focus}",
    )
    examine_handler = examine_image.make_handler(session, fake_vision_bridge)
    examine_res = examine_handler(image_ref=ref, focus="检查蓝色区域")
    assert "error" not in examine_res
    assert examine_res["image_ref"] == ref
    assert examine_res["result"] == "Examined: 检查蓝色区域"

    # 3. 将该 ref 送入 save_sticker 进行表情包收藏
    sticker_handler = save_sticker.make_handler(lambda: session)
    sticker_res = sticker_handler(image_ref=ref, description="蓝色方块")
    assert "error" not in sticker_res
    assert sticker_res["image_ref"] == ref

