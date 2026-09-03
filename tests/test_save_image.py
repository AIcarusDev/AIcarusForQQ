from __future__ import annotations

import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from llm.core.tool_calling.schema import validate_arguments_by_declaration
from llm.media import image_importer
from llm.media.image_importer import (
    ImageImportError,
    ImageImporter,
    MAX_SAVED_IMAGE_BYTES,
    download_image_bytes,
)
from PIL import Image
from tools import build_tools
from tools.core.save_image import TOOL_CONTRACT


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 3), "blue").save(output, format="PNG")
    return output.getvalue()


class _ImportSession:
    def __init__(self, *, result: dict | None = None) -> None:
        self.data = bytearray()
        self.result = result
        self.aborted = False

    async def write(self, chunk: bytes) -> None:
        self.data.extend(chunk)

    async def finish(self) -> dict:
        if self.result is not None:
            return dict(self.result)
        return {"ok": True, "size_bytes": len(self.data)}

    async def abort(self) -> None:
        self.aborted = True


class _FailingImportSession(_ImportSession):
    async def write(self, chunk: bytes) -> None:
        await super().write(chunk)
        raise OSError("sentinel write failure")


class _Workspace:
    def __init__(self, *, result: dict | None = None) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []
        self.session: _ImportSession | None = None

    async def begin_file_import(self, path: str, expected_size: int) -> _ImportSession:
        self.calls.append((path, expected_size))
        self.session = _ImportSession(result=self.result)
        return self.session


class _FailingWorkspace(_Workspace):
    async def begin_file_import(self, path: str, expected_size: int) -> _ImportSession:
        self.calls.append((path, expected_size))
        self.session = _FailingImportSession()
        return self.session


class _Response:
    def __init__(self, status_code: int, *, headers: dict | None = None, chunks=()) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = list(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def aiter_raw(self, _chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def stream(self, method: str, url: str) -> _Response:
        assert method == "GET"
        self.urls.append(url)
        return self.responses.pop(0)


def test_save_image_schema_enforces_exactly_one_source() -> None:
    declaration = TOOL_CONTRACT.declaration()

    assert validate_arguments_by_declaration(
        {"image_ref": "img_1", "path": "/home/agent/images/a.png"},
        declaration,
    )[0]
    assert validate_arguments_by_declaration(
        {"url": "https://example.test/a.png", "path": "/home/agent/images/a.png"},
        declaration,
    )[0]
    assert not validate_arguments_by_declaration(
        {
            "image_ref": "img_1",
            "url": "https://example.test/a.png",
            "path": "/home/agent/images/a.png",
        },
        declaration,
    )[0]
    assert not validate_arguments_by_declaration(
        {"path": "/home/agent/images/a.png"},
        declaration,
    )[0]
    assert not validate_arguments_by_declaration(
        {"url": "file:///tmp/a.png", "path": "/home/agent/images/a.png"},
        declaration,
    )[0]


def test_save_image_is_core_resident_without_workspace_context() -> None:
    session = SimpleNamespace(context_messages=[])

    collection = build_tools({"vision": True}, session=session)

    assert "core.save_image" in collection.active_names()


def test_image_importer_saves_base64_ref_through_atomic_workspace_import() -> None:
    raw = _png_bytes()
    workspace = _Workspace()
    session = SimpleNamespace(
        context_messages=[
            {"images": {"img_1": {"base64": base64.b64encode(raw).decode("ascii"), "mime": "image/jpeg"}}}
        ],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )

    saved = asyncio.run(
        ImageImporter(session, workspace).save(
            image_ref="img_1",
            path="/home/agent/images/ref.png",
        )
    )

    assert saved.path == "/home/agent/images/ref.png"
    assert saved.mime_type == "image/png"
    assert saved.size_bytes == len(raw)
    assert workspace.calls == [(saved.path, len(raw))]
    assert workspace.session is not None and bytes(workspace.session.data) == raw


def test_image_importer_rejects_invalid_content_before_opening_import() -> None:
    workspace = _Workspace()
    session = SimpleNamespace(
        context_messages=[{"images": {"img_1": {"data": b"not-an-image"}}}],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )

    with pytest.raises(ImageImportError, match="图片内容无效") as raised:
        asyncio.run(
            ImageImporter(session, workspace).save(
                image_ref="img_1",
                path="/home/agent/images/ref.png",
            )
        )

    assert raised.value.code == "invalid_image"
    assert workspace.calls == []


def test_image_importer_rejects_extension_mismatch_without_writing() -> None:
    raw = _png_bytes()
    workspace = _Workspace()
    session = SimpleNamespace(
        context_messages=[{"images": {"img_1": {"data": raw}}}],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )

    with pytest.raises(ImageImportError) as raised:
        asyncio.run(
            ImageImporter(session, workspace).save(
                image_ref="img_1",
                path="/home/agent/images/ref.jpg",
            )
        )

    assert raised.value.code == "extension_mismatch"
    assert raised.value.details["expected_extensions"] == [".png"]
    assert workspace.calls == []


def test_image_importer_preserves_no_overwrite_result() -> None:
    raw = _png_bytes()
    workspace = _Workspace(result={"ok": False, "code": "already_exists"})
    session = SimpleNamespace(
        context_messages=[{"images": {"img_1": {"data": raw}}}],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )

    with pytest.raises(ImageImportError) as raised:
        asyncio.run(
            ImageImporter(session, workspace).save(
                image_ref="img_1",
                path="/home/agent/images/ref.png",
            )
        )

    assert raised.value.code == "already_exists"
    assert raised.value.retryable is False


def test_image_importer_aborts_partial_workspace_import_on_write_failure() -> None:
    raw = _png_bytes()
    workspace = _FailingWorkspace()
    session = SimpleNamespace(
        context_messages=[{"images": {"img_1": {"data": raw}}}],
        is_browsing_history=lambda: False,
        forward_browser_stack=[],
    )

    with pytest.raises(OSError, match="sentinel write failure"):
        asyncio.run(
            ImageImporter(session, workspace).save(
                image_ref="img_1",
                path="/home/agent/images/ref.png",
            )
        )

    assert workspace.session is not None and workspace.session.aborted is True


def test_download_image_bytes_follows_redirect_and_checks_declared_size() -> None:
    raw = _png_bytes()
    client = _Client(
        [
            _Response(302, headers={"location": "/final.png"}),
            _Response(200, headers={"content-length": str(len(raw))}, chunks=[raw[:9], raw[9:]]),
        ]
    )

    result = asyncio.run(
        download_image_bytes(
            "https://example.test/start",
            http_client_factory=lambda: client,
        )
    )

    assert result == raw
    assert client.urls == ["https://example.test/start", "https://example.test/final.png"]


def test_image_importer_saves_downloaded_url_bytes() -> None:
    raw = _png_bytes()
    client = _Client([_Response(200, headers={"content-length": str(len(raw))}, chunks=[raw])])
    workspace = _Workspace()

    saved = asyncio.run(
        ImageImporter(
            SimpleNamespace(context_messages=[]),
            workspace,
            http_client_factory=lambda: client,
        ).save(
            url="https://example.test/image",
            path="/home/agent/images/url.png",
        )
    )

    assert saved.mime_type == "image/png"
    assert workspace.session is not None and bytes(workspace.session.data) == raw


def test_default_http_client_cannot_bypass_isolated_gateway(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    monkeypatch.setattr(
        image_importer,
        "get_browser_gateway",
        lambda: SimpleNamespace(proxy_url="http://127.0.0.1:43210"),
    )
    monkeypatch.setattr(
        image_importer.httpx,
        "AsyncClient",
        lambda **kwargs: captured.update(kwargs) or sentinel,
    )

    assert image_importer._default_http_client() is sentinel
    assert captured["proxy"] == "http://127.0.0.1:43210"
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False


def test_download_image_bytes_rejects_redirect_to_private_address() -> None:
    client = _Client([_Response(302, headers={"location": "http://127.0.0.1/a.png"})])

    with pytest.raises(ImageImportError) as raised:
        asyncio.run(
            download_image_bytes(
                "https://example.test/start",
                http_client_factory=lambda: client,
            )
        )

    assert raised.value.code == "unsafe_url"
    assert client.urls == ["https://example.test/start"]


def test_download_image_bytes_rejects_oversized_response_before_body() -> None:
    client = _Client(
        [_Response(200, headers={"content-length": str(MAX_SAVED_IMAGE_BYTES + 1)}, chunks=[b"unused"])]
    )

    with pytest.raises(ImageImportError) as raised:
        asyncio.run(
            download_image_bytes(
                "https://example.test/a.png",
                http_client_factory=lambda: client,
            )
        )

    assert raised.value.code == "image_too_large"
