from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from browser.image_resources import (
    BrowserImageArtifactStore,
    BrowserImageError,
    BrowserImageValidationError,
)
from browser.session import BrowserSession


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (240, 160), (55, 66, 77)).save(output, format="PNG")
    return output.getvalue()


class _FakeCdp:
    def __init__(self, handlers):
        self.handlers = handlers
        self.calls: list[str] = []

    def send(self, method: str, _params=None):
        self.calls.append(method)
        handler = self.handlers.get(method)
        if isinstance(handler, Exception):
            raise handler
        if callable(handler):
            return handler(_params or {})
        if handler is None:
            raise RuntimeError(f"unexpected CDP call: {method}")
        return handler


def _registered_resource(session: BrowserSession):
    session.image_resources.observe_request("request-1", "https://cdn.example/original.png")
    session.image_resources.observe_response(
        request_id="request-1",
        frame_id="frame-1",
        url="https://cdn.example/original.png",
        mime="image/png",
        status=200,
    )
    resource = session.image_resources.register(
        source_url="https://cdn.example/original.png",
        page_url="https://example/",
        identity="main:0",
        alt="original",
        rect={"x": 0, "y": 0, "width": 240, "height": 160},
        natural_size=(240, 160),
    )
    assert resource is not None
    return resource


def test_world_snapshot_registers_resources_without_clips_or_internal_fields(monkeypatch) -> None:
    session = BrowserSession()
    session.context = SimpleNamespace()
    session.page = SimpleNamespace(url="https://example/", title=lambda: "Example")
    monkeypatch.setattr(
        session,
        "viewport_state",
        lambda: {
            "viewport": {"width": 800, "height": 600},
            "scroll": {},
            "click_targets": [],
        },
    )
    monkeypatch.setattr(
        session,
        "viewport_visuals",
        lambda: [{
            "kind": "image",
            "src": "https://cdn.example/original.png",
            "alt": "original",
            "x": 10,
            "y": 20,
            "width": 240,
            "height": 160,
            "natural_width": 1200,
            "natural_height": 800,
            "loaded": True,
        }],
    )
    monkeypatch.setattr(session, "tab_items", lambda: [])
    monkeypatch.setattr(session, "scroll_state", lambda: {})
    monkeypatch.setattr(session, "loading_state", lambda: {})
    monkeypatch.setattr(
        session,
        "capture_viewport_image",
        lambda **_kwargs: {"kind": "viewport", "image_ref": "viewport-only"},
    )

    snapshot = session.world_snapshot()

    image = snapshot["images"][0]
    assert set(image) == {
        "kind",
        "alt",
        "width",
        "height",
        "x",
        "y",
        "loaded",
        "resource_ref",
        "source_url",
        "natural_size",
    }
    assert image["natural_size"] == [1200, 800]
    assert "image_ref" not in image
    assert "data" not in image
    assert "request_id" not in repr(snapshot)
    assert "frame_id" not in repr(snapshot)
    assert not hasattr(session, "capture_viewport_clip")


def test_materialization_uses_response_then_page_cache_without_network(tmp_path) -> None:
    session = BrowserSession()
    session.image_artifacts = BrowserImageArtifactStore(tmp_path)
    resource = _registered_resource(session)
    original = _png()
    cdp = _FakeCdp({
        "Network.getResponseBody": RuntimeError("response evicted"),
        "Page.getResourceContent": {
            "content": base64.b64encode(original).decode("ascii"),
            "base64Encoded": True,
        },
    })
    session._cdp_by_request_id["request-1"] = cdp
    session._cdp_by_frame_id["frame-1"] = cdp

    result = session.materialize_resources([resource.resource_ref])

    assert result[0]["resource_ref"] == resource.resource_ref
    assert cdp.calls == ["Network.getResponseBody", "Page.getResourceContent"]
    stored = session.image_artifacts.read(result[0]["image_ref"])
    assert stored is not None and stored[0] == original


def test_materialization_uses_browser_network_last_and_marks_identity_risk(tmp_path) -> None:
    session = BrowserSession()
    session.image_artifacts = BrowserImageArtifactStore(tmp_path)
    resource = _registered_resource(session)
    original = _png()
    reads = iter([
        {"data": base64.b64encode(original).decode("ascii"), "base64Encoded": True, "eof": True},
    ])
    cdp = _FakeCdp({
        "Network.getResponseBody": RuntimeError("evicted"),
        "Page.getResourceContent": RuntimeError("resource unavailable"),
        "Network.loadNetworkResource": {
            "resource": {
                "success": True,
                "httpStatusCode": 200,
                "headers": {"Content-Type": "image/png"},
                "stream": "stream-1",
            }
        },
        "IO.read": lambda _params: next(reads),
        "IO.close": {},
    })
    session._cdp_by_request_id["request-1"] = cdp
    session._cdp_by_frame_id["frame-1"] = cdp

    result = session.materialize_resources([resource.resource_ref])

    assert result[0]["confirmation_reasons"] == ["resource_identity_unproven"]
    assert cdp.calls[:3] == [
        "Network.getResponseBody",
        "Page.getResourceContent",
        "Network.loadNetworkResource",
    ]


def test_materialization_failure_creates_no_artifact_and_never_screenshots(tmp_path) -> None:
    session = BrowserSession()
    session.image_artifacts = BrowserImageArtifactStore(tmp_path)
    resource = _registered_resource(session)
    cdp = _FakeCdp({
        "Network.getResponseBody": RuntimeError("evicted"),
        "Page.getResourceContent": RuntimeError("missing"),
        "Network.loadNetworkResource": RuntimeError("offline"),
    })
    session._cdp_by_request_id["request-1"] = cdp
    session._cdp_by_frame_id["frame-1"] = cdp

    with pytest.raises(BrowserImageError, match="original browser image unavailable"):
        session.materialize_resources([resource.resource_ref])

    assert list(tmp_path.iterdir()) == []


def test_materialization_hard_blocks_invalid_bytes_in_all_modes(tmp_path) -> None:
    session = BrowserSession()
    session.image_artifacts = BrowserImageArtifactStore(tmp_path)
    resource = _registered_resource(session)
    cdp = _FakeCdp({
        "Network.getResponseBody": {
            "body": base64.b64encode(b"<script>bad</script>").decode("ascii"),
            "base64Encoded": True,
        },
    })
    session._cdp_by_request_id["request-1"] = cdp

    with pytest.raises(BrowserImageValidationError):
        session.materialize_resources([resource.resource_ref])
    assert list(tmp_path.iterdir()) == []


def test_materialization_rejects_more_than_four_before_fetch(tmp_path) -> None:
    session = BrowserSession()
    session.image_artifacts = BrowserImageArtifactStore(tmp_path)
    refs = []
    for index in range(5):
        resource = session.image_resources.register(
            source_url=f"https://cdn.example/{index}.png",
            page_url="https://example/",
            identity=str(index),
            alt="",
            rect={"x": 0, "y": 0, "width": 100, "height": 100},
            natural_size=(100, 100),
        )
        refs.append(resource.resource_ref)

    with pytest.raises(BrowserImageError, match="at most 4"):
        session.materialize_resources(refs)
