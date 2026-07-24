from __future__ import annotations

import io

import pytest
from PIL import Image

from browser.image_resources import (
    BrowserImageArtifactStore,
    BrowserImageResourceRegistry,
    BrowserImageValidationError,
)


def _png(size: tuple[int, int] = (320, 200)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (23, 45, 67)).save(output, format="PNG")
    return output.getvalue()


def _resource(registry: BrowserImageResourceRegistry):
    registry.observe_request("request-secret", "https://cdn.example/art.png?token=secret")
    registry.observe_response(
        request_id="request-secret",
        frame_id="frame-secret",
        url="https://cdn.example/art.png?token=secret",
        mime="image/png",
        status=200,
    )
    return registry.register(
        source_url="https://cdn.example/art.png?token=secret",
        page_url="https://example/",
        identity="main:0",
        alt="artwork",
        rect={"x": 10, "y": 20, "width": 320, "height": 200},
        natural_size=(320, 200),
    )


def test_model_projection_exposes_only_model_contract_fields() -> None:
    registry = BrowserImageResourceRegistry()
    resource = _resource(registry)
    assert resource is not None

    projection = resource.model_projection("full")

    assert projection == {
        "resource_ref": resource.resource_ref,
        "source_url": "https://cdn.example/art.png?token=%3Credacted%3E",
        "alt": "artwork",
        "rect": {"x": 10, "y": 20, "width": 320, "height": 200},
        "natural_size": [320, 200],
    }
    serialized = repr(projection)
    assert "request-secret" not in serialized
    assert "frame-secret" not in serialized
    assert "https://example/" not in serialized


def test_source_url_projection_modes_are_explicit() -> None:
    registry = BrowserImageResourceRegistry()
    resource = _resource(registry)
    assert resource is not None

    assert "source_url" not in resource.model_projection("hidden")
    assert resource.model_projection("sanitized")["source_url"] == "https://cdn.example/art.png"
    full = resource.model_projection("full")["source_url"]
    assert full.endswith("?token=%3Credacted%3E")
    assert "secret" not in full


def test_artifact_store_persists_only_validated_immutable_original(tmp_path) -> None:
    registry = BrowserImageResourceRegistry()
    resource = _resource(registry)
    assert resource is not None
    store = BrowserImageArtifactStore(tmp_path)
    original = _png()

    artifact = store.persist(
        original,
        resource=resource,
        strategy="response_body",
        declared_mime="image/png",
    )

    assert artifact.image_ref.startswith("img_")
    assert artifact.confirmation_reasons == ()
    assert store.read(artifact.image_ref)[:2] == (original, "image/png")
    assert sorted(path.suffix for path in tmp_path.iterdir()) == [".json", ".png"]


def test_artifact_store_rejects_non_image_and_detects_tampering(tmp_path) -> None:
    registry = BrowserImageResourceRegistry()
    resource = _resource(registry)
    assert resource is not None
    store = BrowserImageArtifactStore(tmp_path)

    with pytest.raises(BrowserImageValidationError):
        store.persist(
            b"<html>not an image</html>",
            resource=resource,
            strategy="response_body",
            declared_mime="image/png",
        )
    assert list(tmp_path.glob("*")) == []

    artifact = store.persist(
        _png(),
        resource=resource,
        strategy="response_body",
        declared_mime="image/png",
    )
    data_path = next(path for path in tmp_path.iterdir() if path.suffix == ".png")
    data_path.write_bytes(_png((321, 200)))
    assert store.read(artifact.image_ref) is None


def test_explicit_high_risk_reasons_replace_opaque_score(tmp_path) -> None:
    registry = BrowserImageResourceRegistry()
    resource = registry.register(
        source_url="http://cdn.example/preview",
        page_url="https://example/",
        identity="main:0",
        alt="preview",
        rect={"x": 0, "y": 0, "width": 48, "height": 48},
        natural_size=(100, 100),
    )
    assert resource is not None

    artifact = BrowserImageArtifactStore(tmp_path).persist(
        _png((320, 200)),
        resource=resource,
        strategy="browser_network",
        declared_mime="image/jpeg",
        final_url="https://cdn.example/original.png",
    )

    assert set(artifact.confirmation_reasons) == {
        "resource_identity_unproven",
        "redirect_target_changed",
        "source_url_upgraded",
        "mime_or_content_form_changed",
        "aspect_ratio_changed",
        "very_small_preview",
        "visible_crop_or_partial_preview",
    }
