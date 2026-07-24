from __future__ import annotations

import base64

from browser.world_prompt import render_browser_world_content
from llm.core.prompt_diagnostics import estimate_token_count


def _snapshot(*, with_image_data: bool) -> dict:
    image = {
        "kind": "image",
        "resource_ref": "br_abc123def45678901234",
        "source_url": "https://cdn.example/sample.png",
        "alt": "sample image",
        "x": 10,
        "y": 20,
        "width": 30,
        "height": 40,
        "natural_size": [300, 400],
        "source": "url",
    }
    if with_image_data:
        image.update({
            "data": base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
                "+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
            "mime_type": "image/png",
        })
    return {
        "active": True,
        "url": "https://example.com/",
        "title": "Example",
        "images": [image],
    }


def _text_content(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text") or "")
        for part in content
        if part.get("type") == "text"
    )


def test_browser_resource_metadata_never_embeds_per_image_bytes() -> None:
    content = render_browser_world_content(
        _snapshot(with_image_data=True),
        multimodal_image_limit=1,
    )

    text = _text_content(content)
    assert (
        '<image kind="image" resource_ref="br_abc123def45678901234" '
        'source_url="https://cdn.example/sample.png" alt="sample image" '
        'rect="10,20,30,40" natural_size="300,400"/>'
    ) in text
    assert '<images visible="1" embedded="0" omitted="1">' in text
    assert not isinstance(content, list)
    assert 'source="url"' not in text
    assert 'width="30"' not in text
    assert 'height="40"' not in text
    assert 'x="10"' not in text
    assert 'y="20"' not in text
    assert 'embedded="true"' not in text


def test_omitted_browser_image_keeps_rect_without_embedded_flag() -> None:
    content = render_browser_world_content(
        _snapshot(with_image_data=False),
        multimodal_image_limit=1,
    )

    text = _text_content(content)
    assert (
        '<image kind="image" resource_ref="br_abc123def45678901234" '
        'source_url="https://cdn.example/sample.png" alt="sample image" '
        'rect="10,20,30,40" natural_size="300,400"/>'
    ) in text
    assert '<images visible="1" embedded="0" omitted="1">' in text
    assert 'source="url"' not in text
    assert 'embedded="false"' not in text


def test_five_resource_projection_token_delta_is_bounded_and_has_no_image_parts() -> None:
    base_images = [{
        "kind": "image",
        "resource_ref": f"br_{index:020x}",
        "alt": f"artwork {index}",
        "x": index * 10,
        "y": index * 20,
        "width": 320,
        "height": 200,
        "natural_size": [1280, 800],
    } for index in range(5)]
    snapshot = {
        "active": True,
        "url": "https://example.com/page",
        "title": "Example",
        "images": base_images,
    }
    hidden = render_browser_world_content(snapshot, multimodal_image_limit=5)
    full = render_browser_world_content(
        {
            **snapshot,
            "images": [
                {
                    **image,
                    "source_url": (
                        f"https://cdn.example.com/images/{index}/"
                        f"original-artwork-long-name.png?token=abc{index}"
                    ),
                }
                for index, image in enumerate(base_images)
            ],
        },
        multimodal_image_limit=5,
    )

    assert isinstance(hidden, str)
    assert isinstance(full, str)
    assert '<images visible="5" embedded="0" omitted="5">' in full
    assert estimate_token_count(full) - estimate_token_count(hidden) <= 140
    assert estimate_token_count(full) <= 500


def test_viewport_reference_is_not_presented_as_sendable_image_ref() -> None:
    snapshot = _snapshot(with_image_data=False)
    snapshot["viewport"] = {
        "kind": "viewport",
        "image_ref": "viewport123",
        "mime_type": "image/png",
        "data": base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+A8AAQUBAScY42YAAAAASUVORK5CYII="
        ),
    }

    content = render_browser_world_content(snapshot, multimodal_image_limit=1)
    text = _text_content(content)

    assert '<viewport_image viewport_ref="viewport123"' in text
    assert '<viewport_image image_ref=' not in text
