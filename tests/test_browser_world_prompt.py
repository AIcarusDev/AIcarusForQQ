from __future__ import annotations

import base64

from browser.world_prompt import render_browser_world_content


def _snapshot(*, with_image_data: bool) -> dict:
    image = {
        "kind": "image",
        "image_ref": "abc123def456",
        "alt": "sample image",
        "x": 10,
        "y": 20,
        "width": 30,
        "height": 40,
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


def test_browser_image_metadata_uses_rect_without_transport_details() -> None:
    content = render_browser_world_content(
        _snapshot(with_image_data=True),
        multimodal_image_limit=1,
    )

    text = _text_content(content)
    assert (
        '<image kind="image" image_ref="abc123def456" '
        'alt="sample image" rect="10,20,30,40">'
    ) in text
    assert '<images visible="1" embedded="1">' in text
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
        '<image kind="image" image_ref="abc123def456" '
        'alt="sample image" rect="10,20,30,40"/>'
    ) in text
    assert '<images visible="1" embedded="0" omitted="1">' in text
    assert 'source="url"' not in text
    assert 'embedded="false"' not in text
