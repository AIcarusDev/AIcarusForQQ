from __future__ import annotations

from types import SimpleNamespace

from llm.media.image_resolver import ImageResolver
from tools.core.view_image_by_ref import make_handler


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


def test_view_image_handler_keeps_existing_multimodal_result_shape() -> None:
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
