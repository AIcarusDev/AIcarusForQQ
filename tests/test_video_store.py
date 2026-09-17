from __future__ import annotations

import pytest
from pathlib import Path

from llm.media.video_store import locate_video, download_video_for_ref
from browser.world_prompt import render_browser_world_content


def test_locate_video_nonexistent(tmp_path, monkeypatch):
    import llm.media.video_store as vs
    monkeypatch.setattr(vs, "MEDIA_ROOT", tmp_path)
    assert locate_video("260917_notfound") is None


def test_locate_video_existing(tmp_path, monkeypatch):
    import llm.media.video_store as vs
    monkeypatch.setattr(vs, "MEDIA_ROOT", tmp_path)
    test_file = tmp_path / "260917_abcde.mp4"
    test_file.write_bytes(b"fake-mp4-data")

    found = locate_video("260917_abcde")
    assert found == test_file


def test_browser_world_renders_videos_element():
    snapshot = {
        "active": True,
        "url": "https://example.com/watch",
        "title": "Watch Video",
        "viewport_size":{"width": 1280, "height": 720},
        "scroll": {"y": 0, "viewport_height": 720, "page_height": 1200},
        "videos": [
            {
                "video_ref": "260917_v1234",
                "src": "https://example.com/sample.mp4",
                "poster": "https://example.com/poster.jpg",
                "x": 10,
                "y": 20,
                "width": 640,
                "height": 360,
                "paused": True,
                "current_time": 5.5,
                "duration": 120.0,
                "muted": False,
            }
        ],
        "images": [],
    }


    xml = render_browser_world_content(snapshot, multimodal_image_limit=0)
    assert isinstance(xml, str)
    assert '<videos items="1"' in xml
    assert 'video_ref="260917_v1234"' in xml
    assert 'paused="true"' in xml
    assert 'current_time="5.5"' in xml
    assert 'duration="120.0"' in xml
