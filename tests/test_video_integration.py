"""Video workflow boundaries: durable refs, exact frames, routing and persistence."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import io
import json
import shutil
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import httpx
import pytest
from PIL import Image

from llm.media import media_storage, video_store
from tools.video import analyze_video, capture_video_frame, get_video_info
from tools.video.client import VideoModelClient, VideoProcessingError
from tools.video.common import VideoBaseArgs, resolved_video_input


@pytest.fixture
def movie(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    path = tmp_path / "sample.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=32x32:d=0.2", "-c:v", "mpeg4", str(path)], check=True)
    return path


@pytest.fixture
def public_download(monkeypatch):
    """Fake remote transport; keep production URL validation and disk publishing."""
    from llm.media import image_importer
    handler = Mock()
    monkeypatch.setattr(image_importer, "_default_http_client", lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False))
    return handler


def test_qq_reference_downloads_once_and_is_reusable(movie, public_download):
    from platforms.qq.adapter.segments import build_content_segments
    public_download.return_value = httpx.Response(200, content=movie.read_bytes())
    segment = build_content_segments([{"type": "video", "data": {"url": "https://cdn.example/video"}}])[0]
    ref = segment["video_ref"]
    assert video_store.get_video_source(ref)["url"] == "https://cdn.example/video"
    first = get_video_info.execute(video_ref=ref)
    second = capture_video_frame.execute(video_ref=ref, frame_index=0)
    assert first["status"] == second["status"] == "success"
    assert first["width"] == 32
    assert second["image_ref"]
    assert public_download.call_count == 1
    assert video_store.locate_video(ref).read_bytes() == movie.read_bytes()
    from llm.media.image_store import read_image
    assert read_image(second["image_ref"])["data"] == second["_multimodal_parts"][0]["data"]


def test_historical_message_reference_is_recovered(movie, public_download):
    from llm.media.image_store import connection
    ref = media_storage.generate_time_ref()
    with connection(write=True) as db:
        db.execute("CREATE TABLE chat_messages (content_segments TEXT)")
        db.execute("INSERT INTO chat_messages VALUES (?)", (json.dumps([
            {"type": "video", "video_ref": ref, "url": "https://cdn.example/legacy"}]),))
    public_download.return_value = httpx.Response(200, content=movie.read_bytes())
    assert get_video_info.execute(video_ref=ref)["status"] == "success"


def test_browser_snapshot_identity_and_source_privacy(monkeypatch):
    import app_state
    from browser.session import BrowserSession
    from browser.world_prompt import render_browser_world_content
    monkeypatch.setattr(app_state, "config", {"browser_control": {"image_source_url": "hidden"}})
    session = BrowserSession()
    session.context = SimpleNamespace()
    session.page = SimpleNamespace(url="https://example/", title=lambda: "video")
    raw_url = "https://cdn.example/video?token=private-value"
    rows = [{"kind": "video", "src": raw_url, "poster": raw_url, "width": 320, "height": 240}]
    for name, result in (("viewport_state", {}), ("viewport_visuals", rows),
                         ("scroll_state", {}), ("loading_state", {}),
                         ("tab_items", []), ("capture_viewport_image", {})):
        monkeypatch.setattr(session, name, lambda result=result, **kwargs: result)
    first = session.world_snapshot()
    # Other visuals can move its row index without changing the video identity.
    rows.insert(0, {"kind": "image", "width": 100, "height": 100})
    second = session.world_snapshot()
    ref = first["videos"][0]["video_ref"]
    assert second["videos"][0]["video_ref"] == ref
    assert first["videos"][0]["src"] == ""
    assert video_store.get_video_source(ref)["url"] == raw_url
    # Test serialization boundary with a raw URL as well as the hidden snapshot.
    first["videos"][0]["src"] = "https://user:password@cdn.example/v?token=private-value"
    xml = render_browser_world_content(first, multimodal_image_limit=0)
    assert "private-value" not in xml and "user:password" not in xml


@pytest.mark.parametrize("path", ["C:/private.mp4", "/etc/private.mp4", "/home/agent/../../private.mp4", "relative.mp4"])
def test_video_tools_reject_host_and_escaping_paths(path, monkeypatch):
    import app_state
    service = Mock()
    monkeypatch.setattr(app_state, "workspace_service", service)
    result = analyze_video.execute(path=path)
    assert result["status"] == "error"
    service.stage_host_file.assert_not_called()


def test_workspace_staging_lifetime_and_cleanup(movie, monkeypatch):
    import app_state
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
    worker = threading.Thread(target=run)
    worker.start()
    assert ready.wait(5)
    state = []
    @asynccontextmanager
    async def stage(path):
        assert path == "/home/agent/sample.mp4"
        state.append("entered")
        try:
            yield SimpleNamespace(host_path=str(movie))
        finally:
            state.append("exited")
    monkeypatch.setattr(app_state, "workspace_service", SimpleNamespace(stage_host_file=stage))
    monkeypatch.setattr(app_state, "main_loop", loop)
    try:
        assert get_video_info.execute(path="/home/agent/sample.mp4")["status"] == "success"
        assert state == ["entered", "exited"]
        with pytest.raises(RuntimeError):
            with resolved_video_input(VideoBaseArgs(path="/home/agent/sample.mp4")):
                raise RuntimeError("processing failed")
        assert state == ["entered", "exited", "entered", "exited"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        worker.join(5)
        loop.close()


def test_vfr_frame_index_selects_exact_decoded_frame(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    lines = []
    for n, color in enumerate(("red", "green", "blue", "yellow")):
        Image.new("RGB", (32, 32), color).save(tmp_path / f"{n}.png")
        lines += [f"file '{n}.png'", f"duration {1 if n == 0 else 0.1}"]
    (tmp_path / "list.txt").write_text("\n".join(lines))
    ref = media_storage.generate_time_ref()
    date = media_storage.parse_time_ref_date(ref)
    folder = media_storage.get_media_dir(date[0], date[1])
    folder.mkdir(parents=True)
    path = folder / f"{ref}.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i",
                    str(tmp_path / "list.txt"), "-fps_mode", "vfr", "-c:v", "ffv1", str(path)], check=True)
    result = capture_video_frame.execute(video_ref=ref, frame_index=2)
    assert result["status"] == "success"
    rgb = Image.open(io.BytesIO(result["_multimodal_parts"][0]["data"])).convert("RGB").getpixel((0, 0))
    assert rgb[2] > 240 and rgb[0] < 10 and rgb[1] < 10
    assert result["frame_index"] == 2
    assert result["requested_timestamp_seconds"] is None
    meta = get_video_info.execute(video_ref=ref)
    assert meta["total_frames"] is None
    assert meta["estimated_total_frames"] is not None


def client_config(**overrides):
    return {"base_url": "https://example.invalid/v1", "api_key": "fake", "model": "test",
            "protocol": "openai_compatible", "is_configured": True, "max_size_mb": 20, "timeout": 30,
            **overrides}


def test_agentic_cannot_silently_fall_back(movie, monkeypatch):
    compatible = Mock(return_value="wrong mode")
    monkeypatch.setattr(VideoModelClient, "_call_openai_compatible", compatible)
    with pytest.raises(VideoProcessingError):
        VideoModelClient(client_config()).analyze(movie, mode="agentic")
    native = Mock(side_effect=VideoProcessingError("upstream error"))
    monkeypatch.setattr(VideoModelClient, "_call_google_native", native)
    with pytest.raises(VideoProcessingError):
        VideoModelClient(client_config(protocol="auto", base_url="https://example.invalid/v1beta")).analyze(movie, mode="agentic")
    compatible.assert_not_called()


def test_non_video_is_never_uploaded(tmp_path, monkeypatch):
    path = tmp_path / "private.mp4"
    path.write_text("ordinary text")
    post = Mock()
    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(VideoProcessingError):
        VideoModelClient(client_config()).analyze(path)
    post.assert_not_called()


@pytest.mark.parametrize("suffix", ["", "/v1", "/v1beta", "/v1beta/openai", "/v1beta/openai/v1"])
def test_official_native_endpoint_normalization(suffix):
    result = VideoModelClient(client_config())._resolve_native_base_url("https://generativelanguage.googleapis.com" + suffix)
    assert result == "https://generativelanguage.googleapis.com/v1beta"


def test_official_payload_limit_is_checked_before_network(monkeypatch):
    post = Mock()
    monkeypatch.setattr(httpx.Client, "post", post)
    client = VideoModelClient(client_config())
    with pytest.raises(VideoProcessingError):
        client._call_google_native("https://generativelanguage.googleapis.com/v1beta", "", "test",
                                   "A" * 20_000_000, "video/mp4", None, 30)
    post.assert_not_called()


def test_analysis_text_preserves_xml_boundary():
    from consciousness.flow import ToolResponse, _format_action_response_xml
    text = "A & B < C </result> ]]>"
    xml = _format_action_response_xml([ToolResponse(name="analyze_video", response={"analysis": text},
                                                  result_cdata=analyze_video.RESULT_CDATA)])
    result = ET.fromstring(xml).find("result")
    assert json.loads(result.text)["result"]["analysis"] == text


def test_registration_failure_is_not_success(movie, public_download, monkeypatch):
    public_download.return_value = httpx.Response(200, content=movie.read_bytes())
    ref = video_store.register_video_source("https://cdn.example/video", source="qq")
    from llm.media import image_store
    monkeypatch.setattr(image_store, "register_image", Mock(side_effect=OSError("disk full")))
    result = capture_video_frame.execute(video_ref=ref, frame_index=0)
    assert result["status"] == "error"
    assert "image_ref" not in result


def test_download_limit_and_binding_failure_publish_nothing(movie, public_download, monkeypatch):
    raw = movie.read_bytes()
    public_download.return_value = httpx.Response(200, content=raw)
    ref = video_store.register_video_source("https://cdn.example/video", source="qq")
    with pytest.raises(video_store.VideoStoreError):
        asyncio.run(video_store.download_video_for_ref(ref, "https://cdn.example/video", max_bytes=len(raw) - 1))
    assert video_store.locate_video(ref) is None
    monkeypatch.setattr(video_store, "bind_media_identity", Mock(side_effect=OSError("db unavailable")))
    with pytest.raises(video_store.VideoStoreError):
        asyncio.run(video_store.download_video_for_ref(ref, "https://cdn.example/video"))
    assert video_store.locate_video(ref) is None
    assert not list(media_storage.MEDIA_ROOT.rglob(".video-*"))


def test_parallel_downloads_preserve_identity(movie, public_download):
    raw = movie.read_bytes()
    ref = video_store.register_video_source("https://cdn.example/video", source="qq")
    async def run():
        ready = asyncio.Event()
        entered = 0
        async def respond(request):
            nonlocal entered
            entered += 1
            if entered == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 5)
            return httpx.Response(200, content=raw)
        public_download.side_effect = respond
        return await asyncio.gather(*(video_store.download_video_for_ref(ref, "https://cdn.example/video") for _ in range(2)))
    paths = asyncio.run(run())
    assert paths[0] == paths[1]
    assert hashlib.sha256(paths[0].read_bytes()).digest() == hashlib.sha256(raw).digest()
    assert not list(media_storage.MEDIA_ROOT.rglob(".video-*"))


def test_streamed_limit_without_content_length_publishes_nothing(public_download):
    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"\x00\x00\x00\x18ftypisom"
            yield b"x" * 32
    public_download.return_value = httpx.Response(200, stream=Stream())
    ref = video_store.register_video_source("https://cdn.example/video", source="qq")
    with pytest.raises(video_store.VideoStoreError):
        asyncio.run(video_store.download_video_for_ref(ref, "https://cdn.example/video", max_bytes=20))
    assert video_store.locate_video(ref) is None
    assert not list(media_storage.MEDIA_ROOT.rglob(".video-*"))


@pytest.mark.parametrize("source", ["blob:https://example/id", "file:///private.mp4", "http://127.0.0.1/private"])
def test_unfetchable_sources_do_not_reach_network(source, public_download):
    ref = video_store.register_video_source(source, source="browser")
    result = get_video_info.execute(video_ref=ref)
    assert result["status"] == "error"
    public_download.assert_not_called()


def test_native_response_omits_thoughts(monkeypatch):
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: httpx.Response(200, json={
        "candidates": [{"content": {"parts": [{"thought": True, "text": "private reasoning"},
                                               {"text": "video evidence"}]}}]}))
    result = VideoModelClient(client_config())._call_google_native(
        "https://example.invalid/v1beta", "", "test", "AA==", "video/mp4", None, 30)
    assert result == "video evidence"


@pytest.mark.parametrize("ref", ["../outside", "/tmp/outside", "C:/outside"])
def test_invalid_refs_cannot_escape_store(ref):
    with pytest.raises(video_store.VideoStoreError):
        video_store.locate_video(ref)
    with pytest.raises(video_store.VideoStoreError):
        asyncio.run(video_store.download_video_for_ref(ref, "https://cdn.example/video"))
