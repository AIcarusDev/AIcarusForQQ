"""test_video_tool.py — video 命名空间及相关工具（analyze_video, get_video_info, capture_video_frame）回归测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.video.analyze_video import (
    DEFAULT_AGENTIC_PROMPT,
    DEFAULT_STATIC_PROMPT,
    AnalyzeVideoArgs,
    _build_final_prompt,
    execute as execute_analyze,
)
from tools.video.capture_video_frame import (
    CaptureVideoFrameArgs,
    execute as execute_capture_frame,
)
from tools.video.client import (
    DEFAULT_MAX_SIZE_MB,
    VideoModelClient,
    VideoProcessingError,
    detect_video_mime,
    get_video_config,
)
from tools.video.common import (
    VideoBaseArgs,
    extract_metadata,
    format_duration,
    parse_fraction,
    parse_timestamp_to_seconds,
)
from tools.video.get_video_info import (
    GetVideoInfoArgs,
    execute as execute_get_info,
)


def test_detect_video_mime():
    assert detect_video_mime(Path("sample.mp4")) == "video/mp4"
    assert detect_video_mime(Path("sample.webm")) == "video/webm"
    assert detect_video_mime(Path("sample.mov")) == "video/quicktime"
    assert detect_video_mime(Path("sample.mkv")) == "video/x-matroska"
    assert detect_video_mime(Path("sample.unknown")) == "video/mp4"


def test_prompt_builder():
    # 默认无提问时返回 None，交由底层 System Instruction 处理
    assert _build_final_prompt(None, "static") is None
    assert _build_final_prompt("", "agentic") is None

    # 用户/Agent 传入自定义 prompt 时保持原样定向注入
    query = "视频里有几个人？"
    assert _build_final_prompt(query, "static") == query
    assert _build_final_prompt(query, "agentic") == query


def test_validate_args_credential_required():
    with pytest.raises(Exception):
        AnalyzeVideoArgs.model_validate({})

    args = AnalyzeVideoArgs.model_validate({"path": "foo.mp4"})
    assert args.path == "foo.mp4"
    assert args.mode == "static"


def test_default_max_size_limit():
    assert DEFAULT_MAX_SIZE_MB == 128
    with patch("tools.video.client._load_raw_config", return_value={}):
        cfg = get_video_config()
        assert cfg["max_size_mb"] == 128.0


def test_size_limit_interception():
    client = VideoModelClient({"max_size_mb": 1, "base_url": "http://example", "model": "test"})
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.seek(2 * 1024 * 1024)
        f.write(b"0")
        f_path = Path(f.name)

    try:
        with pytest.raises(VideoProcessingError) as exc_info:
            client.validate_file_size(f_path)
        assert "超出允许的硬限制" in str(exc_info.value)
    finally:
        f_path.unlink(missing_ok=True)


def test_execute_missing_credentials():
    result = execute_analyze()
    assert "error" in result
    assert "工具参数不符合定义" in result["error"]


def test_execute_non_existent_file():
    result = execute_analyze(path="non_existent_video_path_123.mp4")
    assert result["status"] == "error"
    assert "不存在" in result["error"]


def test_execute_mocked_success():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake video data")
        f_path = Path(f.name)

    try:
        with patch.object(VideoModelClient, "analyze", return_value="模拟视频分析结果") as mock_analyze:
            res = execute_analyze(path=str(f_path), mode="agentic")
            assert res["status"] == "success"
            assert res["mode"] == "agentic"
            assert res["analysis"] == "模拟视频分析结果"
            mock_analyze.assert_called_once()
            _, kwargs = mock_analyze.call_args
            assert kwargs.get("mode") == "agentic"
    finally:
        f_path.unlink(missing_ok=True)


def test_google_native_media_processing_payload():
    client = VideoModelClient({
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-3.8-flash",
        "api_key": "AIzaFakeKey",
    })

    captured_payloads = []

    def mock_post(url, headers=None, params=None, json=None):
        captured_payloads.append(json)
        class DummyResp:
            status_code = 200
            def json(self):
                return {"candidates": [{"content": {"parts": [{"text": "分析结果"}]}}]}
        return DummyResp()

    with patch("httpx.Client.post", side_effect=mock_post):
        # 1. agentic 模式 + 定向提问
        client._call_google_native(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="AIzaFakeKey",
            model="gemini-3.8-flash",
            b64_data="dummy_b64",
            mime_type="video/mp4",
            prompt="测试问题",
            timeout=30.0,
            mode="agentic",
        )
        payload1 = captured_payloads[-1]
        parts1 = payload1["contents"][0]["parts"]
        assert parts1[0]["mediaProcessing"] == "AGENTIC"
        assert parts1[0]["inlineData"]["mimeType"] == "video/mp4"
        assert parts1[1]["text"] == "测试问题"
        assert "You are a multimodal video analysis assistant" in payload1["systemInstruction"]["parts"][0]["text"]

        # 2. static 模式且无 prompt
        client._call_google_native(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="AIzaFakeKey",
            model="gemini-3.8-flash",
            b64_data="dummy_b64",
            mime_type="video/mp4",
            prompt=None,
            timeout=30.0,
            mode="static",
        )
        payload2 = captured_payloads[-1]
        parts2 = payload2["contents"][0]["parts"]
        assert parts2[0]["mediaProcessing"] == "STATIC"
        assert len(parts2) == 1
        assert "You are a multimodal video analysis assistant" in payload2["systemInstruction"]["parts"][0]["text"]


def test_openai_compatible_system_instruction_payload():
    client = VideoModelClient({
        "base_url": "https://api.openai-proxy.com/v1",
        "model": "gemini-3.8-flash",
        "api_key": "sk-fake",
    })

    captured_payloads = []

    def mock_post(url, headers=None, json=None):
        captured_payloads.append(json)
        class DummyResp:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "分析结果"}}]}
        return DummyResp()

    with patch("httpx.Client.post", side_effect=mock_post):
        # 1. 传定向提问
        client._call_openai_compatible(
            base_url="https://api.openai-proxy.com/v1",
            api_key="sk-fake",
            model="gemini-3.8-flash",
            b64_data="dummy_b64",
            mime_type="video/mp4",
            prompt="某人在哪？",
            timeout=30.0,
        )
        payload1 = captured_payloads[-1]
        messages1 = payload1["messages"]
        assert messages1[0]["role"] == "system"
        assert "You are a multimodal video analysis assistant" in messages1[0]["content"]
        assert messages1[1]["role"] == "user"
        assert messages1[1]["content"][0]["type"] == "video_url"
        assert messages1[1]["content"][1]["type"] == "text"
        assert messages1[1]["content"][1]["text"] == "某人在哪？"

        # 2. 无提问
        client._call_openai_compatible(
            base_url="https://api.openai-proxy.com/v1",
            api_key="sk-fake",
            model="gemini-3.8-flash",
            b64_data="dummy_b64",
            mime_type="video/mp4",
            prompt=None,
            timeout=30.0,
        )
        payload2 = captured_payloads[-1]
        messages2 = payload2["messages"]
        assert messages2[0]["role"] == "system"
        assert len(messages2[1]["content"]) == 1
        assert messages2[1]["content"][0]["type"] == "video_url"


# --- 扩展工具与 common 功能测试 ---


def test_common_helpers():
    assert format_duration(65.5) == "01:05.500"
    assert format_duration(3665.123) == "01:01:05.123"
    assert format_duration(0) == "00:00.000"

    assert parse_fraction("30/1") == 30.0
    assert parse_fraction("24000/1001") == pytest.approx(23.976, 0.001)
    assert parse_fraction("25") == 25.0
    assert parse_fraction("") == 0.0

    assert parse_timestamp_to_seconds(15.5) == 15.5
    assert parse_timestamp_to_seconds("15.5") == 15.5
    assert parse_timestamp_to_seconds("01:10") == 70.0
    assert parse_timestamp_to_seconds("01:02:03") == 3723.0


def test_extract_metadata_parsing():
    probe_sample = {
        "format": {
            "duration": "12.345",
            "format_name": "mov,mp4",
            "size": "1048576",
            "bit_rate": "800000",
        },
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "pix_fmt": "yuv420p",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
            },
        ],
    }
    meta = extract_metadata(probe_sample, Path("dummy.mp4"))
    assert meta["duration_seconds"] == 12.345
    assert meta["resolution"] == "1920x1080"
    assert meta["fps"] == 30.0
    assert meta["total_frames"] == 370
    assert meta["video_codec"] == "h264"
    assert meta["has_audio"] is True
    assert meta["audio_codec"] == "aac"
    assert meta["size_mb"] == 1.0


def test_get_video_info_tool():
    # 参数校验
    with pytest.raises(Exception):
        GetVideoInfoArgs.model_validate({})

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"video content")
        f_path = Path(f.name)

    try:
        mock_probe = {
            "format": {"duration": "10.0", "format_name": "mp4", "size": "2048"},
            "streams": [{"codec_type": "video", "codec_name": "h264", "width": 1280, "height": 720, "avg_frame_rate": "24/1"}],
        }
        with patch("tools.video.get_video_info.run_ffprobe", return_value=mock_probe):
            res = execute_get_info(path=str(f_path))
            assert res["status"] == "success"
            assert res["resolution"] == "1280x720"
            assert res["duration_seconds"] == 10.0
            assert res["fps"] == 24.0
    finally:
        f_path.unlink(missing_ok=True)


def test_capture_video_frame_args_validation():
    # 缺少位置定位（timestamp 或 frame_index）
    with pytest.raises(Exception):
        CaptureVideoFrameArgs.model_validate({"path": "sample.mp4"})

    # 具备 timestamp
    args = CaptureVideoFrameArgs.model_validate({"path": "sample.mp4", "timestamp": "00:05"})
    assert args.timestamp == "00:05"

    # 具备 frame_index
    args2 = CaptureVideoFrameArgs.model_validate({"path": "sample.mp4", "frame_index": 60})
    assert args2.frame_index == 60


def test_capture_video_frame_execution():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"video content")
        f_path = Path(f.name)

    try:
        fake_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00fake_jpeg"
        mock_probe = {
            "format": {"duration": "60.0", "format_name": "mp4"},
            "streams": [{"codec_type": "video", "width": 1920, "height": 1080, "avg_frame_rate": "30/1"}],
        }

        with patch("tools.video.capture_video_frame.run_ffprobe", return_value=mock_probe), \
             patch("tools.video.capture_video_frame.extract_frame_bytes", return_value=fake_jpeg), \
             patch("tools.video.capture_video_frame.register_frame_as_image", return_value="img_ref_9999"):

            res = execute_capture_frame(path=str(f_path), timestamp=10.5)
            assert res["status"] == "success"
            assert res["timestamp_seconds"] == 10.5
            assert res["image_ref"] == "img_ref_9999"
            assert res["frame_index"] == 315
            assert "_multimodal_parts" in res
            assert len(res["_multimodal_parts"]) == 1
            assert res["_multimodal_parts"][0]["data"] == fake_jpeg
            assert res["_multimodal_parts"][0]["mime_type"] == "image/jpeg"
    finally:
        f_path.unlink(missing_ok=True)
