"""test_video_tool.py — video 命名空间及 analyze_video 工具回归测试。"""

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
    execute,
)
from tools.video.client import (
    DEFAULT_MAX_SIZE_MB,
    VideoModelClient,
    VideoProcessingError,
    detect_video_mime,
    get_video_config,
)


def test_detect_video_mime():
    assert detect_video_mime(Path("sample.mp4")) == "video/mp4"
    assert detect_video_mime(Path("sample.webm")) == "video/webm"
    assert detect_video_mime(Path("sample.mov")) == "video/quicktime"
    assert detect_video_mime(Path("sample.mkv")) == "video/x-matroska"
    assert detect_video_mime(Path("sample.unknown")) == "video/mp4"


def test_prompt_builder():
    # 默认 static 与 agentic
    assert _build_final_prompt(None, "static") == DEFAULT_STATIC_PROMPT
    assert _build_final_prompt(None, "agentic") == DEFAULT_AGENTIC_PROMPT

    # 自定义 prompt
    static_custom = _build_final_prompt("视频里有几个人？", "static")
    assert "视频里有几个人？" in static_custom
    assert "请仔细观察这段视频" in static_custom

    agentic_custom = _build_final_prompt("视频里有几个人？", "agentic")
    assert "视频里有几个人？" in agentic_custom
    assert "智能代理视觉分析器" in agentic_custom


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
    result = execute()
    assert "error" in result
    assert "工具参数不符合定义" in result["error"]


def test_execute_non_existent_file():
    result = execute(path="non_existent_video_path_123.mp4")
    assert result["status"] == "error"
    assert "不存在" in result["error"]


def test_execute_mocked_success():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"fake video data")
        f_path = Path(f.name)

    try:
        with patch.object(VideoModelClient, "analyze", return_value="模拟视频分析结果"):
            res = execute(path=str(f_path), mode="agentic")
            assert res["status"] == "success"
            assert res["mode"] == "agentic"
            assert res["analysis"] == "模拟视频分析结果"
    finally:
        f_path.unlink(missing_ok=True)
