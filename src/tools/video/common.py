"""common.py — 视频处理公共模块。

包含：
1. 统一凭据参数模型 VideoBaseArgs（支持 video_ref 与 path 二选一校验）
2. 本地视频路径解析 resolve_video_path
3. 基于 ffprobe 的视频结构化元数据提取
4. 基于 ffmpeg 管道的快速单帧截取
5. 抽帧结果注册系统 image_ref
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
from typing import Any

from pydantic import Field, model_validator

from tools.contract import ToolArgsModel
from .client import VideoProcessingError

logger = logging.getLogger("AICQ.video")


class VideoBaseArgs(ToolArgsModel):
    video_ref: str | None = Field(
        default=None,
        description="目标视频的 video_ref（例如来自消息上下文中的视频引用凭据）。与 path 必填其一。",
    )
    path: str | None = Field(
        default=None,
        description="目标视频在本地系统的绝对文件路径。与 video_ref 必填其一。",
    )

    @model_validator(mode="after")
    def validate_credential(self) -> "VideoBaseArgs":
        ref = (self.video_ref or "").strip()
        fpath = (self.path or "").strip()
        if not ref and not fpath:
            raise ValueError("必须提供 video_ref 或 path 两者之一作为视频定位凭据")
        return self


def resolve_video_path(args: VideoBaseArgs) -> Path:
    """根据传入凭据定位本地视频文件。"""
    if args.path and args.path.strip():
        resolved_path = Path(args.path.strip()).expanduser().resolve()
        if not resolved_path.is_file():
            raise VideoProcessingError(f"指定的视频文件路径不存在: {args.path}")
        return resolved_path

    ref = (args.video_ref or "").strip()
    try:
        from llm.media.video_store import locate_video

        located = locate_video(ref)
        if located is not None and located.is_file():
            return located.resolve()
    except Exception as exc:
        logger.warning("[video] 尝试通过 locate_video 定位 ref=%s 异常: %s", ref, exc)

    raise VideoProcessingError(f"未能根据 video_ref '{ref}' 定位到已下载的本地视频文件")


def run_ffprobe(video_path: Path, timeout: float = 30.0) -> dict[str, Any]:
    """通过 ffprobe 提取视频基础原始信息。"""
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        raise VideoProcessingError("宿主环境中未检测到 ffprobe，无法提取视频元数据")

    cmd = [
        ffprobe_cmd,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VideoProcessingError(f"ffprobe 解析视频超时 (>{timeout}s)") from exc
    except Exception as exc:
        raise VideoProcessingError(f"执行 ffprobe 解析视频失败: {exc}") from exc

    if res.returncode != 0:
        err_msg = (res.stderr or "").strip() or f"ffprobe 退出码: {res.returncode}"
        raise VideoProcessingError(f"ffprobe 分析视频失败: {err_msg}")

    try:
        return json.loads(res.stdout)
    except Exception as exc:
        raise VideoProcessingError(f"解析 ffprobe JSON 数据异常: {exc}") from exc


def format_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS 或 MM:SS 字符串。"""
    total_sec = max(0.0, float(seconds))
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    secs = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def parse_fraction(val: str | None) -> float:
    """解析如 '30/1' 或 '24000/1001' 形式的帧率分数字符串。"""
    if not val:
        return 0.0
    val_str = str(val).strip()
    if "/" in val_str:
        num, _, denom = val_str.partition("/")
        try:
            d = float(denom)
            return float(num) / d if d != 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(val_str)
    except ValueError:
        return 0.0


def extract_metadata(probe_data: dict[str, Any], file_path: Path) -> dict[str, Any]:
    """从原始 ffprobe 数据中提取并清洗出规范的视频元数据结构。"""
    fmt = probe_data.get("format") or {}
    streams = probe_data.get("streams") or []

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    duration_str = fmt.get("duration") or video_stream.get("duration") or "0"
    try:
        duration_sec = float(duration_str)
    except ValueError:
        duration_sec = 0.0

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    resolution = f"{width}x{height}" if width and height else "unknown"

    fps = parse_fraction(video_stream.get("avg_frame_rate")) or parse_fraction(video_stream.get("r_frame_rate"))
    fps = round(fps, 3)

    nb_frames_raw = video_stream.get("nb_frames")
    total_frames = 0
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        total_frames = int(nb_frames_raw)
    elif duration_sec > 0 and fps > 0:
        total_frames = int(round(duration_sec * fps))

    bit_rate_raw = fmt.get("bit_rate") or video_stream.get("bit_rate")
    bit_rate_kbps = round(int(bit_rate_raw) / 1000, 1) if bit_rate_raw and str(bit_rate_raw).isdigit() else None

    size_bytes = int(fmt.get("size") or (file_path.stat().st_size if file_path.is_file() else 0))
    size_mb = round(size_bytes / (1024 * 1024), 2)

    return {
        "file_name": file_path.name,
        "format_name": fmt.get("format_name", "unknown"),
        "duration_seconds": round(duration_sec, 3),
        "duration_human": format_duration(duration_sec),
        "resolution": resolution,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "video_codec": video_stream.get("codec_name", "unknown"),
        "pix_fmt": video_stream.get("pix_fmt", "unknown"),
        "bit_rate_kbps": bit_rate_kbps,
        "has_audio": bool(audio_stream),
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
    }


def parse_timestamp_to_seconds(ts: float | int | str) -> float:
    """将秒数数字或 HH:MM:SS[.xxx] 格式时间字符串转换为浮点秒数。"""
    if isinstance(ts, (int, float)):
        return max(0.0, float(ts))

    ts_str = str(ts).strip()
    if not ts_str:
        return 0.0

    if ":" not in ts_str:
        try:
            return max(0.0, float(ts_str))
        except ValueError as exc:
            raise VideoProcessingError(f"无效的时间戳数值: {ts}") from exc

    parts = ts_str.split(":")
    try:
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return max(0.0, h * 3600 + m * 60 + s)
        if len(parts) == 2:
            m = float(parts[0])
            s = float(parts[1])
            return max(0.0, m * 60 + s)
    except ValueError as exc:
        raise VideoProcessingError(f"无效的时间格式字符串: {ts}") from exc

    raise VideoProcessingError(f"无法识别的时间格式: {ts}")


def extract_frame_bytes(
    video_path: Path,
    *,
    timestamp_sec: float,
    timeout: float = 30.0,
) -> bytes:
    """利用 ffmpeg 通过管道直接截取单帧并返回 JPEG 字节流。"""
    ffmpeg_cmd = shutil.which("ffmpeg")
    if not ffmpeg_cmd:
        raise VideoProcessingError("宿主环境中未检测到 ffmpeg，无法执行截帧操作")

    cmd = [
        ffmpeg_cmd,
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-vframes",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise VideoProcessingError(f"ffmpeg 抽帧超时 (>{timeout}s)") from exc
    except Exception as exc:
        raise VideoProcessingError(f"执行 ffmpeg 抽帧异常: {exc}") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise VideoProcessingError(f"ffmpeg 抽帧失败: {err_msg[:400]}")

    frame_bytes = proc.stdout
    if not frame_bytes:
        raise VideoProcessingError(f"抽帧未获取到任何图像数据，时间戳 {timestamp_sec:.3f}s 可能已超出视频总时长")

    return frame_bytes


def register_frame_as_image(frame_bytes: bytes, source: str = "video_frame") -> str:
    """将抽取的帧字节注册至系统媒体库，返回分配的 image_ref。"""
    try:
        from llm.media.image_store import register_image

        record = register_image(frame_bytes, source)
        return str(record.get("image_ref") or "")
    except Exception as exc:
        logger.warning("[video] 尝试注册截帧图片到 image_store 失败: %s", exc)
        return ""
