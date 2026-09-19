"""capture_video_frame.py — 精准截取视频单帧并供模型直接查看的工具函数。

功能特征：
1. 凭据支持：支持传入 video_ref 或 本地文件绝对 path（两者必填其一）
2. 抽帧定位：timestamp 时间 seek 或 frame_index 零基解码索引，不用 FPS 猜测帧号
3. 图像注册：抽出的单帧自动注册到系统媒体库，生成全局持久化的 image_ref 供后续复用与发送
4. 视觉感知：通过 _multimodal_parts 将帧图像二进制直接内联注入模型响应，主模型可当场观察画面
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from tools.contract import tool
from .client import VideoProcessingError
from .common import (
    VideoBaseArgs,
    extract_frame_bytes,
    extract_metadata,
    format_duration,
    parse_timestamp_to_seconds,
    register_frame_as_image,
    resolved_video_input,
    run_ffprobe,
)

logger = logging.getLogger("AICQ.video")
RESULT_CDATA = True


class CaptureVideoFrameArgs(VideoBaseArgs):
    timestamp: float | str | None = Field(
        default=None,
        description="抽帧目标时间戳。支持浮点数秒数（如 12.5）或时分秒时间串（如 '00:01:23.500'）。与 frame_index 必填其一。",
    )
    frame_index: int | None = Field(
        default=None,
        ge=0,
        description="按解码顺序从 0 开始的精确帧序号。与 timestamp 必须且只能提供一个。",
    )

    @model_validator(mode="after")
    def validate_frame_position(self) -> "CaptureVideoFrameArgs":
        if (self.timestamp is None) == (self.frame_index is None):
            raise ValueError("timestamp 与 frame_index 必须且只能提供一个")
        return self


@tool(
    name="capture_video_frame",
    description="精准截取视频的某一帧画面并直接呈现给多模态大模型观察。",
    args_model=CaptureVideoFrameArgs,
)
def execute(args: CaptureVideoFrameArgs) -> dict[str, Any]:
    try:
        with resolved_video_input(args) as video_path:
            logger.info(
                "[video] 开始视频截帧 path=%s timestamp=%s frame_index=%s",
                video_path,
                args.timestamp,
                args.frame_index,
            )

            meta = extract_metadata(run_ffprobe(video_path), video_path)
            ts_sec = parse_timestamp_to_seconds(args.timestamp) if args.timestamp is not None else None

            duration = float(meta.get("duration_seconds") or 0.0)
            if duration > 0 and ts_sec is not None and ts_sec >= duration:
                raise VideoProcessingError(
                    f"请求的抽帧时间点 ({ts_sec:.3f}s) 超出视频总时长 ({duration:.3f}s)"
                )

            frame_bytes = extract_frame_bytes(video_path, timestamp_sec=ts_sec, frame_index=args.frame_index)
            image_ref = register_frame_as_image(frame_bytes, source="video_frame")
            if not image_ref:
                raise VideoProcessingError("截帧图片持久化失败")

            return {
                "status": "success",
                "target": args.video_ref or args.path,
                "file_name": Path(args.path).name if args.path else video_path.name,
                "requested_timestamp_seconds": ts_sec,
                "requested_timestamp_human": format_duration(ts_sec) if ts_sec is not None else None,
                "frame_index": args.frame_index,
                "image_ref": image_ref,
                "mime_type": "image/jpeg",
                "size_bytes": len(frame_bytes),
                "width": meta.get("width"),
                "height": meta.get("height"),
                "_multimodal_parts": [
                    {
                        "data": frame_bytes,
                        "mime_type": "image/jpeg",
                        "display_name": "video_frame.jpg",
                    }
                ],
            }
    except VideoProcessingError as exc:
        logger.warning("[video] 视频截帧失败: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "code": "video_processing_failed",
        }
    except Exception as exc:
        logger.exception("[video] 视频截帧过程出现未捕获异常: %s", exc)
        return {
            "status": "error",
            "error": f"视频截帧执行失败: {exc}",
            "code": "internal_error",
        }
