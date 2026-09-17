"""capture_video_frame.py — 精准截取视频单帧并供模型直接查看的工具函数。

功能特征：
1. 凭据支持：支持传入 video_ref 或 本地文件绝对 path（两者必填其一）
2. 抽帧定位：支持指定 timestamp（秒数如 12.5 或时间串 '00:01:23.500'）或指定 frame_index（帧号）
3. 图像注册：抽出的单帧自动注册到系统媒体库，生成全局持久化的 image_ref 供后续复用与发送
4. 视觉感知：通过 _multimodal_parts 将帧图像二进制直接内联注入模型响应，主模型可当场观察画面
"""

from __future__ import annotations

import logging
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
    resolve_video_path,
    run_ffprobe,
)

logger = logging.getLogger("AICQ.video")


class CaptureVideoFrameArgs(VideoBaseArgs):
    timestamp: float | str | None = Field(
        default=None,
        description="抽帧目标时间戳。支持浮点数秒数（如 12.5）或时分秒时间串（如 '00:01:23.500'）。与 frame_index 必填其一。",
    )
    frame_index: int | None = Field(
        default=None,
        description="抽帧目标帧序号（如第 150 帧）。若未提供 timestamp，将结合视频 FPS 计算对应时间戳。与 timestamp 必填其一。",
    )

    @model_validator(mode="after")
    def validate_frame_position(self) -> "CaptureVideoFrameArgs":
        if self.timestamp is None and self.frame_index is None:
            raise ValueError("必须提供 timestamp 或 frame_index 两者之一作为抽帧目标位置")
        return self


@tool(
    name="capture_video_frame",
    description="精准截取视频的某一帧画面并直接呈现给多模态大模型观察。",
    args_model=CaptureVideoFrameArgs,
)
def execute(args: CaptureVideoFrameArgs) -> dict[str, Any]:
    try:
        video_path = resolve_video_path(args)
        logger.info(
            "[video] 开始视频截帧 path=%s timestamp=%s frame_index=%s",
            video_path,
            args.timestamp,
            args.frame_index,
        )

        # 获取视频元数据以校验与辅助换算
        meta: dict[str, Any] = {}
        try:
            probe_data = run_ffprobe(video_path)
            meta = extract_metadata(probe_data, video_path)
        except Exception as exc:
            logger.warning("[video] 截帧前尝试提取元数据轻微受挫: %s", exc)

        fps = float(meta.get("fps") or 30.0)
        if fps <= 0:
            fps = 30.0

        if args.timestamp is not None:
            ts_sec = parse_timestamp_to_seconds(args.timestamp)
            calc_frame = int(round(ts_sec * fps))
        else:
            calc_frame = max(0, int(args.frame_index or 0))
            ts_sec = float(calc_frame) / fps

        duration = float(meta.get("duration_seconds") or 0.0)
        if duration > 0 and ts_sec > duration:
            raise VideoProcessingError(
                f"请求的抽帧时间点 ({ts_sec:.3f}s) 超出视频总时长 ({duration:.3f}s)"
            )

        frame_bytes = extract_frame_bytes(video_path, timestamp_sec=ts_sec)
        image_ref = register_frame_as_image(frame_bytes, source="video_frame")

        return {
            "status": "success",
            "target": args.video_ref if args.video_ref else str(video_path),
            "file_name": video_path.name,
            "timestamp_seconds": round(ts_sec, 3),
            "timestamp_human": format_duration(ts_sec),
            "frame_index": calc_frame,
            "image_ref": image_ref,
            "mime_type": "image/jpeg",
            "size_bytes": len(frame_bytes),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "_multimodal_parts": [
                {
                    "data": frame_bytes,
                    "mime_type": "image/jpeg",
                    "display_name": f"frame_at_{round(ts_sec, 3)}s.jpg",
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
