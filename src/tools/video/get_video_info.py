"""get_video_info.py — 提取视频结构化元数据的工具函数。

功能特征：
1. 凭据支持：支持传入 video_ref 或 本地文件绝对 path（两者必填其一）
2. 元数据分析：解析视频时长、分辨率、帧率、总帧数、视频编码、音频轨道、文件体积等
3. 零额外重负：基于系统底层 ffprobe 直接解析，性能优越
"""

from __future__ import annotations

import logging
from typing import Any

from tools.contract import tool
from .client import VideoProcessingError
from .common import (
    VideoBaseArgs,
    extract_metadata,
    resolved_video_input,
    run_ffprobe,
)

logger = logging.getLogger("AICQ.video")
RESULT_CDATA = True


class GetVideoInfoArgs(VideoBaseArgs):
    """提取视频元数据参数模型。"""


@tool(
    name="get_video_info",
    description="快速提取视频的结构化元数据信息。",
    args_model=GetVideoInfoArgs,
)
def execute(args: GetVideoInfoArgs) -> dict[str, Any]:
    try:
        with resolved_video_input(args) as video_path:
            logger.info("[video] 开始获取视频元数据 path=%s", video_path)

            probe_data = run_ffprobe(video_path)
            meta = extract_metadata(probe_data, video_path)
            if args.path:
                from pathlib import PurePosixPath
                meta["file_name"] = PurePosixPath(args.path).name

            return {
                "status": "success",
                "target": args.video_ref or args.path,
                **meta,
            }
    except VideoProcessingError as exc:
        logger.warning("[video] 获取视频元数据失败: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "code": "video_processing_failed",
        }
    except Exception as exc:
        logger.exception("[video] 获取视频元数据过程出现未捕获异常: %s", exc)
        return {
            "status": "error",
            "error": f"获取视频元数据执行失败: {exc}",
            "code": "internal_error",
        }
