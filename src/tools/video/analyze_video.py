"""analyze_video.py — 围绕 Gemini 多模态 API 进行视频识别与分析的工具函数。

功能特征：
1. 凭据支持：支持传入 video_ref 或 本地文件绝对 path（两者必填其一）
2. 传输协议：直接通过 Base64 内联请求模型，不调用 File API
3. 安全限制：内建请求体/文件大小硬限制（默认 128MB），超限直接拦截
4. 分析模式：支持 "static"（全局总结）与 "agentic"（精准时戳与决策线索），支持自定义 prompt
5. 采样参数：temperature 保持默认，resolution 保持默认
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from tools.contract import tool
from .client import DEFAULT_VIDEO_SYSTEM_INSTRUCTION, VideoModelClient, VideoProcessingError
from .common import VideoBaseArgs, resolve_video_path

logger = logging.getLogger("AICQ.video")

# 统一通用系统指令（保持兼容别名）
DEFAULT_SYSTEM_INSTRUCTION = DEFAULT_VIDEO_SYSTEM_INSTRUCTION
DEFAULT_STATIC_PROMPT = DEFAULT_VIDEO_SYSTEM_INSTRUCTION
DEFAULT_AGENTIC_PROMPT = DEFAULT_VIDEO_SYSTEM_INSTRUCTION


class AnalyzeVideoArgs(VideoBaseArgs):
    prompt: str | None = Field(
        default=None,
        description="针对视频的定向查询问题或特定关注重点。不传时模型将默认全面总结整段视频的核心内容。",
    )
    mode: Literal["static", "agentic"] = Field(
        default="static",
        description="视频采样与检索模式：'static' 为全量固定抽帧（适合整体粗看与短视频）；'agentic' 为动态交互检索（适合长视频中精准定位特定瞬间与细节）。默认为 'static'。",
    )


# 保持向后兼容别名
_resolve_video_path = resolve_video_path


def _build_final_prompt(prompt: str | None, mode: str = "static") -> str | None:
    """提取用户定向提问。若无提问则返回 None，由模型依据 System Instruction 执行默认的全视频理解。"""
    custom_query = (prompt or "").strip()
    return custom_query if custom_query else None


@tool(
    name="analyze_video",
    description="围绕多模态模型对视频进行内容识别与深度分析。",
    args_model=AnalyzeVideoArgs,
)
def execute(args: AnalyzeVideoArgs) -> dict[str, Any]:
    try:
        video_path = _resolve_video_path(args)
        user_prompt = _build_final_prompt(args.prompt, args.mode)

        logger.info(
            "[video] 开始分析视频 path=%s mode=%s has_custom_prompt=%s",
            video_path,
            args.mode,
            bool(user_prompt),
        )

        client = VideoModelClient()
        analysis_result = client.analyze(video_path, prompt=user_prompt, mode=args.mode)

        return {
            "status": "success",
            "target": args.video_ref if args.video_ref else str(video_path),
            "file_name": video_path.name,
            "mode": args.mode,
            "prompt_used": user_prompt,
            "analysis": analysis_result,
        }
    except VideoProcessingError as exc:
        logger.warning("[video] 视频分析业务失败: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "code": "video_processing_failed",
        }
    except Exception as exc:
        logger.exception("[video] 视频分析过程出现未捕获异常: %s", exc)
        return {
            "status": "error",
            "error": f"视频分析执行失败: {exc}",
            "code": "internal_error",
        }
