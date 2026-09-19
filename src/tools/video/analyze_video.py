"""analyze_video.py — 围绕 Gemini 多模态 API 进行视频识别与分析的工具函数。

功能特征：
1. 凭据支持：video_ref 或 Agent-home Linux path（只能提供一个）
2. 传输协议：直接通过 Base64 内联请求模型，不调用 File API
3. 安全限制：文件默认 20 MiB、硬上限 128 MiB，并校验内联请求体大小
4. 分析模式：支持 "static"（全局总结）与 "agentic"（精准时戳与决策线索），支持自定义 prompt
5. 采样参数：temperature 保持默认，resolution 保持默认
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from tools.contract import tool
from .client import DEFAULT_VIDEO_SYSTEM_INSTRUCTION, VideoModelClient, VideoProcessingError
from .common import VideoBaseArgs, resolved_video_input

logger = logging.getLogger("AICQ.video")
RESULT_CDATA = True

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
        description="'static' 为固定采样，适合整体概括；'agentic' 为动态检索，需要支持该能力的 Gemini 原生端点和模型，OpenAI 兼容协议不支持。默认 static。",
    )


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
        with resolved_video_input(args) as video_path:
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
                "target": args.video_ref or args.path,
                "file_name": Path(args.path).name if args.path else video_path.name,
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
