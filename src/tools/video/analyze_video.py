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

from tools.contract import ToolArgsModel, tool
from .client import VideoModelClient, VideoProcessingError

logger = logging.getLogger("AICQ.video")

DEFAULT_STATIC_PROMPT = """请对这段视频进行全面、连贯的视频内容理解与分析，输出内容涵盖：
1. 视频全局总结：概括视频的核心内容、主要场景与主题。
2. 核心主体与对象：视频中出现的人物、物体、关键文字或标识。
3. 过程与情节脉络：视频整体发生的关键过程或状态发展。
4. 重要细节与亮点：任何值得关注的背景细节或突出特征。"""

DEFAULT_AGENTIC_PROMPT = """你是一个专业的智能代理视觉分析器，请对这段视频进行结构化、带精准时间戳的高敏度分析，以供后续智能体决策行动：
1. 时序事件线索：按时间戳区间（如 [00:01 - 00:05]）清晰列出发生的所有关键动作、状态转变或交互。
2. 关键实体与位置：明确关键主体所处画面位置、动作轨迹及状态。
3. 置信度与不确定项：对画面中细节不足、模糊或有歧义的观察点标注置信度与潜在疑问。
4. 行动参考线索：提炼出对后续任务规划有直接参考价值的关键事实。"""


class AnalyzeVideoArgs(ToolArgsModel):
    video_ref: str | None = Field(
        default=None,
        description="目标视频的 video_ref（例如来自消息上下文中的视频引用凭据）。与 path 必填其一。",
    )
    path: str | None = Field(
        default=None,
        description="目标视频在本地系统的绝对文件路径。与 video_ref 必填其一。",
    )
    prompt: str | None = Field(
        default=None,
        description="针对视频的定向查询问题或特定关注重点。不传时将使用默认的全视频深入分析 Prompt。",
    )
    mode: Literal["static", "agentic"] = Field(
        default="static",
        description="分析模式：'static' 模式输出全局连贯的内容总结；'agentic' 模式输出带精准时间戳、关键动作/事件线索与置信度的结构化分析，便于后续行动决策。默认为 'static'。",
    )

    @model_validator(mode="after")
    def validate_credential(self) -> "AnalyzeVideoArgs":
        ref = (self.video_ref or "").strip()
        fpath = (self.path or "").strip()
        if not ref and not fpath:
            raise ValueError("必须提供 video_ref 或 path 两者之一作为视频定位凭据")
        return self


def _resolve_video_path(args: AnalyzeVideoArgs) -> Path:
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


def _build_final_prompt(prompt: str | None, mode: str) -> str:
    """根据模式与自定义提问组装最终发送给多模态模型的提示词。"""
    custom_query = (prompt or "").strip()
    if not custom_query:
        return DEFAULT_AGENTIC_PROMPT if mode == "agentic" else DEFAULT_STATIC_PROMPT

    if mode == "agentic":
        return (
            f"你是一个专业的智能代理视觉分析器。请结合精准时序与动作线索深入分析视频，并针对以下重点问题进行结构化解答：\n"
            f"【重点关注/提问】：{custom_query}\n\n"
            f"要求：\n"
            f"1. 优先针对提问给出精准分析与解答。\n"
            f"2. 涉及视频片段时必须标注具体时间戳（如 [00:02 - 00:06]）。\n"
            f"3. 提取出有助于后续行动决策的关键线索与观察置信度。"
        )

    return (
        f"请仔细观察这段视频，并重点回答以下问题或针对以下关注点进行分析：\n"
        f"【提问/关注点】：{custom_query}\n\n"
        f"请结合全视频上下文，提供清晰、客观、详实完整的回答与概括。"
    )


@tool(
    name="analyze_video",
    description=(
        "围绕多模态模型对单段视频进行内容识别与深度分析。"
        "支持通过 video_ref 或 本地文件 path 指定视频；"
        "支持 static（全局总结）与 agentic（时序线索与决策线索）两种模式；"
        "可注入自定义 prompt 进行定向查询，默认执行全视频详尽分析。"
    ),
    args_model=AnalyzeVideoArgs,
)
def execute(args: AnalyzeVideoArgs) -> dict[str, Any]:
    try:
        video_path = _resolve_video_path(args)
        final_prompt = _build_final_prompt(args.prompt, args.mode)

        logger.info(
            "[video] 开始分析视频 path=%s mode=%s has_custom_prompt=%s",
            video_path,
            args.mode,
            bool(args.prompt),
        )

        client = VideoModelClient()
        analysis_result = client.analyze(video_path, final_prompt)

        return {
            "status": "success",
            "target": args.video_ref if args.video_ref else str(video_path),
            "file_name": video_path.name,
            "mode": args.mode,
            "prompt_used": final_prompt,
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
