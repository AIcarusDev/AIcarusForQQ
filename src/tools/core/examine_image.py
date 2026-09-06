"""examine_image.py — 定向精细观察对话中的图片

主模型主动调用此工具，指定 image_ref（12位十六进制图片引用）和 focus，
VisionBridge 带焦点重新询问 VLM，结果写入内存和 sidecar。

启用条件：session 和 vision_bridge 均在运行时上下文中就绪。
"""

import base64
import logging

from llm.media.image_resolver import ImageResolver, image_bytes, normalize_image_ref
from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract

logger = logging.getLogger("AICQ.tools")

class ExamineImageArgs(ToolArgsModel):
    image_ref: str = Field(
        min_length=1,
        description=(
            "目标可见或已收藏图片的 image_ref"
            "（来自上下文 XML 中 <description> 或工具响应里标注的 image_ref）"
        ),
    )
    focus: str = Field(
        min_length=1,
        description=(
            "本次重点观察的内容，尽量具体，例如："
            "'右侧的报错文字' / '人物的面部表情' / '左上角的数字' / '图中的二维码'"
        ),
    )


TOOL_CONTRACT = ToolContract(
    name="examine_image",
    description=(
        "对某张图片进行定向精细观察。"
        "当你在上下文中看到 [图片] 标记，需要了解图片特定区域或细节时调用。"
        "需写入目标图片的的 image_ref。"
    ),
    args_model=ExamineImageArgs,
)

# 需要 session（遍历上下文消息找 image_ref）和 vision_bridge（调用 VLM）
REQUIRES_CONTEXT: list[str] = ["session", "vision_bridge"]
PARALLEL_SAFE = True
PARALLEL_KEY = "vision_model"


def make_handler(session, vision_bridge):
    """工厂函数：绑定 session 和 vision_bridge，返回工具处理函数。"""

    def handler(image_ref: str, focus: str, **_) -> dict:
        image_ref = normalize_image_ref(image_ref)
        resolver = ImageResolver(session)
        found = resolver.resolve(image_ref)
        if found is None:
            return {"error": "未找到可见或已收藏的图片", "code": "not_found"}
        target_img, _source = found
        payload = image_bytes(target_img)
        if payload is None:
            return {"error": "图片原始数据不可用，无法精查", "code": resolver.unavailable_status(target_img)}
        raw, mime = payload
        b64 = base64.b64encode(raw).decode("ascii")
        phash = target_img.get("phash")
        image_ref = str(target_img.get("image_ref") or image_ref)

        if not vision_bridge.enabled:
            logger.warning("[tools] examine_image: VisionBridge 未启用")
            return {"error": "视觉桥（VisionBridge）未启用，无法进行图片精查"}

        # ── 3. 调用 VLM 精查 ─────────────────────────────────────
        logger.info("[tools] examine_image: 开始精查 focus=%r image_ref=%s", focus, image_ref)
        result_text = vision_bridge.examine(phash, b64, mime, focus)
        if result_text is None:
            logger.warning("[tools] examine_image: VLM 返回为空 image_ref=%s", image_ref)
            return {"error": "精查失败，VLM 未能返回有效结果，请稍后重试"}

        logger.info("[tools] examine_image: 精查完成 image_ref=%s", image_ref)
        # ── 4. 同步更新内存中的 examinations ─────────────────────
        if "examinations" not in target_img:
            target_img["examinations"] = []
        target_img["examinations"].append(
            {"focus": focus, "result": result_text}
        )

        logger.info("[examine_image] image_ref=%s focus=%r", image_ref, focus)

        return {
            "image_ref": image_ref,
            "focus": focus,
            "result": result_text,
        }

    return handler
