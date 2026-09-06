"""收藏可见聊天、历史或转发窗口中的图片，保留原 image_ref。"""

from llm.media.image_resolver import ImageResolver, image_bytes, normalize_image_ref
from llm.media.sticker_collection import MAX_STICKERS, StickerCollectionError, save_sticker
from pydantic import Field

from platforms.qq.session_context import NO_CURRENT_SESSION_ERROR, ensure_session_provider
from tools.contract import ToolArgsModel, ToolContract


class SaveStickerArgs(ToolArgsModel):
    image_ref: str = Field(
        min_length=4,
        description=
        "目标图片/表情的 image_ref，12位十六进制字符串"
        "（来自上下文 XML 中的 image_ref 标注）"
            )
    description: str = Field(min_length=1, description="表情包的大致描述、印象和适用场景。")


TOOL_CONTRACT = ToolContract(
    name="save_sticker",
    description=(
        "将任何可见图片收藏为 qq 表情包。"
        "需写入目标图片的 image_ref。"
    ),
    args_model=SaveStickerArgs,
)
REQUIRES_CONTEXT = ["qq_session_provider"]


def make_handler(qq_session_provider):
    provider = ensure_session_provider(qq_session_provider)

    def handler(image_ref: str, description: str, **_) -> dict:
        session = provider()
        if session is None:
            return {"error": NO_CURRENT_SESSION_ERROR}
        image_ref = normalize_image_ref(image_ref)
        resolver = ImageResolver(session)
        found = resolver.resolve(image_ref, include_browser=False)
        if found is None:
            return {"error": "未找到可见或已收藏的图片", "code": "not_found"}
        image, _source = found
        payload = image_bytes(image)
        if payload is None:
            return {"error": "图片原始数据不可用", "code": resolver.unavailable_status(image)}
        raw, mime = payload
        try:
            result = save_sticker(raw, mime, description, image_ref=image_ref)
        except StickerCollectionError as exc:
            return {"error": str(exc), "code": exc.code}
        if result is None:
            return {"error": f"收藏已满（上限 {MAX_STICKERS} 个），请先用 delete_sticker 删除旧收藏", "code": "collection_full"}
        ref, duplicate = result
        return {
            "image_ref": ref,
            "duplicate": duplicate,
            "message": "图片已在收藏中，已保留原印象。" if duplicate else "已收藏，可通过 image_ref 使用。",
        }

    return handler
