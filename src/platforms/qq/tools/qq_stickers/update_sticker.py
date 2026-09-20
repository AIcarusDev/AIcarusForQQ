"""按 image_ref 修改收藏表情包的印象。"""

from llm.media.image_resolver import normalize_image_ref
from llm.media.sticker_collection import StickerCollectionError, update_sticker_description
from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract


class UpdateStickerArgs(ToolArgsModel):
    image_ref: str = Field(min_length=4, description="已收藏表情包的 image_ref。")
    description: str = Field(min_length=1, description="新的表情包印象和适用场景。")


TOOL_CONTRACT = ToolContract(
    name="update_sticker",
    description="修改已收藏表情包的印象和适用场景；可用 list_stickers 查看收藏引用。",
    args_model=UpdateStickerArgs,
)


def execute(image_ref: str, description: str, **_) -> dict:
    try:
        ref = update_sticker_description(normalize_image_ref(image_ref), description)
    except StickerCollectionError as exc:
        return {"error": str(exc), "code": exc.code}
    if ref is None:
        return {"error": "该 image_ref 不属于已收藏表情包", "code": "not_found"}
    return {"image_ref": ref, "new_description": description, "message": "表情包印象已更新。"}
