"""按 image_ref 删除收藏表情包。"""

from llm.media.image_resolver import normalize_image_ref
from llm.media.sticker_collection import StickerCollectionError, delete_sticker
from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract


class DeleteStickerArgs(ToolArgsModel):
    image_ref: str = Field(min_length=4, description="要删除的已收藏表情包的 image_ref。")


TOOL_CONTRACT = ToolContract(
    name="delete_sticker",
    description=(
        "通过 image_ref 取消表情包收藏；原图和历史 image_ref 仍可读取。"
        "已保存到工作空间的副本独立保留。"
    ),
    args_model=DeleteStickerArgs,
)


def execute(image_ref: str, **_) -> dict:
    try:
        ref = delete_sticker(normalize_image_ref(image_ref))
    except StickerCollectionError as exc:
        return {"error": str(exc), "code": exc.code}
    if ref is None:
        return {"error": "该 image_ref 不属于已收藏表情包", "code": "not_found"}
    return {"image_ref": ref, "message": "已取消表情包收藏，原图和 image_ref 仍保留。"}
