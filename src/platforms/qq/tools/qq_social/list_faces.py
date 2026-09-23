"""Page through the native QQ face catalog exposed by the local QQ resources."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal, NamedTuple

from pydantic import Field

from tools.contract import ToolArgsModel, ToolContract


logger = logging.getLogger("AICQ.tools")
PAGE_SIZE = 30
_FACE_RESOURCE = Path("nt_qq/global/nt_data/Emoji/emoji-resource/face_config.json")
_DECIMAL_ID = re.compile(r"[0-9]+\Z")


class _FaceEntry(NamedTuple):
    id: int
    des: str
    normal: bool
    super: bool


class ListFacesArgs(ToolArgsModel):
    kind: Literal["normal", "super"] = Field(
        description=(
            "必填。normal 查询可与文字、图片组合发送的普通表情；"
            "super 查询适合单独发送的大动画表情。部分 ID 同时支持两种用法。"
        ),
    )
    page: int = Field(default=1, ge=1, description="页码，从 1 开始；每页固定 30 项。")
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description="可选。按表情名称包含匹配，可填写“微笑”或“/微笑”；先筛选再翻页。",
    )


TOOL_CONTRACT = ToolContract(
    name="list_faces",
    description=(
        "按 kind 分开查询 QQ 普通表情或超级表情，每次只返回一类，不混合展示。"
        "普通表情可与文字、图片组合；超级表情适合单独发送。"
        "有些普通表情单独发送时也有大动画，因此同一 ID 可能出现在两类查询中。"
        "两类发送时都使用 QQ face 消息段。"
        "每项返回数字 id 和以 / 开头的名称 des。"
        "默认返回第 1 页、每页 30 项；使用 next_page 继续查询。"
        "可用 query 按名称筛选。需要表情 ID 时以查询结果为准。"
    ),
    args_model=ListFacesArgs,
)

REQUIRES_CONTEXT = ["config"]
PARALLEL_SAFE = True
PARALLEL_KEY = "qq_face_catalog_read"


def _resource_path(config: dict[str, Any]) -> Path | None:
    platforms = config.get("platforms")
    qq = platforms.get("qq") if isinstance(platforms, dict) else None
    adapter = qq.get("adapter") if isinstance(qq, dict) else None
    if not isinstance(adapter, dict):
        return None

    explicit = str(adapter.get("face_config_path") or "").strip()
    if explicit:
        return Path(explicit)

    transfer = adapter.get("file_transfer")
    host_directory = transfer.get("host_directory") if isinstance(transfer, dict) else None
    if not host_directory:
        return None
    transfer_path = Path(str(host_directory))
    if transfer_path.name.casefold() != "transfer":
        return None
    # NapCat Docker mounts data/transfer and data/qq as sibling directories.
    return transfer_path.parent / "qq" / _FACE_RESOURCE


def _load_faces(path: Path) -> list[_FaceEntry]:
    if path.stat().st_size > 2_000_000:
        raise ValueError("QQ face catalog is unexpectedly large")
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = document.get("sysface") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("QQ face catalog has no sysface entries")

    faces: list[_FaceEntry] = []
    seen_ids: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("QQ face catalog contains an invalid entry")
        raw_id = entry.get("QSid")
        des = entry.get("QDes")
        if not isinstance(raw_id, str) or not _DECIMAL_ID.fullmatch(raw_id):
            raise ValueError("QQ face catalog contains an invalid QSid")
        if not isinstance(des, str) or not des.startswith("/") or len(des) == 1:
            raise ValueError("QQ face catalog contains an invalid QDes")
        face_id = int(raw_id)
        if face_id in seen_ids:
            raise ValueError("QQ face catalog contains a duplicate QSid")
        seen_ids.add(face_id)

        ani_type = entry.get("AniStickerType")
        if ani_type is not None and (
            isinstance(ani_type, bool) or not isinstance(ani_type, int) or ani_type < 1
        ):
            raise ValueError("QQ face catalog contains an invalid AniStickerType")
        animated = ani_type is not None
        # Pack 1 contains ordinary faces with optional standalone animation.
        # Hidden animated entries and other packs are standalone-only.
        normal = not animated or (
            entry.get("AniStickerPackId") == "1" and entry.get("QHide") != "1"
        )
        faces.append(_FaceEntry(face_id, des, normal=normal, super=animated))
    faces.sort(key=lambda item: item.id)
    return faces


def load_face_entries(config: dict[str, Any]) -> dict[int, _FaceEntry]:
    """Resolve send metadata from the same local catalog used by list_faces."""
    path = _resource_path(config)
    if path is None:
        return {}
    try:
        return {face.id: face for face in _load_faces(path)}
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        logger.debug("[tools] face catalog unavailable for send metadata: %s", exc)
        return {}


def load_face_descriptions(config: dict[str, Any]) -> dict[int, str]:
    """Resolve sent-face names from the same local catalog used by list_faces."""
    return {face_id: face.des for face_id, face in load_face_entries(config).items()}


def make_handler(config: dict[str, Any]):
    path = _resource_path(config)

    def execute(
        kind: Literal["normal", "super"],
        page: int = 1,
        query: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if kind not in ("normal", "super"):
            return {"error": "kind 必须是 normal 或 super", "code": "invalid_kind"}
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            return {"error": "page 必须是从 1 开始的页码", "code": "invalid_page"}
        if query is not None and not isinstance(query, str):
            return {"error": "query 需要包含表情名称", "code": "invalid_query"}
        if path is None:
            return {"error": "QQ 内置表情目录不可用", "code": "face_catalog_unavailable"}
        try:
            faces = _load_faces(path)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            logger.warning("[tools] list_faces: 无法读取 QQ 内置表情目录: %s", exc)
            return {"error": "QQ 内置表情目录不可用", "code": "face_catalog_unavailable"}

        faces = [face for face in faces if getattr(face, kind)]
        if query is not None:
            needle = query.strip().lstrip("/").casefold()
            if not needle:
                return {"error": "query 需要包含表情名称", "code": "invalid_query"}
            faces = [face for face in faces if needle in face.des[1:].casefold()]

        start = (page - 1) * PAGE_SIZE
        current = faces[start:start + PAGE_SIZE]
        return {
            "kind": kind,
            "page": page,
            "page_size": PAGE_SIZE,
            "total": len(faces),
            "next_page": page + 1 if start + PAGE_SIZE < len(faces) else None,
            "faces": [{"id": face.id, "des": face.des} for face in current],
        }

    return execute
