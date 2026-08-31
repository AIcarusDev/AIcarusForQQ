"""Linux-shaped logical paths shared by both QQ file storage backends."""

from __future__ import annotations

import re
import unicodedata
from pathlib import PurePosixPath


AGENT_HOME = PurePosixPath("/home/agent")
QQ_ROOT = AGENT_HOME / "qq"
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_INTERNAL_TEMP_PREFIX = ".aicq-qq-file-"


class LogicalPathError(ValueError):
    pass


def _truncate_utf8(value: str, byte_limit: int) -> str:
    if byte_limit <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= byte_limit:
        return value
    return encoded[:byte_limit].decode("utf-8", errors="ignore")


def require_agent_qq(value: object) -> str:
    qq = str(value or "").strip()
    if not qq or not qq.isdigit():
        raise LogicalPathError("当前 QQ 账号标识不可用")
    return qq


def account_file_root(agent_qq: str) -> PurePosixPath:
    return QQ_ROOT / require_agent_qq(agent_qq) / "file"


def conversation_root(agent_qq: str, conversation_type: str, conversation_id: str) -> PurePosixPath:
    conv_type = str(conversation_type or "").strip()
    conv_id = str(conversation_id or "").strip()
    if conv_type not in {"private", "group"} or not conv_id or "/" in conv_id or conv_id in {".", ".."}:
        raise LogicalPathError("QQ 会话标识无效")
    return account_file_root(agent_qq) / f"{conv_type}_{conv_id}"


def sanitize_filename(value: object) -> str:
    name = str(value or "")
    name = name.replace("/", "_").replace("\\", "_")
    name = _CONTROL_RE.sub("_", name).strip()
    if name in {"", ".", ".."}:
        name = "file"
    if name.startswith(_INTERNAL_TEMP_PREFIX):
        name = "_" + name
    if len(name.encode("utf-8")) <= 255:
        return name
    suffix = PurePosixPath(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    suffix_bytes = len(suffix.encode("utf-8"))
    if suffix and suffix_bytes < 251:
        fitted_stem = _truncate_utf8(stem, 255 - suffix_bytes)
        if fitted_stem:
            return fitted_stem + suffix
    return _truncate_utf8(name, 255) or "file"


def collision_name(filename: str, index: int) -> str:
    if index <= 0:
        return filename
    path = PurePosixPath(filename)
    suffix = path.suffix
    stem = filename[: -len(suffix)] if suffix else filename
    marker = f"({index})"
    suffix_limit = max(0, 255 - len(marker.encode("utf-8")) - 1)
    fitted_suffix = _truncate_utf8(suffix, suffix_limit)
    stem_limit = 255 - len(marker.encode("utf-8")) - len(fitted_suffix.encode("utf-8"))
    fitted_stem = _truncate_utf8(stem, stem_limit) or "f"
    return f"{fitted_stem}{marker}{fitted_suffix}"


def validate_logical_path(path: object, agent_qq: str, *, allow_root: bool = False) -> PurePosixPath:
    raw = str(path or "")
    if not raw.startswith("/") or "\x00" in raw or "\\" in raw or raw.startswith("//"):
        raise LogicalPathError("必须提供绝对 Linux 路径")
    pure = PurePosixPath(raw)
    if any(part in {".", ".."} for part in raw.split("/")):
        raise LogicalPathError("路径包含不允许的组件")
    root = account_file_root(agent_qq)
    try:
        relative = pure.relative_to(root)
    except ValueError as exc:
        raise LogicalPathError("路径不在当前 QQ 账号的文件根目录内") from exc
    if not allow_root and not relative.parts:
        raise LogicalPathError("必须提供文件路径")
    return pure


def agent_home_parts(path: PurePosixPath) -> tuple[str, ...]:
    try:
        return tuple(path.relative_to(AGENT_HOME).parts)
    except ValueError as exc:
        raise LogicalPathError("路径不在 /home/agent 内") from exc


def extension_for(name: str) -> str | None:
    suffix = PurePosixPath(name).suffix
    if not suffix or suffix == name:
        return None
    return suffix[1:].casefold()


def normalized_filename(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()
