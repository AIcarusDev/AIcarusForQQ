"""Normalize images before sending them to OpenAI-compatible vision APIs."""

from __future__ import annotations

import base64
import binascii
import io
import logging

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

logger = logging.getLogger("AICQ.llm.media.outbound_image")

def _clean_mime(mime: str) -> str:
    mime = (mime or "image/jpeg").split(";", 1)[0].strip().lower()
    return "image/jpeg" if mime == "image/jpg" else mime


def _decode_base64(b64: str) -> bytes | None:
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        logger.warning("[outbound_image] base64 解码失败: %s", exc)
        return None


def _open_image(raw: bytes) -> Image.Image | None:
    """打开图片。多帧图保留原对象以便后续 seek，单帧图深拷贝脱壳。"""
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        if getattr(img, "n_frames", 1) > 1:
            return img  # 保留原 fp 以便 seek
        copied = img.copy()
        copied.info.update(img.info)
        copied.format = img.format
        return copied
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning("[outbound_image] 图片校验失败: %s", exc)
        return None


def _mime_from_format(fmt: str | None) -> str | None:
    fmt = (fmt or "").upper()
    if fmt in {"JPEG", "JPG"}:
        return "image/jpeg"
    if fmt == "PNG":
        return "image/png"
    return None


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA") or (
        img.mode == "P" and "transparency" in img.info
    )


_MAX_EDGE = 1568  # 主流 vision API 推荐上限，超出则等比缩放
_PASSTHROUGH_MAX_BYTES = 4 * 1024 * 1024  # 4 MB 内的"干净"静态图直接透传


# ── SiliconFlow 兼容补丁开关 ─────────────────────────────────────────────────
# 开启后：所有出站图强制压成 JPEG；动图抽 4/9 帧拼九宫格。
# 关闭（默认）：仅做基础校验/必要转码，保留原 JPEG/PNG 与动图首帧。
# 由 main.py 在启动时根据 config 设置（generation.siliconflow_image_compat）。
_siliconflow_compat_enabled = False


def set_siliconflow_compat(enabled: bool) -> None:
    """开关 SiliconFlow 服务端 PIL bug 兼容补丁。"""
    global _siliconflow_compat_enabled
    _siliconflow_compat_enabled = bool(enabled)
    logger.info(
        "[outbound_image] SiliconFlow 图片兼容补丁 %s",
        "已启用" if _siliconflow_compat_enabled else "已关闭",
    )


def is_siliconflow_compat_enabled() -> bool:
    return _siliconflow_compat_enabled


# ── 动图九宫格抽帧 ───────────────────────────────────────────────────────────

def _sample_frame_indices(total: int, sample: int) -> list[int]:
    """从 [0, total) 均匀采样 sample 帧索引，首尾必含。"""
    if sample >= total:
        return list(range(total))
    if sample <= 1:
        return [0]
    step = (total - 1) / (sample - 1)
    return [min(total - 1, round(i * step)) for i in range(sample)]


def _grab_frame(animated: Image.Image, idx: int) -> Image.Image | None:
    try:
        animated.seek(idx)
        return animated.convert("RGBA").copy()
    except (EOFError, OSError, ValueError) as exc:
        logger.warning("[outbound_image] seek 帧失败 idx=%d: %s", idx, exc)
        return None


def _compose_animated_grid(animated: Image.Image, n_frames: int) -> Image.Image | None:
    """把多帧动图按 2x2 / 3x3 网格合成一张静态图，每格左上角标注帧号。"""
    sample_count = 4 if n_frames <= 6 else 9
    cols = rows = 2 if sample_count == 4 else 3
    indices = _sample_frame_indices(n_frames, sample_count)

    frames: list[tuple[int, Image.Image]] = []
    for idx in indices:
        f = _grab_frame(animated, idx)
        if f is not None:
            frames.append((idx, f))
    if not frames:
        return None

    # 单格尺寸：总图长边接近 _MAX_EDGE，留出 4px 间隙
    gap = 4
    cell_edge = max(64, (_MAX_EDGE - gap * (cols + 1)) // cols)
    cell_w = cell_h = cell_edge

    canvas_w = cols * cell_w + (cols + 1) * gap
    canvas_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (32, 32, 32, 255))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, (idx, frame) in enumerate(frames):
        # 等比缩放到单格
        scale = min(cell_w / frame.size[0], cell_h / frame.size[1])
        new_size = (max(1, int(frame.size[0] * scale)), max(1, int(frame.size[1] * scale)))
        thumb = frame.resize(new_size, Image.LANCZOS)

        col = i % cols
        row = i // cols
        x = gap + col * (cell_w + gap) + (cell_w - thumb.size[0]) // 2
        y = gap + row * (cell_h + gap) + (cell_h - thumb.size[1]) // 2
        canvas.paste(thumb, (x, y), thumb if thumb.mode == "RGBA" else None)

        # 帧号标签（黑底白字）
        label = f"#{idx + 1}/{n_frames}"
        draw = ImageDraw.Draw(canvas)
        text_xy = (gap + col * (cell_w + gap) + 4, gap + row * (cell_h + gap) + 2)
        try:
            bbox = draw.textbbox(text_xy, label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 200))
            draw.text(text_xy, label, fill=(255, 255, 255, 255), font=font)
        except Exception:
            draw.text(text_xy, label, fill=(255, 255, 255, 255), font=font)

    return canvas


def normalize_for_openai_compatible(
    b64: str,
    mime: str,
) -> tuple[str, str] | None:
    """Return ``(base64, mime)`` safe for broad OpenAI-compatible vision APIs.

    默认模式（``_siliconflow_compat_enabled = False``）：
      - 单帧 JPEG / PNG 且尺寸/体积达标 → 透传；
      - 多帧动图 → 取第一帧；
      - 其它格式 → 转码到 PNG（带 alpha）或 JPEG；
      - 长边超阈值 → 等比缩放。

    SiliconFlow 兼容模式（``_siliconflow_compat_enabled = True``）：
      - 所有输出强制 JPEG（带 alpha 合到白底）；
      - 动图按 2×2 / 3×3 网格抽 4 或 9 帧合成一张静态图，每格标注帧号。
      - 用于绕过 SiliconFlow 服务端 PIL ``verify`` bug。
    """
    raw = _decode_base64(b64)
    if raw is None:
        return None

    _ = _clean_mime(mime)
    img = _open_image(raw)
    if img is None:
        return None

    actual_mime = _mime_from_format(img.format)
    try:
        n_frames = int(getattr(img, "n_frames", 1) or 1)
    except (AttributeError, OSError):
        n_frames = 1

    if _siliconflow_compat_enabled:
        return _normalize_siliconflow(raw, img, actual_mime, n_frames, mime)
    return _normalize_default(raw, img, actual_mime, n_frames, mime)


# ── 默认模式：温和处理 ───────────────────────────────────────────────────────

def _normalize_default(
    raw: bytes,
    img: Image.Image,
    actual_mime: str | None,
    n_frames: int,
    orig_mime: str,
) -> tuple[str, str] | None:
    max_edge = max(img.size)

    # 透传：单帧 + JPEG/PNG + 尺寸/体积达标
    if (
        n_frames == 1
        and actual_mime in {"image/jpeg", "image/png"}
        and max_edge <= _MAX_EDGE
        and len(raw) <= _PASSTHROUGH_MAX_BYTES
    ):
        return base64.b64encode(raw).decode("ascii"), actual_mime

    # 多帧：取第一帧
    if n_frames > 1:
        try:
            img.seek(0)
        except (EOFError, OSError):
            pass

    # 等比缩放
    if max(img.size) > _MAX_EDGE:
        try:
            ratio = _MAX_EDGE / float(max(img.size))
            new_size = (
                max(1, int(img.size[0] * ratio)),
                max(1, int(img.size[1] * ratio)),
            )
            img = img.resize(new_size, Image.LANCZOS)
        except (OSError, ValueError) as exc:
            logger.warning("[outbound_image] 图片缩放失败: %s", exc)

    try:
        output = io.BytesIO()
        if _has_alpha(img):
            img.convert("RGBA").save(output, format="PNG", optimize=True)
            out_mime = "image/png"
        else:
            img.convert("RGB").save(
                output, format="JPEG", quality=88, optimize=True
            )
            out_mime = "image/jpeg"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(
            "[outbound_image] 图片格式转换失败 mime=%s: %s", orig_mime, exc
        )
        return None

    return base64.b64encode(output.getvalue()).decode("ascii"), out_mime


# ── SiliconFlow 兼容模式：全部压成 JPEG + 动图九宫格 ─────────────────────────

def _normalize_siliconflow(
    raw: bytes,
    img: Image.Image,
    actual_mime: str | None,
    n_frames: int,
    orig_mime: str,
) -> tuple[str, str] | None:
    max_edge = max(img.size)

    # 透传：单帧 JPEG + 尺寸/体积达标（PNG 不在此列，因服务端 PIL bug）
    if (
        n_frames == 1
        and actual_mime == "image/jpeg"
        and max_edge <= _MAX_EDGE
        and len(raw) <= _PASSTHROUGH_MAX_BYTES
    ):
        return base64.b64encode(raw).decode("ascii"), actual_mime

    # 动图九宫格
    if n_frames > 1:
        grid = _compose_animated_grid(img, n_frames)
        if grid is not None:
            img = grid
        else:
            try:
                img.seek(0)
                img = img.convert("RGBA").copy()
            except (EOFError, OSError):
                pass

    # 等比缩放
    if max(img.size) > _MAX_EDGE:
        try:
            ratio = _MAX_EDGE / float(max(img.size))
            new_size = (
                max(1, int(img.size[0] * ratio)),
                max(1, int(img.size[1] * ratio)),
            )
            img = img.resize(new_size, Image.LANCZOS)
        except (OSError, ValueError) as exc:
            logger.warning("[outbound_image] 图片缩放失败: %s", exc)

    try:
        output = io.BytesIO()
        if _has_alpha(img):
            rgba = img.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[3])
            background.save(output, format="JPEG", quality=88, optimize=True)
        else:
            img.convert("RGB").save(
                output, format="JPEG", quality=88, optimize=True
            )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        logger.warning(
            "[outbound_image] 图片格式转换失败 mime=%s: %s", orig_mime, exc
        )
        return None

    return base64.b64encode(output.getvalue()).decode("ascii"), "image/jpeg"


def make_data_url(b64: str, mime: str) -> str | None:
    """Build a data URL after normalizing the image payload."""
    normalized = normalize_for_openai_compatible(b64, mime)
    if normalized is None:
        return None
    out_b64, out_mime = normalized
    return f"data:{out_mime};base64,{out_b64}"
