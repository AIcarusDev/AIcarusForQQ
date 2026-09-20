"""Validation shared by settings and runtime video clients."""
import math

DEFAULT_MAX_SIZE_MB = 20.0
HARD_MAX_SIZE_MB = 128.0
DEFAULT_TIMEOUT_SECONDS = 120.0
PROTOCOLS = {"auto", "gemini_native", "openai_compatible"}


def validate_video_settings(settings: dict) -> dict:
    result = dict(settings)
    protocol = str(result.get("protocol", "auto")).strip().lower()
    if protocol not in PROTOCOLS:
        raise ValueError("视频调用协议无效")
    result["protocol"] = protocol
    for key, default, low, high in (
        ("max_size_mb", DEFAULT_MAX_SIZE_MB, 1, HARD_MAX_SIZE_MB),
        ("timeout", DEFAULT_TIMEOUT_SECONDS, 10, 600),
    ):
        try:
            value = float(result.get(key, default))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"视频 {key} 必须是数字") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"视频 {key} 必须在 {low}~{high} 范围内")
        result[key] = value
    return result
