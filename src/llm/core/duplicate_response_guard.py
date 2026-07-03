"""Duplicate full-response guard for model outputs.

The guard is intentionally conservative: it only treats normalized full raw
assistant responses as duplicates when they are exactly equal.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import re
from typing import Any


PASSIVE_DUPLICATE_TOOL_NAMES = frozenset({"runtime_manage"})

COGNITION_PREFILL_POOL: tuple[str, ...] = (
    "啊，刚刚好像有点走神了，让我仔细看看当前的情况",
    "嗯，我先把当前的外界情况和刚才的动作结果对齐一下",
    "我先理一理目前的情况，看看我刚才做的动作和结果",
    "嗯，我先停一下，重新核对当前上下文和刚才的动作结果",
    "刚刚感觉恍惚了一下，让我仔细看看目前的情况",
    "好像分心了一下，现在回过神来了。我看看现在是什么情况",
    "好，我确认一下我刚刚动作的返回结果",
    "我先确认一下目前发生了什么、我刚做了什么",
    "我看看当前最新上下文",
    "我先看一下目前的外界情况和刚才的动作结果",
)


@dataclass(frozen=True)
class DuplicateModelResponseGuardConfig:
    enabled: bool = False
    lookback_rounds: int = 3
    max_retries: int = 2
    normalize_whitespace: bool = True
    fallback_sleep_minutes: int = 30


@dataclass(frozen=True)
class CognitionPrefillGuidanceConfig:
    enabled: bool = True
    lookback_rounds: int = 8
    similarity_threshold: float = 0.9
    min_chars: int = 80
    max_retries: int = 2


class CognitionPrefillRetrySignal(Exception):
    """Raised while streaming when a cognition block repeats visible cognition."""

    stream_abort = True

    def __init__(
        self,
        *,
        cognition: str,
        similarity: float,
        matched_index: int,
        matched_cognition: str,
    ) -> None:
        super().__init__("repeated cognition detected before action")
        self.cognition = cognition
        self.similarity = similarity
        self.matched_index = matched_index
        self.matched_cognition = matched_cognition


def normalize_duplicate_model_response_guard_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    cfg = DuplicateModelResponseGuardConfig()

    def _int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    return {
        "enabled": bool(raw.get("enabled", cfg.enabled)),
        "lookback_rounds": _int("lookback_rounds", cfg.lookback_rounds, 1, 20),
        "max_retries": _int("max_retries", cfg.max_retries, 1, 10),
        "normalize_whitespace": bool(raw.get("normalize_whitespace", cfg.normalize_whitespace)),
        "fallback_sleep_minutes": _int("fallback_sleep_minutes", cfg.fallback_sleep_minutes, 1, 600),
        "prefill_guidance": normalize_cognition_prefill_guidance_config(
            raw.get("prefill_guidance")
        ),
    }


def normalize_cognition_prefill_guidance_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    cfg = CognitionPrefillGuidanceConfig()

    def _int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    def _float(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    return {
        "enabled": bool(raw.get("enabled", cfg.enabled)),
        "lookback_rounds": _int("lookback_rounds", cfg.lookback_rounds, 1, 20),
        "similarity_threshold": _float(
            "similarity_threshold",
            cfg.similarity_threshold,
            0.5,
            1.0,
        ),
        "min_chars": _int("min_chars", cfg.min_chars, 20, 2000),
        "max_retries": _int("max_retries", cfg.max_retries, 1, 10),
    }


def normalize_response_text(text: str, *, normalize_whitespace: bool = True) -> str:
    normalized = (text or "").strip()
    if normalize_whitespace:
        normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def is_passive_duplicate_tool_set(tool_names: list[str] | tuple[str, ...]) -> bool:
    """Return true when repeating the tool set is a benign wait-style action."""
    if not tool_names:
        return False
    return all(str(name or "").strip() in PASSIVE_DUPLICATE_TOOL_NAMES for name in tool_names)


def cognition_prefill_provider_supported(provider: str, model: str = "") -> bool:
    """Return whether assistant-prefill retry is allowed for this provider."""
    combined = f"{provider or ''} {model or ''}".lower()
    return "gemini" not in combined


def normalize_cognition_text(text: str) -> str:
    """Normalize cognition text for strict visible-cognition comparison."""
    return re.sub(r"\s+", "", (text or "").strip())


def cognition_similarity(left: str, right: str) -> float:
    left_norm = normalize_cognition_text(left)
    right_norm = normalize_cognition_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm, autojunk=False).ratio()


def find_repeated_visible_cognition(
    cognition: str,
    visible_cognitions: list[str] | tuple[str, ...],
    *,
    similarity_threshold: float,
    min_chars: int,
) -> dict[str, Any] | None:
    cognition_norm = normalize_cognition_text(cognition)
    if len(cognition_norm) < min_chars:
        return None

    best: dict[str, Any] | None = None
    for index, visible in enumerate(visible_cognitions):
        visible_norm = normalize_cognition_text(visible)
        if not visible_norm:
            continue
        score = cognition_similarity(cognition_norm, visible_norm)
        if best is None or score > best["similarity"]:
            best = {
                "similarity": score,
                "matched_index": index,
                "matched_cognition": visible,
            }
    if best is not None and best["similarity"] >= similarity_threshold:
        return best
    return None


def choose_cognition_prefill(
    visible_cognitions: list[str] | tuple[str, ...],
    *,
    used_prefills: list[str] | tuple[str, ...] = (),
    seed_text: str = "",
) -> str:
    visible_norms = {
        normalize_cognition_text(text)
        for text in visible_cognitions
        if normalize_cognition_text(text)
    }
    used_norms = {
        normalize_cognition_text(text)
        for text in used_prefills
        if normalize_cognition_text(text)
    }
    pool = COGNITION_PREFILL_POOL
    if not pool:
        return ""
    digest = hashlib.sha1((seed_text or "").encode("utf-8")).hexdigest()
    start = int(digest[:8], 16) % len(pool)
    ordered = (*pool[start:], *pool[:start])
    for candidate in ordered:
        norm = normalize_cognition_text(candidate)
        if norm and norm not in visible_norms and norm not in used_norms:
            return candidate
    for candidate in ordered:
        norm = normalize_cognition_text(candidate)
        if norm and norm not in visible_norms:
            return candidate
    return ordered[0]


def format_cognition_prefill(prefill_body: str) -> str:
    body = (prefill_body or "").strip()
    if not body:
        return ""
    return "<cognition>\n" + body


class CognitionRepeatStreamGuard:
    """Detect repeated cognition immediately after a streamed cognition block closes."""

    _TAG_RE = re.compile(r"^</?\s*([a-zA-Z_][\w:-]*)")

    def __init__(
        self,
        *,
        visible_cognitions: list[str] | tuple[str, ...],
        similarity_threshold: float,
        min_chars: int,
    ) -> None:
        self.visible_cognitions = tuple(visible_cognitions or ())
        self.similarity_threshold = similarity_threshold
        self.min_chars = min_chars
        self._mode = "outside"
        self._tag_buf = ""
        self._cognition_buf: list[str] = []

    def feed(self, text: str) -> None:
        for ch in text or "":
            if self._tag_buf:
                self._tag_buf += ch
                if ch == ">":
                    self._handle_tag(self._tag_buf)
                    self._tag_buf = ""
                continue

            if ch == "<":
                self._tag_buf = "<"
                continue

            if self._mode == "cognition":
                self._cognition_buf.append(ch)

    def _handle_tag(self, raw_tag: str) -> None:
        normalized = raw_tag.strip().lower()
        match = self._TAG_RE.match(normalized)
        name = match.group(1) if match else ""
        is_close = normalized.startswith("</")
        if name != "cognition":
            if self._mode == "cognition":
                self._cognition_buf.append(raw_tag)
            return

        if is_close:
            cognition = "".join(self._cognition_buf).strip()
            self._mode = "outside"
            self._cognition_buf = []
            repeated = find_repeated_visible_cognition(
                cognition,
                self.visible_cognitions,
                similarity_threshold=self.similarity_threshold,
                min_chars=self.min_chars,
            )
            if repeated is not None:
                raise CognitionPrefillRetrySignal(
                    cognition=cognition,
                    similarity=float(repeated["similarity"]),
                    matched_index=int(repeated["matched_index"]),
                    matched_cognition=str(repeated["matched_cognition"]),
                )
            return

        self._mode = "cognition"
        self._cognition_buf = []


def build_duplicate_model_response_error(*, duplicate_count: int, max_retries: int) -> dict[str, Any]:
    return {
        "error": "DUPLICATE_MODEL_RESPONSE",
        "message": (
            "本轮模型输出与最近一次模型输出完全一致，包括 cognition 和 tool_call。"
            "系统未执行其中的工具。请重新评估当前 world，避免重复执行已完成的行为。"
        ),
        "tool_not_executed": True,
        "retryable": True,
        "duplicate_count": duplicate_count,
        "max_retries": max_retries,
    }


def build_duplicate_model_response_limit_error(*, duplicate_count: int) -> dict[str, Any]:
    return {
        "error": "DUPLICATE_MODEL_RESPONSE_LIMIT",
        "message": "模型连续输出完全相同内容，已停止重试并进入 runtime_manage.sleep，等待新的外部输入。",
        "tool_not_executed": True,
        "retryable": False,
        "duplicate_count": duplicate_count,
        "fallback": "runtime_manage.sleep",
    }
