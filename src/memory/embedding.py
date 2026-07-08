"""Memory embedding helpers.

The default client is deterministic and local.  It gives memory recall a stable
interface and testable vector path without introducing a network dependency.
External embedding providers can replace it behind ``MemoryEmbeddingClient``.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI


class EmbeddingError(RuntimeError):
    pass


class EmbeddingConfigError(EmbeddingError):
    pass


class EmbeddingProviderError(EmbeddingError):
    pass


class EmbeddingInvalidResponseError(EmbeddingError):
    pass


class EmbeddingUnsupportedDimensionError(EmbeddingError):
    pass


@dataclass(slots=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    model: str
    model_version: str
    dim: int
    normalized: bool


class MemoryEmbeddingClient:
    """Ordered batch embedding interface used by memory."""

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        raise NotImplementedError


class HashEmbeddingClient(MemoryEmbeddingClient):
    """Small deterministic embedding fallback.

    This is not intended to be semantically strong.  It keeps the vector storage,
    stale detection, and recall plumbing operational until a real provider is
    configured.
    """

    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise EmbeddingUnsupportedDimensionError(f"invalid dimension: {dim}")
        self.dim = int(dim)
        self.model = "local-hash-embedding"
        self.model_version = "v1"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        vectors: list[list[float]] = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingInvalidResponseError("empty text cannot be embedded")
            vectors.append(_hash_embed(text, self.dim))
        return EmbeddingBatch(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
            dim=self.dim,
            normalized=True,
        )


class OpenAICompatEmbeddingClient(MemoryEmbeddingClient):
    """OpenAI-compatible embeddings client."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        provider: str = "openai-compatible",
        model_version: str = "",
    ) -> None:
        if not model:
            raise EmbeddingConfigError("embedding model is required")
        if not base_url:
            raise EmbeddingConfigError("embedding base_url is required")
        proxy_url = os.getenv("OPENAI_PROXY", "").strip()
        kwargs: dict[str, Any] = {"api_key": api_key or "openai-compat", "base_url": base_url}
        if proxy_url:
            kwargs["http_client"] = httpx.Client(proxy=proxy_url)
        self.client = OpenAI(**kwargs)
        self.model = model
        self.model_version = model_version
        self.provider = provider

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(
                vectors=[],
                model=self.model,
                model_version=self.model_version,
                dim=0,
                normalized=True,
            )
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingInvalidResponseError("empty text cannot be embedded")
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise EmbeddingProviderError(str(exc)) from exc
        data = list(getattr(response, "data", []) or [])
        data.sort(key=lambda item: int(getattr(item, "index", 0)))
        if len(data) != len(texts):
            raise EmbeddingInvalidResponseError(
                f"embedding response length {len(data)} != input length {len(texts)}"
            )
        vectors: list[list[float]] = []
        dim = 0
        for item in data:
            raw = list(getattr(item, "embedding", []) or [])
            if not raw:
                raise EmbeddingInvalidResponseError("empty embedding vector")
            if not dim:
                dim = len(raw)
            elif len(raw) != dim:
                raise EmbeddingUnsupportedDimensionError("inconsistent embedding dimensions")
            vectors.append(normalize([float(x) for x in raw]))
        return EmbeddingBatch(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
            dim=dim,
            normalized=True,
        )


def build_embedding_client(cfg: dict[str, Any] | None) -> MemoryEmbeddingClient:
    """Build a memory embedding client from Memory config.

    Supported shapes:
    - ``{"provider": "hash"}`` or missing config: local deterministic fallback.
    - ``{"provider": "...", "model": "..."}``: resolve provider from
      top-level ``model_providers`` when present.
    - explicit ``base_url`` and ``api_key_env`` can be supplied directly under
      the embedding config.
    """

    cfg = dict(cfg or {})
    provider = str(cfg.get("provider") or "hash").strip()
    if not provider or provider == "hash":
        return HashEmbeddingClient(dim=int(cfg.get("dim", 128) or 128))
    model = str(cfg.get("model") or "").strip()
    base_url = str(cfg.get("base_url") or "").strip()
    api_key_env = str(cfg.get("api_key_env") or "").strip()
    requires_api_key = bool(cfg.get("requires_api_key", True))
    provider_cfg = cfg.get("provider_config")
    if isinstance(provider_cfg, dict):
        base_url = base_url or str(provider_cfg.get("base_url") or "").strip()
        api_key_env = api_key_env or str(provider_cfg.get("api_key_env") or "").strip()
        requires_api_key = bool(provider_cfg.get("requires_api_key", requires_api_key))
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if not api_key and not requires_api_key:
        api_key = "openai-compat"
    return OpenAICompatEmbeddingClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        provider=provider,
        model_version=str(cfg.get("model_version") or ""),
    )


def source_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(vector), *vector)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    if dim <= 0:
        return []
    expected = dim * 4
    if len(blob) != expected:
        raise EmbeddingInvalidResponseError(
            f"vector blob length {len(blob)} does not match dim {dim}"
        )
    return list(struct.unpack("<" + "f" * dim, blob))


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 0.0:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _hash_embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    lowered = text.strip().lower()
    tokens = _features(lowered)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[bucket] += sign
    return normalize(vec)


def _features(text: str) -> list[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    feats: list[str] = []
    parts = [p for p in text.replace("_", " ").replace("-", " ").split() if p]
    feats.extend(parts)
    if compact:
        feats.append(compact)
    for n in (2, 3):
        if len(compact) >= n:
            feats.extend(compact[i : i + n] for i in range(len(compact) - n + 1))
    return feats or [text]
