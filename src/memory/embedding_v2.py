"""Memory V2 embedding helpers.

The default client is deterministic and local.  It gives V2 recall a stable
interface and testable vector path without introducing a network dependency.
External embedding providers can replace it behind ``MemoryEmbeddingClient``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass


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
    """Ordered batch embedding interface used by memory V2."""

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

