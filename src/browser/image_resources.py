"""Browser image resource observation and immutable send artifact storage.

``resource_ref`` identifies a lightweight browser observation.  It is never
sendable.  ``image_ref`` identifies validated, content-addressed original
bytes and is the only browser-image reference accepted by a send adapter.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from PIL import Image


MAX_BROWSER_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BROWSER_IMAGE_PIXELS = 100_000_000
_RESOURCE_REF_RE = re.compile(r"^br_[0-9a-f]{20}$")
_IMAGE_REF_RE = re.compile(r"^img_[0-9a-f]{32}$")
_FORMAT_MIME = {
    "AVIF": "image/avif",
    "BMP": "image/bmp",
    "GIF": "image/gif",
    "ICO": "image/x-icon",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_MIME_EXTENSION = {
    "image/avif": ".avif",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/x-icon": ".ico",
}


class BrowserImageError(RuntimeError):
    """Base class for explicit browser-image pipeline failures."""


class BrowserResourceNotFound(BrowserImageError):
    """The selected resource is no longer present in the registry."""


class BrowserImageValidationError(BrowserImageError):
    """Materialized bytes failed a hard security or integrity check."""


@dataclass(frozen=True, slots=True)
class NetworkObservation:
    request_id: str
    frame_id: str
    source_url: str
    final_url: str
    mime: str
    status: int
    observed_at: float
    from_disk_cache: bool = False
    from_service_worker: bool = False


@dataclass(frozen=True, slots=True)
class BrowserImageResource:
    resource_ref: str
    source_url: str
    page_url: str
    request_id: str
    frame_id: str
    observed_at: float
    alt: str
    rect: dict[str, int]
    natural_size: tuple[int, int]
    mime: str = ""
    final_url: str = ""

    def model_projection(self, source_url_mode: str = "full") -> dict[str, Any]:
        projection: dict[str, Any] = {
            "resource_ref": self.resource_ref,
            "alt": self.alt,
            "rect": dict(self.rect),
            "natural_size": [self.natural_size[0], self.natural_size[1]],
        }
        source_url = project_source_url(self.source_url, source_url_mode)
        if source_url:
            projection["source_url"] = source_url
        return projection


@dataclass(frozen=True, slots=True)
class MaterializedBrowserImage:
    image_ref: str
    mime: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    strategy: str
    source_url: str
    final_url: str
    resource_ref: str
    confirmation_reasons: tuple[str, ...] = field(default_factory=tuple)

    def public_result(self) -> dict[str, Any]:
        return {
            "image_ref": self.image_ref,
            "mime": self.mime,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "resource_ref": self.resource_ref,
            "sha256": self.sha256,
            "strategy": self.strategy,
            "confirmation_reasons": list(self.confirmation_reasons),
        }


def normalize_source_url_mode(value: object) -> str:
    mode = str(value or "full").strip().lower().replace("-", "_")
    return mode if mode in {"hidden", "sanitized", "full"} else "full"


def project_source_url(source_url: str, mode: str) -> str:
    mode = normalize_source_url_mode(mode)
    if mode == "hidden":
        return ""
    try:
        parsed = urlsplit(str(source_url or ""))
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            return ""
        safe_netloc = f"{hostname}{port}"
        if mode == "sanitized":
            return urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))
        sensitive = re.compile(
            r"(?:^|[_-])(token|key|auth|authorization|signature|sig|password|"
            r"credential|secret|session)(?:$|[_-])",
            re.IGNORECASE,
        )
        query = urlencode([
            (key, "<redacted>" if sensitive.search(key) else value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        ])
        return urlunsplit((parsed.scheme, safe_netloc, parsed.path, query, ""))
    except ValueError:
        return ""


class BrowserImageResourceRegistry:
    """In-memory browser observations; no response body is retained here."""

    def __init__(self, *, max_resources: int = 2000) -> None:
        self._max_resources = max(32, int(max_resources))
        self._resources: dict[str, BrowserImageResource] = {}
        self._observations_by_url: dict[str, NetworkObservation] = {}
        self._request_initial_url: dict[str, str] = {}
        self._lock = threading.RLock()

    def observe_request(self, request_id: object, url: object) -> None:
        request = str(request_id or "").strip()
        source_url = str(url or "").strip()
        if not request or not source_url:
            return
        with self._lock:
            self._request_initial_url.setdefault(request, source_url)
            while len(self._request_initial_url) > self._max_resources * 2:
                self._request_initial_url.pop(next(iter(self._request_initial_url)))

    def observe_response(
        self,
        *,
        request_id: object,
        frame_id: object,
        url: object,
        mime: object,
        status: object,
        from_disk_cache: object = False,
        from_service_worker: object = False,
        observed_at: float | None = None,
    ) -> None:
        request = str(request_id or "").strip()
        final_url = str(url or "").strip()
        normalized_mime = str(mime or "").split(";", 1)[0].strip().lower()
        if not request or not final_url or not normalized_mime.startswith("image/"):
            return
        try:
            status_code = int(status or 0)
        except (TypeError, ValueError):
            status_code = 0
        with self._lock:
            source_url = self._request_initial_url.get(request, final_url)
            observation = NetworkObservation(
                request_id=request,
                frame_id=str(frame_id or ""),
                source_url=source_url,
                final_url=final_url,
                mime=normalized_mime,
                status=status_code,
                observed_at=float(observed_at if observed_at is not None else time.time()),
                from_disk_cache=bool(from_disk_cache),
                from_service_worker=bool(from_service_worker),
            )
            self._observations_by_url[source_url] = observation
            self._observations_by_url[final_url] = observation
            while len(self._observations_by_url) > self._max_resources * 2:
                self._observations_by_url.pop(next(iter(self._observations_by_url)))

    @staticmethod
    def _resource_ref(page_url: str, source_url: str, identity: str) -> str:
        digest = hashlib.sha256(
            f"{page_url}\0{source_url}\0{identity}".encode("utf-8", errors="replace")
        ).hexdigest()
        return f"br_{digest[:20]}"

    def register(
        self,
        *,
        source_url: object,
        page_url: object,
        identity: object,
        alt: object,
        rect: object,
        natural_size: object,
    ) -> BrowserImageResource | None:
        source = str(source_url or "").strip()
        page = str(page_url or "").strip()
        if not source or not page:
            return None
        if not isinstance(rect, dict):
            return None
        try:
            normalized_rect = {
                key: int(round(float(rect.get(key) or 0)))
                for key in ("x", "y", "width", "height")
            }
            if isinstance(natural_size, (list, tuple)) and len(natural_size) >= 2:
                normalized_size = (int(natural_size[0] or 0), int(natural_size[1] or 0))
            else:
                normalized_size = (0, 0)
        except (TypeError, ValueError):
            return None
        ref = self._resource_ref(page, source, str(identity or ""))
        with self._lock:
            observation = self._observations_by_url.get(source)
            resource = BrowserImageResource(
                resource_ref=ref,
                source_url=source,
                page_url=page,
                request_id=observation.request_id if observation else "",
                frame_id=observation.frame_id if observation else "",
                observed_at=observation.observed_at if observation else time.time(),
                alt=str(alt or ""),
                rect=normalized_rect,
                natural_size=normalized_size,
                mime=observation.mime if observation else "",
                final_url=observation.final_url if observation else source,
            )
            self._resources.pop(ref, None)
            self._resources[ref] = resource
            while len(self._resources) > self._max_resources:
                self._resources.pop(next(iter(self._resources)))
            return resource

    def get(self, resource_ref: object) -> BrowserImageResource:
        ref = str(resource_ref or "").strip()
        if not _RESOURCE_REF_RE.fullmatch(ref):
            raise BrowserResourceNotFound("invalid browser resource_ref")
        with self._lock:
            resource = self._resources.get(ref)
        if resource is None:
            raise BrowserResourceNotFound(f"browser resource is unavailable: {ref}")
        return resource

    def project(
        self,
        resource_refs: Iterable[str],
        *,
        source_url_mode: str = "full",
    ) -> list[dict[str, Any]]:
        return [
            self.get(resource_ref).model_projection(source_url_mode)
            for resource_ref in resource_refs
        ]


class BrowserImageArtifactStore:
    """Content-addressed, immutable, sendable browser-image artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    @staticmethod
    def _inspect(data: bytes) -> tuple[str, int, int, int]:
        try:
            with Image.open(io.BytesIO(data)) as image:
                detected_format = str(image.format or "").upper()
                mime = _FORMAT_MIME.get(detected_format, "")
                width, height = (int(image.width), int(image.height))
                frame_count = int(getattr(image, "n_frames", 1) or 1)
                image.verify()
        except Exception as exc:
            raise BrowserImageValidationError(
                f"materialized bytes are not a valid supported raster image: {exc}"
            ) from exc
        if not mime:
            raise BrowserImageValidationError(
                f"unsupported browser image format: {detected_format or 'unknown'}"
            )
        if width <= 0 or height <= 0 or width * height > MAX_BROWSER_IMAGE_PIXELS:
            raise BrowserImageValidationError("browser image dimensions exceed the safety limit")
        return mime, width, height, frame_count

    @staticmethod
    def _semantic_reasons(
        *,
        resource: BrowserImageResource,
        actual_mime: str,
        declared_mime: str,
        width: int,
        height: int,
        frame_count: int,
        strategy: str,
        final_url: str,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if strategy == "browser_network":
            reasons.append("resource_identity_unproven")
        if final_url and final_url != resource.source_url:
            reasons.append("redirect_target_changed")
            if resource.source_url.startswith("http://") and final_url.startswith("https://"):
                reasons.append("source_url_upgraded")
        normalized_declared = declared_mime.split(";", 1)[0].strip().lower()
        if normalized_declared and normalized_declared != actual_mime:
            reasons.append("mime_or_content_form_changed")
        natural_width, natural_height = resource.natural_size
        if natural_width > 0 and natural_height > 0:
            natural_ratio = natural_width / natural_height
            actual_ratio = width / height
            if abs(actual_ratio / natural_ratio - 1.0) > 0.05:
                reasons.append("aspect_ratio_changed")
        rect_width = int(resource.rect.get("width") or 0)
        rect_height = int(resource.rect.get("height") or 0)
        if rect_width < 96 or rect_height < 96 or rect_width * rect_height < 9216:
            reasons.append("very_small_preview")
        if rect_width > 0 and rect_height > 0 and width > 0 and height > 0:
            visible_ratio = rect_width / rect_height
            actual_ratio = width / height
            if abs(visible_ratio / actual_ratio - 1.0) > 0.08:
                reasons.append("visible_crop_or_partial_preview")
        if frame_count > 1:
            reasons.append("animated_or_multiframe")
        return tuple(dict.fromkeys(reasons))

    def persist(
        self,
        data: bytes,
        *,
        resource: BrowserImageResource,
        strategy: str,
        declared_mime: str,
        final_url: str = "",
    ) -> MaterializedBrowserImage:
        if not data:
            raise BrowserImageValidationError("materialized browser image is empty")
        if len(data) > MAX_BROWSER_IMAGE_BYTES:
            raise BrowserImageValidationError("browser image exceeds the 20 MiB safety limit")
        normalized_declared = str(declared_mime or "").split(";", 1)[0].strip().lower()
        if normalized_declared and not normalized_declared.startswith("image/"):
            raise BrowserImageValidationError("browser response MIME is not an image")
        actual_mime, width, height, frame_count = self._inspect(data)
        digest = hashlib.sha256(data).hexdigest()
        image_ref = f"img_{digest[:32]}"
        reasons = self._semantic_reasons(
            resource=resource,
            actual_mime=actual_mime,
            declared_mime=normalized_declared,
            width=width,
            height=height,
            frame_count=frame_count,
            strategy=strategy,
            final_url=final_url or resource.final_url or resource.source_url,
        )
        artifact = MaterializedBrowserImage(
            image_ref=image_ref,
            mime=actual_mime,
            size_bytes=len(data),
            width=width,
            height=height,
            sha256=digest,
            strategy=strategy,
            source_url=resource.source_url,
            final_url=final_url or resource.final_url or resource.source_url,
            resource_ref=resource.resource_ref,
            confirmation_reasons=reasons,
        )
        extension = _MIME_EXTENSION[actual_mime]
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            data_path = self.root / f"{image_ref}{extension}"
            manifest_path = self.root / f"{image_ref}.json"
            if data_path.exists() and data_path.read_bytes() != data:
                raise BrowserImageValidationError("browser image_ref collision")
            if manifest_path.exists():
                if self.read(image_ref) is None:
                    raise BrowserImageValidationError(
                        "existing browser image artifact failed integrity validation"
                    )
                return artifact
            if not data_path.exists():
                temp_path = self.root / f".{image_ref}.{os.getpid()}.tmp"
                temp_path.write_bytes(data)
                temp_path.replace(data_path)
            manifest = {
                **asdict(artifact),
                "confirmation_reasons": list(artifact.confirmation_reasons),
                "file": data_path.name,
                "created_at": time.time(),
            }
            encoded = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            temp_manifest = self.root / f".{image_ref}.{os.getpid()}.json.tmp"
            temp_manifest.write_text(encoded, encoding="utf-8")
            temp_manifest.replace(manifest_path)
        return artifact

    def read(self, image_ref: object) -> tuple[bytes, str, dict[str, Any]] | None:
        ref = str(image_ref or "").strip()
        if not _IMAGE_REF_RE.fullmatch(ref):
            return None
        manifest_path = self.root / f"{ref}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            filename = str(manifest["file"])
            data_path = self.root / filename
            if data_path.parent.resolve() != self.root.resolve() or not data_path.name.startswith(f"{ref}."):
                return None
            data = data_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != str(manifest.get("sha256") or ""):
                return None
            if ref != f"img_{digest[:32]}":
                return None
            mime, width, height, _frame_count = self._inspect(data)
            if (
                mime != str(manifest.get("mime") or "")
                or width != int(manifest.get("width") or 0)
                or height != int(manifest.get("height") or 0)
                or len(data) != int(manifest.get("size_bytes") or 0)
            ):
                return None
            return data, mime, manifest
        except (OSError, ValueError, TypeError, KeyError, BrowserImageValidationError):
            return None


__all__ = [
    "BrowserImageArtifactStore",
    "BrowserImageError",
    "BrowserImageResource",
    "BrowserImageResourceRegistry",
    "BrowserImageValidationError",
    "BrowserResourceNotFound",
    "MaterializedBrowserImage",
    "MAX_BROWSER_IMAGE_BYTES",
    "normalize_source_url_mode",
    "project_source_url",
]
