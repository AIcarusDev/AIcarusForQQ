"""Shared Playwright browser session for browser_control.

The module keeps a single persistent Chromium context so browser tools can
share cookies, history, scroll position, and response-cached images.
"""

from __future__ import annotations

import base64
import queue
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urljoin, urlparse

from browser.config import (
    DEFAULT_BROWSER_RESULT_LIMITS,
    browser_screenshot_annotations_enabled as _config_browser_screenshot_annotations_enabled,
    normalize_browser_control_config,
)

logger = logging.getLogger("AICQ.browser")
T = TypeVar("T")

ROOT = Path(__file__).resolve().parents[2]
BROWSER_IMAGE_DIR = ROOT / "cache" / "browser_image"
BROWSER_PROFILE_DIR = ROOT / "cache" / "browser_profile"

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/avif": ".avif",
    "image/vnd.microsoft.icon": ".ico",
}

_BROWSER_RESULT_LIMITS = dict(DEFAULT_BROWSER_RESULT_LIMITS)


@dataclass
class BrowserImage:
    ref: str
    url: str
    path: str
    mime: str
    size_bytes: int
    sha256: str
    page_url: str


@dataclass
class BrowserActivity:
    timestamp_ms: int
    action: str
    url: str
    title: str
    events: list[str]
    viewport_image_ref: str
    cached_images_count: int
    visible_images_count: int
    click_targets_count: int


_ACTIVITY_HISTORY: list[BrowserActivity] = []
_LATEST_VIEWPORT_REF: str = ""


def _image_extension(mime: str, url: str) -> str:
    mime = mime.split(";", 1)[0].strip().lower()
    if mime in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[mime]
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".ico"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return mimetypes.guess_extension(mime) or ".bin"


def _write_browser_image(ref: str, data: bytes, ext: str) -> Path:
    BROWSER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = BROWSER_IMAGE_DIR / f"{ref}{ext}"
    path.write_bytes(data)
    return path


def _resize_png(png_bytes: bytes, max_side: int = 1280) -> bytes:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        width, height = img.size
        if width <= max_side and height <= max_side:
            return png_bytes
        ratio = max_side / max(width, height)
        img = img.resize((int(width * ratio), int(height * ratio)), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return png_bytes


def _normalize_load_state(wait_until: str | None) -> str:
    value = str(wait_until or "domcontentloaded").strip().lower()
    return value if value in {"domcontentloaded", "load", "networkidle", "commit"} else "domcontentloaded"


def _limit(key: str) -> int:
    return int(_BROWSER_RESULT_LIMITS[key])


def configure_browser_result_limits(config: dict[str, Any] | None) -> None:
    cfg = config if isinstance(config, dict) else {}
    browser_cfg = cfg.get("browser_control")
    normalized = normalize_browser_control_config(
        browser_cfg if isinstance(browser_cfg, dict) else {}
    )
    _BROWSER_RESULT_LIMITS.update(normalized["result_limits"])


def _shorten(value: object, limit: int) -> str:
    if limit <= 0:
        return ""
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 16)] + "...(truncated)"


def _compact_url(value: object) -> str:
    return _shorten(value, _limit("url_chars"))


def _browser_screenshot_annotations_enabled() -> bool:
    try:
        import app_state

        return _config_browser_screenshot_annotations_enabled(getattr(app_state, "config", {}) or {})
    except Exception:
        return False


class BrowserSession:
    def __init__(self) -> None:
        self.owner_thread_id = threading.get_ident()
        self._pw: Any | None = None
        self._browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.profile_dir = BROWSER_PROFILE_DIR / f"thread_{self.owner_thread_id}"
        self.pending_click_xy: tuple[float, float] | None = None
        self.latest_click_targets: list[dict[str, Any]] = []
        self.latest_scroll_regions: list[dict[str, Any]] = []
        self.cached_by_sha: dict[str, BrowserImage] = {}
        self.cached_by_url: dict[str, str] = {}
        self.response_errors: list[str] = []

    def ensure(self, *, headful: bool = False, channel: str | None = None) -> None:
        if self.context is not None and self.page is not None:
            return

        BROWSER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.profile_dir.mkdir(parents=True, exist_ok=True)

        from playwright.sync_api import sync_playwright

        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
        if browsers_path:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path

        self._pw = sync_playwright().start()
        launch_kwargs: dict[str, Any] = {"headless": not headful}
        if channel in {"chrome", "msedge"}:
            launch_kwargs["channel"] = channel

        proxy_url = (
            os.environ.get("HTTP_PROXY")
            or os.environ.get("HTTPS_PROXY")
            or os.environ.get("TAVILY_PROXY", "").strip()
            or None
        )
        if proxy_url:
            launch_kwargs["proxy"] = {"server": proxy_url}

        self.context = self._pw.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            viewport={"width": 1280, "height": 900},
            user_agent=DEFAULT_UA,
            locale="zh-CN",
            **launch_kwargs,
        )
        self.page = self.context.new_page()
        self.page.on("response", self._cache_response)

    def close(self) -> None:
        try:
            if self.context is not None:
                self.context.close()
        finally:
            self.context = None
            self.page = None
            try:
                if self._browser is not None:
                    self._browser.close()
            finally:
                self._browser = None
                if self._pw is not None:
                    self._pw.stop()
                    self._pw = None

    def _cache_response(self, response: Any) -> None:
        if len(self.cached_by_sha) >= 80:
            return
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        mime = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if not mime.startswith("image/"):
            return
        try:
            body = response.body()
        except Exception as exc:
            self.response_errors.append(f"{response.url}: {exc}")
            return
        if len(body) < 1024:
            return

        digest = hashlib.sha256(body).hexdigest()
        if digest in self.cached_by_sha:
            self.cached_by_url[response.url] = self.cached_by_sha[digest].ref
            return

        ref = digest[:12]
        path = BROWSER_IMAGE_DIR / f"{ref}{_image_extension(mime, response.url)}"
        path.write_bytes(body)
        try:
            page_url = response.frame.url
        except Exception:
            page_url = self.page.url if self.page is not None else ""
        item = BrowserImage(
            ref=ref,
            url=response.url,
            path=str(path),
            mime=mime,
            size_bytes=len(body),
            sha256=digest,
            page_url=page_url,
        )
        self.cached_by_sha[digest] = item
        self.cached_by_url[response.url] = ref
        logger.debug("[browser] cached image ref=%s size=%d url=%s", ref, len(body), response.url[:100])

    def wait_ready(
        self,
        *,
        wait_until: str = "domcontentloaded",
        wait_ms: int = 800,
        visible_images: int = 0,
        selector: str = "",
        timeout_ms: int = 10_000,
    ) -> list[str]:
        page = self.require_page()
        events: list[str] = []
        load_state = _normalize_load_state(wait_until)
        if load_state != "commit":
            try:
                page.wait_for_load_state(load_state, timeout=timeout_ms)
                events.append(f"load_state={load_state}")
            except Exception as exc:
                events.append(f"load_state_failed={exc}")
        if selector:
            try:
                page.wait_for_selector(selector, timeout=timeout_ms)
                events.append(f"selector={selector!r}")
            except Exception as exc:
                events.append(f"selector_failed={exc}")
        if visible_images > 0:
            deadline = time.perf_counter() + timeout_ms / 1000
            last_count = 0
            while time.perf_counter() < deadline:
                last_count = int(page.evaluate(_COUNT_VISIBLE_IMAGES_JS) or 0)
                if last_count >= visible_images:
                    events.append(f"visible_images={last_count}")
                    break
                page.wait_for_timeout(250)
            else:
                events.append(f"visible_images_timeout={last_count}/{visible_images}")
        if wait_ms > 0:
            page.wait_for_timeout(max(0, int(wait_ms)))
            events.append(f"wait_ms={wait_ms}")
        return events

    def require_page(self) -> Any:
        self.ensure()
        if self.page is None:
            raise RuntimeError("browser page is not available")
        return self.page

    def open(self, url: str, **wait_kwargs: Any) -> dict[str, Any]:
        if not re.match(r"^(https?|file)://", url, re.IGNORECASE):
            raise ValueError("url must start with http://, https://, or file://")
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        page = self.require_page()
        wait_until = _normalize_load_state(wait_kwargs.get("wait_until"))
        timeout_ms = int(wait_kwargs.get("timeout_ms") or 30_000)
        events: list[str] = []
        try:
            page.goto(url, wait_until=wait_until, timeout=timeout_ms)
            events.append(f"goto={wait_until}")
        except PlaywrightTimeoutError:
            if page.url in {"", "about:blank"}:
                raise
            logger.warning(
                "[browser] goto timeout wait_until=%s timeout_ms=%d url=%s current=%s",
                wait_until,
                timeout_ms,
                url,
                page.url,
            )
            events.append(f"goto_timeout={wait_until}:{timeout_ms}ms")
            if wait_until == "networkidle":
                wait_kwargs = {**wait_kwargs, "wait_until": "domcontentloaded"}
                events.append("fallback_wait_until=domcontentloaded")
            elif wait_until == "load":
                wait_kwargs = {**wait_kwargs, "wait_until": "commit"}
                events.append("fallback_wait_until=commit")
        events.extend(self.wait_ready(**wait_kwargs))
        return self.result(events=events)

    def result(self, *, events: list[str] | None = None, include_screenshot: bool = True) -> dict[str, Any]:
        page = self.require_page()
        cached_images = self.cached_images()
        response_error_limit = _limit("response_errors")
        result = {
            "snapshot_id": f"brsnap_{int(time.time() * 1000)}",
            "url": page.url,
            "title": page.title() or "",
            "text_preview": self.text_preview(),
            "image_urls": self.page_image_urls(),
            "scroll": self.scroll_state(),
            "click_targets": self.click_targets(limit=_limit("click_targets")),
            "visible_images": self.visible_images(limit=_limit("visible_images")),
            "cached_images": [
                self.public_cached_image(item)
                for item in cached_images[:_limit("cached_images")]
            ],
            "cached_images_total": len(cached_images),
            "response_errors": [
                _shorten(item, _limit("text_chars"))
                for item in (
                    self.response_errors[-response_error_limit:]
                    if response_error_limit > 0
                    else []
                )
            ],
            "events": events or [],
        }
        result["image_count"] = len(result["image_urls"])
        if self.pending_click_xy is not None:
            result["pending_click"] = {
                "x": self.pending_click_xy[0],
                "y": self.pending_click_xy[1],
            }
        if include_screenshot:
            shot = self.capture_viewport_image()
            ref = shot["ref"]
            result["viewport_image_ref"] = ref
            result["_multimodal_parts"] = [
                {
                    "mime_type": "image/png",
                    "display_name": "browser_viewport.png",
                    "data": shot["data"],
                }
            ]
        return result

    def capture_viewport_image(
        self,
        *,
        target_overlay: list[dict[str, Any]] | None = None,
        annotate: bool | None = None,
    ) -> dict[str, Any]:
        """Capture the current viewport and return an internal image part."""
        page = self.require_page()
        annotate_screenshot = _browser_screenshot_annotations_enabled() if annotate is None else bool(annotate)
        click_preview_overlayed = False
        target_overlayed = False
        if annotate_screenshot and target_overlay:
            page.evaluate(_SHOW_TARGET_OVERLAY_JS, target_overlay)
            target_overlayed = True
        if annotate_screenshot and self.pending_click_xy is not None:
            page.evaluate(_SHOW_CLICK_PREVIEW_JS, {
                "x": self.pending_click_xy[0],
                "y": self.pending_click_xy[1],
            })
            click_preview_overlayed = True
        try:
            png = _resize_png(page.screenshot(full_page=False, type="png"))
        finally:
            if click_preview_overlayed:
                page.evaluate(_CLEAR_CLICK_PREVIEW_JS)
            if target_overlayed:
                page.evaluate(_CLEAR_TARGET_OVERLAY_JS)
        digest = hashlib.sha256(png).hexdigest()
        ref = digest[:12]
        _write_browser_image(ref, png, ".png")
        global _LATEST_VIEWPORT_REF
        _LATEST_VIEWPORT_REF = ref
        return {
            "kind": "viewport",
            "ref": ref,
            "sha256": digest,
            "mime_type": "image/png",
            "display_name": "browser_viewport.png",
            "data": png,
        }

    def capture_visual_element(self, selector: object, frame: Any | None = None) -> dict[str, Any] | None:
        selector_text = str(selector or "").strip()
        if not selector_text:
            return None
        try:
            owner = frame if frame is not None else self.require_page()
            png = _resize_png(owner.locator(selector_text).first.screenshot(type="png", timeout=1500))
        except Exception:
            return None
        digest = hashlib.sha256(png).hexdigest()
        ref = digest[:12]
        _write_browser_image(ref, png, ".png")
        return {
            "ref": ref,
            "sha256": digest,
            "mime_type": "image/png",
            "display_name": "browser_visible_image",
            "data": png,
        }

    def capture_viewport_clip(self, rect: object) -> dict[str, Any] | None:
        if not isinstance(rect, dict):
            return None
        try:
            clip = {
                "x": max(0, float(rect.get("x") or 0)),
                "y": max(0, float(rect.get("y") or 0)),
                "width": float(rect.get("width") or 0),
                "height": float(rect.get("height") or 0),
            }
        except (TypeError, ValueError):
            return None
        if clip["width"] <= 1 or clip["height"] <= 1:
            return None
        try:
            png = _resize_png(
                self.require_page().screenshot(full_page=False, type="png", clip=clip)
            )
        except Exception:
            return None
        digest = hashlib.sha256(png).hexdigest()
        ref = digest[:12]
        _write_browser_image(ref, png, ".png")
        return {
            "ref": ref,
            "sha256": digest,
            "mime_type": "image/png",
            "display_name": "browser_visible_image",
            "data": png,
        }

    def capture_visible_image_element(self, dom_index: object) -> dict[str, Any] | None:
        try:
            index = int(dom_index)
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        return self.capture_visual_element(f"img >> nth={index}")

    def scroll_state(self) -> dict[str, Any]:
        return dict(self.require_page().evaluate(_SCROLL_STATE_JS) or {})

    def loading_state(self) -> dict[str, Any]:
        try:
            return dict(self.require_page().evaluate(_PAGE_LOADING_STATE_JS) or {})
        except Exception:
            return {
                "active": False,
                "ready_state": "unknown",
                "images": 0,
                "visible_images": 0,
                "pending_images": 0,
                "pending_visible_images": 0,
            }

    def visible_images(self, limit: int | None = 20, *, compact: bool = True) -> list[dict[str, Any]]:
        if limit is not None and limit <= 0:
            return []
        rows = list(self.require_page().evaluate(_VISIBLE_IMAGES_JS, limit) or [])
        for row in rows:
            if isinstance(row, dict):
                if compact:
                    row["src"] = _compact_url(row.get("src"))
                row["alt"] = _shorten(row.get("alt"), _limit("text_chars"))
        return rows

    def _viewport_size(self) -> dict[str, int]:
        size = dict(self.require_page().evaluate("() => ({width: window.innerWidth || 0, height: window.innerHeight || 0})") or {})
        return {"width": int(size.get("width") or 0), "height": int(size.get("height") or 0)}

    @staticmethod
    def _clip_rect(rect: object, viewport: dict[str, Any]) -> dict[str, int] | None:
        if not isinstance(rect, dict):
            return None
        try:
            x = float(rect.get("x") or 0)
            y = float(rect.get("y") or 0)
            width = float(rect.get("width") or 0)
            height = float(rect.get("height") or 0)
            viewport_width = float(viewport.get("width") or 0)
            viewport_height = float(viewport.get("height") or 0)
        except (TypeError, ValueError):
            return None
        left = max(0.0, x)
        top = max(0.0, y)
        right = min(viewport_width, x + width)
        bottom = min(viewport_height, y + height)
        clipped_width = max(0.0, right - left)
        clipped_height = max(0.0, bottom - top)
        if clipped_width <= 1 or clipped_height <= 1:
            return None
        return {
            "x": round(left),
            "y": round(top),
            "width": round(clipped_width),
            "height": round(clipped_height),
        }

    @staticmethod
    def _offset_rect(rect: object, offset: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(rect, dict):
            return None
        try:
            return {
                "x": float(rect.get("x") or 0) + float(offset.get("x") or 0),
                "y": float(rect.get("y") or 0) + float(offset.get("y") or 0),
                "width": float(rect.get("width") or 0),
                "height": float(rect.get("height") or 0),
            }
        except (TypeError, ValueError):
            return None

    def _visible_child_frame_entries(self, viewport: dict[str, Any]) -> list[dict[str, Any]]:
        page = self.require_page()
        frames: list[dict[str, Any]] = []
        main_frame = getattr(page, "main_frame", None)
        for frame in list(getattr(page, "frames", []) or []):
            if frame == main_frame:
                continue
            try:
                element = frame.frame_element()
                meta = dict(element.evaluate("""el => {
                    const compact = value => String(value || '').replace(/\\s+/g, ' ').trim();
                    const tag = String(el.tagName || '').toLowerCase();
                    const title = compact(el.getAttribute('title') || el.title || '');
                    const aria = compact(el.getAttribute('aria-label') || '');
                    const nameAttr = compact(el.getAttribute('name') || '');
                    const id = compact(el.getAttribute('id') || '');
                    const name = title || aria || nameAttr || id;
                    const source = title ? 'title' : (aria ? 'aria' : (nameAttr ? 'name' : (id ? 'id' : '')));
                    return {tag, title, name, source};
                }""") or {})
                tag = str(meta.get("tag") or "")
                if tag in {"object", "embed"}:
                    continue
                box = element.bounding_box()
            except Exception:
                continue
            clipped = self._clip_rect(box, viewport)
            if clipped is not None:
                frames.append({
                    "_frame": frame,
                    "tag": tag,
                    "rect": clipped,
                    "name": str(meta.get("name") or ""),
                    "source": str(meta.get("source") or ""),
                    "title": str(meta.get("title") or ""),
                    "url": str(getattr(frame, "url", "") or ""),
                })
        return frames

    def _visible_child_frames(self, viewport: dict[str, Any]) -> list[tuple[Any, dict[str, int]]]:
        return [
            (item["_frame"], item["rect"])
            for item in self._visible_child_frame_entries(viewport)
            if item.get("_frame") is not None and isinstance(item.get("rect"), dict)
        ]

    def viewport_visuals(self) -> list[dict[str, Any]]:
        page = self.require_page()
        viewport = self._viewport_size()
        rows: list[dict[str, Any]] = []

        def collect(owner: Any, offset: dict[str, int], *, frame: Any | None = None, frame_index: int | None = None) -> None:
            try:
                local_rows = list(owner.evaluate(_VIEWPORT_VISUALS_JS) or [])
            except Exception:
                return
            for row in local_rows:
                if not isinstance(row, dict):
                    continue
                rect = self._offset_rect(row, offset)
                clipped = self._clip_rect(rect, viewport)
                if clipped is None:
                    continue
                row = dict(row)
                row.update(clipped)
                if frame is not None:
                    row["_frame"] = frame
                if frame_index is not None:
                    row["frame"] = frame_index
                rows.append(row)

        collect(page, {"x": 0, "y": 0})
        for frame_index, entry in enumerate(self._visible_child_frame_entries(viewport)):
            frame = entry.get("_frame")
            offset = entry.get("rect")
            if frame is not None and isinstance(offset, dict):
                collect(frame, offset, frame=frame, frame_index=frame_index)

        rows.sort(key=lambda item: (int(item.get("y") or 0), int(item.get("x") or 0)))
        return rows

    def text_preview(self, limit: int | None = None) -> str:
        try:
            text = str(self.require_page().evaluate(_TEXT_PREVIEW_JS) or "")
        except Exception:
            return ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        limit = _limit("text_preview_chars") if limit is None else max(0, int(limit))
        if limit <= 0:
            return ""
        if len(text) > limit:
            return text[:limit] + f"...(truncated, total {len(text)} chars)"
        return text

    def page_image_urls(self, limit: int | None = None) -> list[str]:
        page = self.require_page()
        limit = _limit("page_image_urls") if limit is None else max(0, int(limit))
        if limit <= 0:
            return []
        try:
            raw = list(page.evaluate(_PAGE_IMAGE_URLS_JS) or [])
        except Exception:
            return []
        urls: list[str] = []
        seen: set[str] = set()
        for item in raw:
            url = str(item or "").strip()
            if not url or url.startswith("data:"):
                continue
            if url.startswith("//"):
                url = "https:" + url
            elif not re.match(r"^https?://", url, re.IGNORECASE):
                url = urljoin(page.url, url)
            lower = url.lower()
            if ".ico" in lower or "favicon" in lower:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(_compact_url(url))
            if len(urls) >= limit:
                break
        return urls

    def click_targets(self, limit: int = 30) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        rows = list(self.require_page().evaluate(_CLICK_TARGETS_JS, limit) or [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("text", "alt"):
                row[key] = _shorten(row.get(key), _limit("text_chars"))
            for key in ("href", "src"):
                row[key] = _compact_url(row.get(key))
        return rows

    def click_target(self, index: int) -> dict[str, Any]:
        page = self.require_page()
        try:
            item = self.latest_click_targets[int(index)]
        except (IndexError, TypeError, ValueError):
            item = None
        if isinstance(item, dict):
            center = item.get("center") if isinstance(item.get("center"), dict) else {}
            try:
                x = float(center.get("x"))
                y = float(center.get("y"))
            except (TypeError, ValueError):
                x = y = -1.0
            if x >= 0 and y >= 0:
                page.mouse.click(x, y)
                return {
                    "ok": True,
                    "target": {
                        "index": index,
                        "role": item.get("role"),
                        "name": item.get("name"),
                        "href": item.get("href"),
                        "x": x,
                        "y": y,
                    },
                    "count": len(self.latest_click_targets),
                    "mode": "snapshot_center",
                }
        return dict(page.evaluate(_CLICK_TARGET_JS, {"index": int(index), "limit": 60}) or {})

    def scroll_region(self, index: int, pixels: int) -> dict[str, Any]:
        page = self.require_page()
        try:
            item = self.latest_scroll_regions[int(index)]
        except (IndexError, TypeError, ValueError):
            return {"ok": False, "error": f"scroll region index out of range: {index}", "count": len(self.latest_scroll_regions)}
        center = item.get("center") if isinstance(item.get("center"), dict) else {}
        try:
            x = float(center.get("x"))
            y = float(center.get("y"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "scroll region has no usable center", "index": index}
        page.mouse.move(x, y)
        page.mouse.wheel(0, int(pixels))
        return {
            "ok": True,
            "index": index,
            "x": x,
            "y": y,
            "pixels": int(pixels),
            "region": {
                "index": item.get("index"),
                "tag": item.get("tag"),
                "role": item.get("role"),
                "name": item.get("name"),
            },
        }

    def set_pending_click(self, x: float, y: float) -> dict[str, Any]:
        self.pending_click_xy = (float(x), float(y))
        return {"ok": True, "x": self.pending_click_xy[0], "y": self.pending_click_xy[1]}

    def confirm_pending_click(self) -> dict[str, Any]:
        if self.pending_click_xy is None:
            return {"ok": False, "error": "No pending click. Use move_xy first."}
        x, y = self.pending_click_xy
        self.pending_click_xy = None
        self.require_page().mouse.click(x, y)
        return {"ok": True, "x": x, "y": y}

    def cached_images(self) -> list[BrowserImage]:
        return sorted(self.cached_by_sha.values(), key=lambda item: item.size_bytes, reverse=True)

    def public_cached_image(self, item: BrowserImage) -> dict[str, Any]:
        return {
            "ref": item.ref,
            "url": _compact_url(item.url),
            "mime": item.mime,
            "size_bytes": item.size_bytes,
        }

    def _cached_image_for_url(self, url: object) -> BrowserImage | None:
        ref = self.cached_by_url.get(str(url or ""))
        if not ref:
            return None
        for item in self.cached_by_sha.values():
            if item.ref == ref:
                return item
        return None

    def viewport_state(self) -> dict[str, Any]:
        page = self.require_page()
        state = dict(page.evaluate(_BROWSER_VIEWPORT_STATE_JS) or {})
        viewport = state.get("viewport") if isinstance(state.get("viewport"), dict) else self._viewport_size()
        frame_entries = self._visible_child_frame_entries(viewport)
        frames: list[dict[str, Any]] = []
        sources: list[tuple[dict[str, Any], dict[str, int], int | None]] = [(state, {"x": 0, "y": 0}, None)]
        for entry in frame_entries:
            frame = entry.get("_frame")
            offset = entry.get("rect")
            if frame is None or not isinstance(offset, dict):
                continue
            try:
                frame_state = dict(frame.evaluate(_BROWSER_VIEWPORT_STATE_JS) or {})
            except Exception:
                continue
            frame_index = len(frames)
            frames.append({
                "index": frame_index,
                "tag": entry.get("tag") or "iframe",
                "rect": dict(offset),
                "name": entry.get("name") or "",
                "source": entry.get("source") or "",
                "title": entry.get("title") or "",
                "url": entry.get("url") or "",
            })
            sources.append((frame_state, offset, frame_index))
        merged_targets, merged_blocks, merged_scroll_regions, merged_tables, merged_indicators = self._merge_viewport_sources(sources, viewport)
        state["click_targets"] = merged_targets
        state["target_candidates"] = len(merged_targets)
        state["text_blocks"] = merged_blocks
        state["scroll_regions"] = merged_scroll_regions
        state["tables"] = merged_tables
        state["indicators"] = merged_indicators
        state["frames"] = frames
        self.latest_click_targets = merged_targets
        self.latest_scroll_regions = merged_scroll_regions
        return state

    def _merge_viewport_sources(
        self,
        sources: list[tuple[dict[str, Any], dict[str, int], int | None]],
        viewport: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        targets: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        scroll_regions: list[dict[str, Any]] = []
        tables: list[dict[str, Any]] = []
        indicators: list[dict[str, Any]] = []
        local_maps: list[dict[str, str]] = []
        cell_state_keys = {
            "aria_sort", "aria_busy", "deleted", "inserted", "marked",
            "background_tone", "text_tone",
        }

        def has_cell_state(cell: dict[str, Any]) -> bool:
            return any(str(cell.get(key) or "") for key in cell_state_keys)

        for source_index, (state, offset, frame_index) in enumerate(sources):
            local_map: dict[str, str] = {}
            for item in state.get("click_targets") or []:
                if not isinstance(item, dict):
                    continue
                rect = self._clip_rect(self._offset_rect(item.get("rect"), offset), viewport)
                if rect is None:
                    continue
                tmp_id = f"{source_index}:{item.get('index')}"
                local_map[str(item.get("index"))] = tmp_id
                target = dict(item)
                target["_tmp_target_id"] = tmp_id
                target["_source_order"] = len(targets)
                target["rect"] = rect
                target["center"] = {
                    "x": round(rect["x"] + rect["width"] / 2),
                    "y": round(rect["y"] + rect["height"] / 2),
                }
                if frame_index is not None:
                    target["frame"] = frame_index
                targets.append(target)
            local_maps.append(local_map)

            for block in state.get("text_blocks") or []:
                if not isinstance(block, dict):
                    continue
                rect = self._clip_rect(self._offset_rect(block.get("rect"), offset), viewport)
                if rect is None:
                    continue
                parts: list[dict[str, Any]] = []
                for part in block.get("parts") or []:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "ref":
                        tmp_id = local_map.get(str(part.get("target")))
                        if tmp_id is not None:
                            parts.append({"type": "ref", "target": tmp_id})
                    elif part.get("type") == "text":
                        text = str(part.get("text") or "")
                        if text:
                            parts.append({"type": "text", "text": text})
                if not parts:
                    continue
                merged_block = dict(block)
                merged_block["_source_order"] = len(blocks)
                merged_block["rect"] = rect
                merged_block["parts"] = parts
                if frame_index is not None:
                    merged_block["frame"] = frame_index
                blocks.append(merged_block)

            for region in state.get("scroll_regions") or []:
                if not isinstance(region, dict):
                    continue
                rect = self._clip_rect(self._offset_rect(region.get("rect"), offset), viewport)
                if rect is None:
                    continue
                item = dict(region)
                item["_source_order"] = len(scroll_regions)
                item["rect"] = rect
                item["center"] = {
                    "x": round(rect["x"] + rect["width"] / 2),
                    "y": round(rect["y"] + rect["height"] / 2),
                }
                if frame_index is not None:
                    item["frame"] = frame_index
                scroll_regions.append(item)

            for table in state.get("tables") or []:
                if not isinstance(table, dict):
                    continue
                table_rect = self._clip_rect(self._offset_rect(table.get("rect"), offset), viewport)
                if table_rect is None:
                    continue
                rows: list[dict[str, Any]] = []
                for row in table.get("rows") or []:
                    if not isinstance(row, dict):
                        continue
                    row_rect = self._clip_rect(self._offset_rect(row.get("rect"), offset), viewport)
                    if row_rect is None:
                        continue
                    cells: list[dict[str, Any]] = []
                    for cell in row.get("cells") or []:
                        if not isinstance(cell, dict):
                            continue
                        cell_rect = self._clip_rect(self._offset_rect(cell.get("rect"), offset), viewport)
                        if cell_rect is None:
                            continue
                        parts: list[dict[str, Any]] = []
                        for part in cell.get("parts") or []:
                            if not isinstance(part, dict):
                                continue
                            if part.get("type") == "ref":
                                tmp_id = local_map.get(str(part.get("target")))
                                if tmp_id is not None:
                                    parts.append({"type": "ref", "target": tmp_id})
                            elif part.get("type") == "text":
                                text = str(part.get("text") or "")
                                if text:
                                    parts.append({"type": "text", "text": text})
                        if not parts and not has_cell_state(cell):
                            continue
                        merged_cell = dict(cell)
                        merged_cell["rect"] = cell_rect
                        merged_cell["parts"] = parts
                        cells.append(merged_cell)
                    if not cells:
                        continue
                    merged_row = dict(row)
                    merged_row["rect"] = row_rect
                    merged_row["cells"] = cells
                    rows.append(merged_row)
                if not rows:
                    continue
                merged_table = dict(table)
                merged_table["_source_order"] = len(tables)
                merged_table["rect"] = table_rect
                merged_table["rows"] = rows
                if frame_index is not None:
                    merged_table["frame"] = frame_index
                tables.append(merged_table)

            for indicator in state.get("indicators") or []:
                if not isinstance(indicator, dict):
                    continue
                rect = self._clip_rect(self._offset_rect(indicator.get("rect"), offset), viewport)
                if rect is None:
                    continue
                item = dict(indicator)
                item["_source_order"] = len(indicators)
                item["rect"] = rect
                if frame_index is not None:
                    item["frame"] = frame_index
                indicators.append(item)

        targets.sort(key=lambda item: (
            int((item.get("rect") or {}).get("y") or 0),
            int((item.get("rect") or {}).get("x") or 0),
            int(item.get("_source_order") or 0),
        ))
        final_index_by_tmp: dict[str, int] = {}
        for index, item in enumerate(targets):
            tmp_id = str(item.pop("_tmp_target_id", ""))
            item.pop("_source_order", None)
            final_index_by_tmp[tmp_id] = index
            item["index"] = index

        blocks.sort(key=lambda item: (
            int((item.get("rect") or {}).get("y") or 0),
            int((item.get("rect") or {}).get("x") or 0),
            int(item.get("_source_order") or 0),
        ))
        final_blocks: list[dict[str, Any]] = []
        for index, block in enumerate(blocks):
            parts: list[dict[str, Any]] = []
            for part in block.get("parts") or []:
                if part.get("type") == "ref":
                    final_target = final_index_by_tmp.get(str(part.get("target")))
                    if final_target is not None:
                        parts.append({"type": "ref", "target": final_target})
                elif part.get("type") == "text":
                    parts.append(part)
            if not parts:
                continue
            block.pop("_source_order", None)
            block["index"] = index
            block["parts"] = parts
            final_blocks.append(block)

        scroll_regions.sort(key=lambda item: (
            int((item.get("rect") or {}).get("y") or 0),
            int((item.get("rect") or {}).get("x") or 0),
            int(item.get("_source_order") or 0),
        ))
        for index, item in enumerate(scroll_regions):
            item.pop("_source_order", None)
            item["index"] = index
        tables.sort(key=lambda item: (
            int((item.get("rect") or {}).get("y") or 0),
            int((item.get("rect") or {}).get("x") or 0),
            int(item.get("_source_order") or 0),
        ))
        final_tables: list[dict[str, Any]] = []
        for table_index, table in enumerate(tables):
            final_rows: list[dict[str, Any]] = []
            for row in table.get("rows") or []:
                final_cells: list[dict[str, Any]] = []
                for cell in row.get("cells") or []:
                    parts: list[dict[str, Any]] = []
                    for part in cell.get("parts") or []:
                        if part.get("type") == "ref":
                            final_target = final_index_by_tmp.get(str(part.get("target")))
                            if final_target is not None:
                                parts.append({"type": "ref", "target": final_target})
                        elif part.get("type") == "text":
                            parts.append(part)
                    if not parts and not has_cell_state(cell):
                        continue
                    cell["parts"] = parts
                    final_cells.append(cell)
                if not final_cells:
                    continue
                row["cells"] = final_cells
                row["index"] = len(final_rows)
                final_rows.append(row)
            if not final_rows:
                continue
            table.pop("_source_order", None)
            table["index"] = len(final_tables)
            table["rows"] = final_rows
            final_tables.append(table)
        indicators.sort(key=lambda item: (
            int((item.get("rect") or {}).get("y") or 0),
            int((item.get("rect") or {}).get("x") or 0),
            int(item.get("_source_order") or 0),
        ))
        for index, item in enumerate(indicators):
            item.pop("_source_order", None)
            item["index"] = index
        return targets, final_blocks, scroll_regions, final_tables, indicators

    @staticmethod
    def _world_target_key(value: object) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _world_part_refs(parts: object) -> set[str]:
        refs: set[str] = set()
        if not isinstance(parts, list):
            return refs
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "ref":
                refs.add(BrowserSession._world_target_key(part.get("target")))
        return refs

    @staticmethod
    def _world_referenced_targets(state: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        for block in state.get("text_blocks") or []:
            if isinstance(block, dict):
                refs.update(BrowserSession._world_part_refs(block.get("parts")))
        for table in state.get("tables") or []:
            if not isinstance(table, dict):
                continue
            for row in table.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                for cell in row.get("cells") or []:
                    if isinstance(cell, dict):
                        refs.update(BrowserSession._world_part_refs(cell.get("parts")))
        return refs

    @staticmethod
    def _world_rect(rect: object) -> tuple[float, float, float, float] | None:
        if not isinstance(rect, dict):
            return None
        try:
            x = float(rect.get("x") or 0)
            y = float(rect.get("y") or 0)
            width = float(rect.get("width") or 0)
            height = float(rect.get("height") or 0)
        except (TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        return x, y, width, height

    @staticmethod
    def _world_rect_covers(
        container: tuple[float, float, float, float],
        child: tuple[float, float, float, float],
        *,
        minimum_ratio: float = 0.65,
    ) -> bool:
        left = max(container[0], child[0])
        top = max(container[1], child[1])
        right = min(container[0] + container[2], child[0] + child[2])
        bottom = min(container[1] + container[3], child[1] + child[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        child_area = max(0.0, child[2]) * max(0.0, child[3])
        return child_area > 0 and intersection / child_area >= minimum_ratio

    @staticmethod
    def _world_opaque_href(href: object) -> bool:
        raw = str(href or "").strip()
        if not raw:
            return False
        try:
            parsed = urlparse(raw)
        except Exception:
            return len(raw) > 240
        path = parsed.path or ""
        if len(path) <= 240:
            return False
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) >= 2 and segments[0] in {"x", "sspa", "gp"}:
            return True
        return any(len(segment) > 96 for segment in segments)

    @staticmethod
    def _world_has_equivalent_visible_href(
        item: dict[str, Any],
        targets: list[dict[str, Any]],
        refs: set[str],
    ) -> bool:
        href = str(item.get("href") or "").strip()
        if not href:
            return False
        item_key = BrowserSession._world_target_key(item.get("index"))
        for other in targets:
            if BrowserSession._world_target_key(other.get("index")) == item_key:
                continue
            if BrowserSession._world_target_key(other.get("index")) not in refs:
                continue
            if other.get("role") != "link" or other.get("source") != "visible":
                continue
            if str(other.get("href") or "").strip() == href:
                return True
        return False

    @staticmethod
    def _world_covers_referenced_visible_link(
        item: dict[str, Any],
        targets: list[dict[str, Any]],
        refs: set[str],
    ) -> bool:
        item_rect = BrowserSession._world_rect(item.get("rect"))
        if item_rect is None:
            return False
        item_key = BrowserSession._world_target_key(item.get("index"))
        for other in targets:
            if BrowserSession._world_target_key(other.get("index")) == item_key:
                continue
            if BrowserSession._world_target_key(other.get("index")) not in refs:
                continue
            if other.get("role") != "link" or other.get("source") != "visible":
                continue
            other_rect = BrowserSession._world_rect(other.get("rect"))
            if other_rect is not None and BrowserSession._world_rect_covers(item_rect, other_rect):
                return True
        return False

    @staticmethod
    def _world_equivalent_targets(state: dict[str, Any]) -> set[str]:
        targets = [item for item in (state.get("click_targets") or []) if isinstance(item, dict)]
        refs = BrowserSession._world_referenced_targets(state)
        covered: set[str] = set()
        for item in targets:
            if item.get("role") != "link":
                continue
            source = str(item.get("source") or "")
            key = BrowserSession._world_target_key(item.get("index"))
            if source in {"alt", "graphic", "visual", "href_tail"} and BrowserSession._world_has_equivalent_visible_href(
                item,
                targets,
                refs,
            ):
                covered.add(key)
                continue
            rect = BrowserSession._world_rect(item.get("rect"))
            thin_visible_sliver = rect is not None and (rect[2] <= 5 or rect[3] <= 5)
            if source == "href_tail" and (
                thin_visible_sliver
                or (
                    BrowserSession._world_opaque_href(item.get("href"))
                    and BrowserSession._world_covers_referenced_visible_link(item, targets, refs)
                )
            ):
                covered.add(key)
        return covered

    @staticmethod
    def _world_overlay_targets(state: dict[str, Any]) -> list[dict[str, Any]]:
        covered = BrowserSession._world_equivalent_targets(state)
        return [
            item
            for item in (state.get("click_targets") or [])
            if isinstance(item, dict)
            and BrowserSession._world_target_key(item.get("index")) not in covered
        ]

    @staticmethod
    def _signature_safe(value: object) -> object:
        if isinstance(value, dict):
            cleaned: dict[str, object] = {}
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                key_text = str(key)
                if key_text.startswith("_") or key_text in {"data", "sha256", "path"}:
                    continue
                cleaned[key_text] = BrowserSession._signature_safe(item)
            return cleaned
        if isinstance(value, list):
            return [BrowserSession._signature_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def world_change_signature(self) -> dict[str, Any]:
        """Return a lightweight semantic signature for browser wait triggers."""
        page = self.require_page()
        state = self.viewport_state()
        visuals = self.viewport_visuals()
        payload = {
            "url": page.url,
            "title": page.title() or "",
            "viewport": state.get("viewport") or {},
            "scroll": state.get("scroll") or {},
            "loading": self.loading_state(),
            "click_targets": state.get("click_targets") or [],
            "text_blocks": state.get("text_blocks") or [],
            "scroll_regions": state.get("scroll_regions") or [],
            "frames": state.get("frames") or [],
            "tables": state.get("tables") or [],
            "indicators": state.get("indicators") or [],
            "visuals": visuals,
        }
        encoded = json.dumps(
            self._signature_safe(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "url": page.url,
            "title": page.title() or "",
        }

    def world_snapshot(self) -> dict[str, Any]:
        """Return the compact browser surface rendered into <world>.

        This intentionally differs from result(): it only describes the current
        viewport and keeps image URLs out of the model-facing payload.
        """
        page = self.require_page()
        state = self.viewport_state()
        image_rows = self.viewport_visuals()
        images: list[dict[str, Any]] = []
        for row in image_rows:
            if not isinstance(row, dict):
                continue
            image_item = {
                "kind": row.get("kind") or "visual",
                "alt": row.get("alt") or "",
                "width": int(row.get("width") or 0),
                "height": int(row.get("height") or 0),
                "x": int(row.get("x") or 0),
                "y": int(row.get("y") or 0),
            }
            if "loaded" in row:
                image_item["loaded"] = bool(row.get("loaded"))
            if image_item.get("loaded") is not False:
                image_payload = self.capture_viewport_clip(row)
                if image_payload is not None:
                    image_item.update(image_payload)
            if row.get("frame") is not None:
                image_item["frame"] = int(row.get("frame"))
            if row.get("pseudo"):
                image_item["pseudo"] = str(row.get("pseudo"))
            images.append(image_item)
        return {
            "active": True,
            "url": page.url,
            "title": page.title() or "",
            "viewport_size": state.get("viewport") or {},
            "scroll": state.get("scroll") or self.scroll_state(),
            "loading": self.loading_state(),
            "click_targets": state.get("click_targets") or [],
            "target_candidates": state.get("target_candidates", len(state.get("click_targets") or [])),
            "text_blocks": state.get("text_blocks") or [],
            "scroll_regions": state.get("scroll_regions") or [],
            "frames": state.get("frames") or [],
            "tables": state.get("tables") or [],
            "indicators": state.get("indicators") or [],
            "images": images,
            "cached_images_total": len(self.cached_by_sha),
            "viewport": self.capture_viewport_image(target_overlay=self._world_overlay_targets(state)),
            "pending_click": (
                {"x": self.pending_click_xy[0], "y": self.pending_click_xy[1]}
                if self.pending_click_xy is not None
                else None
            ),
        }

    def read_image_file(self, image_ref: str) -> tuple[bytes, str] | None:
        for item in self.cached_by_sha.values():
            if item.ref == image_ref:
                return Path(item.path).read_bytes(), item.mime
        safe_ref = re.sub(r"[^a-zA-Z0-9_-]", "", image_ref)
        if safe_ref != image_ref or not safe_ref:
            return None
        for path in BROWSER_IMAGE_DIR.glob(f"{safe_ref}.*"):
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return path.read_bytes(), mime
        return None

    def make_locator(self, strategy: str, value: str, options: dict[str, Any] | None = None) -> Any:
        page = self.require_page()
        options = options or {}
        strategy = strategy.strip().lower()
        if strategy in {"css", "locator"}:
            return page.locator(value)
        if strategy == "text":
            return page.get_by_text(value, exact=bool(options.get("exact", False)))
        if strategy == "role":
            role_name = str(options.get("name", ""))
            role_options = {"exact": bool(options.get("exact", False))}
            if role_name:
                role_options["name"] = role_name
            return page.get_by_role(value, **role_options)
        if strategy == "label":
            return page.get_by_label(value, exact=bool(options.get("exact", False)))
        if strategy == "placeholder":
            return page.get_by_placeholder(value, exact=bool(options.get("exact", False)))
        if strategy == "test_id":
            return page.get_by_test_id(value)
        raise ValueError(f"unknown locator strategy: {strategy}")

    def locator_operation(
        self,
        *,
        strategy: str,
        value: str,
        op: str,
        nth: int | None = None,
        text: str = "",
        attr: str = "",
        key: str = "",
        options: dict[str, Any] | None = None,
        timeout_ms: int = 10_000,
        wait_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        locator = self.make_locator(strategy, value, options)
        count = locator.count()
        target = locator.nth(nth) if nth is not None else locator
        op = op.strip().lower()
        detail: dict[str, Any] = {"count": count}

        if op == "count":
            return {"detail": detail, "changed": False}
        if op in {"click", "fill", "press"} and nth is None and count != 1:
            raise ValueError(f"locator matched {count} elements; pass nth for non-unique locator")
        if op == "click":
            target.click(timeout=timeout_ms)
            events = self.wait_ready(**(wait_kwargs or {}))
            detail["events"] = events
            return {"detail": detail, "changed": True}
        if op == "fill":
            target.fill(text, timeout=timeout_ms)
            return {"detail": detail, "changed": True}
        if op == "press":
            target.press(key or text, timeout=timeout_ms)
            return {"detail": detail, "changed": True}
        if op == "text":
            detail["text"] = target.inner_text(timeout=timeout_ms)
            return {"detail": detail, "changed": False}
        if op == "attr":
            detail["attr"] = attr
            detail["value"] = target.get_attribute(attr, timeout=timeout_ms)
            return {"detail": detail, "changed": False}
        if op == "is_visible":
            detail["visible"] = target.is_visible(timeout=timeout_ms)
            return {"detail": detail, "changed": False}
        if op == "wait":
            target.wait_for(state=str((options or {}).get("state", "visible")), timeout=timeout_ms)
            return {"detail": detail, "changed": False}
        raise ValueError(f"unknown locator operation: {op}")


_SESSIONS: dict[int, BrowserSession] = {}


def get_browser_session() -> BrowserSession:
    thread_id = threading.get_ident()
    session = _SESSIONS.get(thread_id)
    if session is None:
        session = BrowserSession()
        _SESSIONS[thread_id] = session
        logger.debug("[browser] created session for thread=%s", thread_id)
    return session


def close_browser_session() -> None:
    thread_id = threading.get_ident()
    session = _SESSIONS.pop(thread_id, None)
    if session is not None:
        session.close()


_BROWSER_WORKER_THREAD: threading.Thread | None = None
_BROWSER_WORKER_THREAD_ID: int | None = None
_BROWSER_WORKER_QUEUE: queue.Queue[tuple[Callable[[], Any], queue.Queue[Any]]] | None = None
_BROWSER_WORKER_LOCK = threading.Lock()


def _browser_worker_main(work_queue: queue.Queue[tuple[Callable[[], Any], queue.Queue[Any]]]) -> None:
    global _BROWSER_WORKER_THREAD_ID
    _BROWSER_WORKER_THREAD_ID = threading.get_ident()
    logger.debug("[browser] worker started thread=%s", _BROWSER_WORKER_THREAD_ID)
    while True:
        fn, result_queue = work_queue.get()
        try:
            result_queue.put((True, fn()))
        except BaseException as exc:
            result_queue.put((False, exc))


def run_in_browser_thread(fn: Callable[[], T]) -> T:
    global _BROWSER_WORKER_QUEUE, _BROWSER_WORKER_THREAD
    if threading.get_ident() == _BROWSER_WORKER_THREAD_ID:
        return fn()

    with _BROWSER_WORKER_LOCK:
        if _BROWSER_WORKER_THREAD is None or not _BROWSER_WORKER_THREAD.is_alive():
            _BROWSER_WORKER_QUEUE = queue.Queue()
            _BROWSER_WORKER_THREAD = threading.Thread(
                target=_browser_worker_main,
                args=(_BROWSER_WORKER_QUEUE,),
                name="browser-control-worker",
                daemon=True,
            )
            _BROWSER_WORKER_THREAD.start()

    assert _BROWSER_WORKER_QUEUE is not None
    result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    _BROWSER_WORKER_QUEUE.put((fn, result_queue))
    ok, value = result_queue.get()
    if ok:
        return value
    raise value


def record_browser_activity(action: str, result: dict[str, Any]) -> None:
    if not isinstance(result, dict) or result.get("error"):
        return
    item = BrowserActivity(
        timestamp_ms=int(time.time() * 1000),
        action=action,
        url=str(result.get("url") or ""),
        title=str(result.get("title") or ""),
        events=[str(x) for x in (result.get("events") or [])][:12],
        viewport_image_ref=str(result.get("viewport_image_ref") or _LATEST_VIEWPORT_REF or ""),
        cached_images_count=int(result.get("cached_images_total") or len(result.get("cached_images") or [])),
        visible_images_count=len(result.get("visible_images") or []),
        click_targets_count=len(result.get("click_targets") or []),
    )
    _ACTIVITY_HISTORY.append(item)
    del _ACTIVITY_HISTORY[:-50]


def browser_debug_state() -> dict[str, Any]:
    latest = _ACTIVITY_HISTORY[-1] if _ACTIVITY_HISTORY else None
    return {
        "active": bool(_SESSIONS),
        "latest": asdict(latest) if latest else None,
        "history": [asdict(item) for item in reversed(_ACTIVITY_HISTORY[-20:])],
    }


def _browser_world_snapshot_in_thread() -> dict[str, Any] | None:
    session = _SESSIONS.get(threading.get_ident())
    if session is None or session.page is None:
        return None
    return session.world_snapshot()


def browser_world_snapshot() -> dict[str, Any] | None:
    """Return the active browser state without creating a new browser session."""
    if _BROWSER_WORKER_THREAD is None or not _BROWSER_WORKER_THREAD.is_alive():
        return None
    return run_in_browser_thread(_browser_world_snapshot_in_thread)


def _browser_world_signature_in_thread() -> dict[str, Any] | None:
    session = _SESSIONS.get(threading.get_ident())
    if session is None or session.page is None:
        return None
    return session.world_change_signature()


def browser_world_signature() -> dict[str, Any] | None:
    """Return a lightweight browser world signature without capturing screenshots."""
    if _BROWSER_WORKER_THREAD is None or not _BROWSER_WORKER_THREAD.is_alive():
        return None
    return run_in_browser_thread(_browser_world_signature_in_thread)


def browser_image_path(image_ref: str) -> Path | None:
    safe_ref = re.sub(r"[^a-zA-Z0-9_-]", "", image_ref)
    if safe_ref != image_ref or not safe_ref:
        return None
    for path in BROWSER_IMAGE_DIR.glob(f"{safe_ref}.*"):
        return path
    return None


def read_browser_image_file(image_ref: str) -> tuple[bytes, str] | None:
    """Read a browser image cache entry without creating a browser session."""
    path = browser_image_path(image_ref)
    if path is None or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return path.read_bytes(), mime


def make_image_data_url(image_ref: str) -> str | None:
    item = read_browser_image_file(image_ref)
    if item is None:
        return None
    raw, mime = item
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


_SCROLL_STATE_JS = """() => {
    const doc = document.documentElement;
    const body = document.body;
    const scrollY = Math.max(window.scrollY || 0, doc.scrollTop || 0, body ? body.scrollTop || 0 : 0);
    const viewportHeight = window.innerHeight || doc.clientHeight || 0;
    const pageHeight = Math.max(doc.scrollHeight || 0, body ? body.scrollHeight || 0 : 0);
    return {
        y: Math.round(scrollY),
        viewport_height: Math.round(viewportHeight),
        page_height: Math.round(pageHeight),
        can_scroll_up: scrollY > 0,
        can_scroll_down: scrollY + viewportHeight + 2 < pageHeight
    };
}"""

_PAGE_LOADING_STATE_JS = """() => {
    const imageRows = Array.from(document.querySelectorAll('img')).map(img => {
        const rect = img.getBoundingClientRect();
        const visible = rect.width > 20 && rect.height > 20
            && rect.bottom > 0 && rect.right > 0
            && rect.top < window.innerHeight && rect.left < window.innerWidth;
        const loaded = !!(img.complete && img.naturalWidth > 0 && img.naturalHeight > 0);
        return { visible, loaded };
    });
    const pendingImages = imageRows.filter(item => !item.loaded).length;
    const pendingVisibleImages = imageRows.filter(item => item.visible && !item.loaded).length;
    const readyState = String(document.readyState || 'unknown');
    return {
        active: readyState !== 'complete' || pendingVisibleImages > 0,
        ready_state: readyState,
        images: imageRows.length,
        visible_images: imageRows.filter(item => item.visible).length,
        pending_images: pendingImages,
        pending_visible_images: pendingVisibleImages
    };
}"""

_VISIBLE_IMAGES_JS = """(limit) => {
    const maxRows = limit === null || limit === undefined ? -1 : Math.max(0, Number(limit) || 0);
    const rows = [];
    for (const [domIndex, img] of Array.from(document.querySelectorAll('img')).entries()) {
        const rect = img.getBoundingClientRect();
        const visible = rect.width > 20 && rect.height > 20
            && rect.bottom > 0 && rect.right > 0
            && rect.top < window.innerHeight && rect.left < window.innerWidth;
        if (!visible) continue;
        rows.push({
            dom_index: domIndex,
            src: img.currentSrc || img.src || '',
            alt: img.alt || '',
            loaded: !!(img.complete && img.naturalWidth > 0),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            x: Math.round(rect.x),
            y: Math.round(rect.y)
        });
        if (maxRows >= 0 && rows.length >= maxRows) break;
    }
    return rows;
}"""

_VIEWPORT_VISUALS_JS = """() => {
    const marker = 'data-aicq-browser-visual-id';
    const rows = [];
    const seen = new Set();
    const skipTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'META', 'LINK']);

    function queryAllDeep(selector, root = document) {
        const result = [];
        const visit = (node) => {
            if (!node || !node.querySelectorAll) return;
            result.push(...Array.from(node.querySelectorAll(selector)));
            for (const el of Array.from(node.querySelectorAll('*'))) {
                if (el.shadowRoot) visit(el.shadowRoot);
            }
        };
        visit(root);
        return Array.from(new Set(result));
    }

    queryAllDeep(`[${marker}]`).forEach(el => el.removeAttribute(marker));

    function compact(value) {
        return String(value || '').replace(/\\s+/g, ' ').trim();
    }

    function styleVisible(el) {
        if (!el || skipTags.has(el.tagName)) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
        if (Number(style.opacity || '1') === 0) return false;
        return style;
    }

    function composedParent(el) {
        if (!el) return null;
        if (el.parentElement) return el.parentElement;
        const root = el.getRootNode && el.getRootNode();
        return root && root.host ? root.host : null;
    }

    function clipsOverflow(style) {
        const values = [style.overflow, style.overflowX, style.overflowY].join(' ').toLowerCase();
        return /\\b(hidden|auto|scroll|clip)\\b/.test(values) || String(style.contain || '').includes('paint');
    }

    function containsComposed(root, node) {
        let cur = node;
        while (cur) {
            if (cur === root) return true;
            cur = composedParent(cur);
        }
        return false;
    }

    function closestComposed(el, selector) {
        let cur = el;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            if (cur.matches && cur.matches(selector)) return cur;
            cur = composedParent(cur);
        }
        return cur && cur.matches && cur.matches(selector) ? cur : null;
    }

    function alphaOf(color) {
        const text = String(color || '').trim().toLowerCase();
        if (!text || text === 'transparent') return 0;
        const rgba = text.match(/^rgba?\\(([^)]+)\\)$/);
        if (!rgba) return 1;
        const parts = rgba[1].split(',').map(part => part.trim());
        if (parts.length < 4) return 1;
        const alpha = Number(parts[3]);
        return Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    }

    function paintsOverPoint(el) {
        if (!el || el === document.documentElement || el === document.body) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
        const opacity = Number(style.opacity || '1');
        if (!Number.isFinite(opacity) || opacity <= 0.05) return false;
        if (['IMG', 'VIDEO', 'CANVAS', 'SVG', 'IFRAME', 'OBJECT', 'EMBED'].includes(el.tagName)) return true;
        if (alphaOf(style.backgroundColor) * opacity >= 0.85) return true;
        const bg = String(style.backgroundImage || '');
        return !!bg && bg !== 'none' && opacity >= 0.85;
    }

    let activeModalRootCache = null;
    function activeModalRoots() {
        if (activeModalRootCache !== null) return activeModalRootCache;
        activeModalRootCache = queryAllDeep('dialog,[role="dialog"],[role="alertdialog"]').filter(el => {
            const style = styleVisible(el);
            if (!style) return false;
            const ariaModal = compact(el.getAttribute('aria-modal')).toLowerCase() === 'true';
            let nativeModal = false;
            try {
                nativeModal = !!(el.matches && el.matches(':modal'));
            } catch (_) {}
            if (!ariaModal && !nativeModal) return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 1 && rect.height > 1 &&
                rect.bottom > 0 && rect.right > 0 &&
                rect.top < window.innerHeight && rect.left < window.innerWidth;
        });
        return activeModalRootCache;
    }

    function blockedByActiveModal(el) {
        const roots = activeModalRoots();
        return roots.length > 0 && !roots.some(root => root === el || containsComposed(root, el));
    }

    function hitVisible(el, rect) {
        if (blockedByActiveModal(el)) return false;
        const style = window.getComputedStyle(el);
        if (style && style.pointerEvents === 'none') return true;
        if (closestComposed(el, '[inert]')) return true;
        const points = [
            [rect.left + rect.width / 2, rect.top + rect.height / 2],
            [rect.left + Math.min(4, rect.width / 2), rect.top + Math.min(4, rect.height / 2)],
            [rect.right - Math.min(4, rect.width / 2), rect.top + Math.min(4, rect.height / 2)],
            [rect.left + Math.min(4, rect.width / 2), rect.bottom - Math.min(4, rect.height / 2)],
            [rect.right - Math.min(4, rect.width / 2), rect.bottom - Math.min(4, rect.height / 2)]
        ];
        return points.some(([x, y]) => {
            const px = Math.max(0, Math.min(window.innerWidth - 1, x));
            const py = Math.max(0, Math.min(window.innerHeight - 1, y));
            const stack = document.elementsFromPoint ? document.elementsFromPoint(px, py) : [document.elementFromPoint(px, py)];
            for (const hit of stack) {
                if (!hit) continue;
                if (
                    hit === el ||
                    containsComposed(el, hit) ||
                    (hit !== document.body && hit !== document.documentElement && containsComposed(hit, el))
                ) return true;
                if (paintsOverPoint(hit)) return false;
            }
            return false;
        });
    }

    function clippedRawRect(raw, el) {
        let left = Math.max(0, raw.left);
        let top = Math.max(0, raw.top);
        let right = Math.min(window.innerWidth || 0, raw.right);
        let bottom = Math.min(window.innerHeight || 0, raw.bottom);
        let cur = el;
        while (cur && cur !== document.documentElement) {
            const style = styleVisible(cur);
            if (!style) return null;
            if (clipsOverflow(style)) {
                const clip = cur.getBoundingClientRect();
                left = Math.max(left, clip.left);
                top = Math.max(top, clip.top);
                right = Math.min(right, clip.right);
                bottom = Math.min(bottom, clip.bottom);
            }
            cur = composedParent(cur);
        }
        const width = right - left;
        const height = bottom - top;
        if (width <= 1 || height <= 1) return null;
        const clipped = { left, top, right, bottom, width, height };
        return hitVisible(el, clipped) ? clipped : null;
    }

    function clippedGeometryRawRect(raw, el) {
        let left = Math.max(0, raw.left);
        let top = Math.max(0, raw.top);
        let right = Math.min(window.innerWidth || 0, raw.right);
        let bottom = Math.min(window.innerHeight || 0, raw.bottom);
        let cur = el;
        while (cur && cur !== document.documentElement) {
            const style = styleVisible(cur);
            if (!style) return null;
            if (clipsOverflow(style)) {
                const clip = cur.getBoundingClientRect();
                left = Math.max(left, clip.left);
                top = Math.max(top, clip.top);
                right = Math.min(right, clip.right);
                bottom = Math.min(bottom, clip.bottom);
            }
            cur = composedParent(cur);
        }
        const width = right - left;
        const height = bottom - top;
        return width > 1 && height > 1 ? { left, top, right, bottom, width, height } : null;
    }

    function visibleRects(el) {
        const style = styleVisible(el);
        if (!style) return [];
        return Array.from(el.getClientRects())
            .map(rect => clippedRawRect(rect, el))
            .filter(rect => rect && rect.width > 2 && rect.height > 2);
    }

    function clippedRect(raw) {
        const left = Math.max(0, raw.left);
        const top = Math.max(0, raw.top);
        const right = Math.min(window.innerWidth || 0, raw.right);
        const bottom = Math.min(window.innerHeight || 0, raw.bottom);
        return {
            x: Math.round(left),
            y: Math.round(top),
            width: Math.max(0, Math.round(right - left)),
            height: Math.max(0, Math.round(bottom - top))
        };
    }

    function rectOf(el) {
        const rects = visibleRects(el).sort((a, b) => (a.top - b.top) || (a.left - b.left));
        if (!rects.length) return null;
        return clippedRect(rects[0]);
    }

    function largeEnough(rect, minWidth, minHeight, minArea) {
        return rect && rect.width >= minWidth && rect.height >= minHeight && rect.width * rect.height >= minArea;
    }

    function imageAlt(el) {
        return compact(el.getAttribute('alt') || el.getAttribute('aria-label') || el.getAttribute('title') || '');
    }

    function meaningfulVisualLabel(value) {
        const text = compact(value);
        if (!text) return '';
        if (/[\uE000-\uF8FF]/.test(text)) return '';
        if (!/[\\p{L}\\p{N}]/u.test(text)) return '';
        return text;
    }

    function decorativeVisual(el) {
        if (!el) return true;
        if (closestComposed(el, '[aria-hidden="true"]')) return true;
        const role = compact(el.getAttribute('role')).toLowerCase();
        return role === 'presentation' || role === 'none';
    }

    function addVisual(el, kind, rect, extra = {}) {
        if (!rect || seen.has(el)) return;
        seen.add(el);
        const id = `brvis_${rows.length}`;
        el.setAttribute(marker, id);
        rows.push({
            kind,
            capture_selector: `[${marker}="${id}"]`,
            alt: imageAlt(el),
            width: rect.width,
            height: rect.height,
            x: rect.x,
            y: rect.y,
            ...extra
        });
    }

    function maskImageOf(style) {
        return String(style.webkitMaskImage || style.maskImage || '');
    }

    function hasMaskPaint(style) {
        if (alphaOf(style.backgroundColor) > 0.05) return true;
        const bg = String(style.backgroundImage || '');
        return !!bg && bg !== 'none';
    }

    function cssPixel(value) {
        const match = String(value || '').trim().match(/^(-?\\d+(?:\\.\\d+)?)px$/);
        if (!match) return NaN;
        const number = Number(match[1]);
        return Number.isFinite(number) ? number : NaN;
    }

    function pseudoVisualInfo(el, pseudo) {
        let style;
        try {
            style = window.getComputedStyle(el, pseudo);
        } catch (_) {
            return null;
        }
        if (!style || style.display === 'none' || style.visibility === 'hidden') return null;
        const opacity = Number(style.opacity || '1');
        if (!Number.isFinite(opacity) || opacity <= 0.05) return null;
        const content = String(style.content || '').trim();
        const bg = String(style.backgroundImage || '');
        const mask = maskImageOf(style);
        const hasContentImage = /\\b(url|image)\\(/i.test(content);
        const hasBackgroundImage = !!bg && bg !== 'none' && /url\\(/i.test(bg);
        const hasMaskImage = !!mask && mask !== 'none' && /url\\(/i.test(mask);
        if (!hasContentImage && !hasBackgroundImage && !hasMaskImage) return null;
        if (hasBackgroundImage) {
            const repeat = `${style.backgroundRepeatX || style.backgroundRepeat} ${style.backgroundRepeatY || ''}`.toLowerCase();
            const bgSize = String(style.backgroundSize || '').toLowerCase();
            if (repeat.includes('repeat') && !bgSize.includes('cover') && !bgSize.includes('contain')) return null;
        }
        if (hasMaskImage && !hasMaskPaint(style)) return null;
        const width = cssPixel(style.width);
        const height = cssPixel(style.height);
        let kind = 'pseudo_image';
        if (hasMaskImage) kind = 'pseudo_mask';
        else if (hasBackgroundImage) kind = 'pseudo_background';
        return { kind, pseudo: pseudo.replace('::', ''), width, height };
    }

    function pseudoVisualLargeEnough(rect, info) {
        if (!rect || !info) return false;
        const visualWidth = Number.isFinite(info.width) ? info.width : rect.width;
        const visualHeight = Number.isFinite(info.height) ? info.height : rect.height;
        return largeEnough(rect, 48, 48, 2304) &&
            visualWidth >= 48 && visualHeight >= 48 && visualWidth * visualHeight >= 2304;
    }

    for (const img of queryAllDeep('img')) {
        if (decorativeVisual(img)) continue;
        const rect = rectOf(img);
        if (!largeEnough(rect, 32, 32, 1024)) continue;
        addVisual(img, 'image', rect, {
            src: img.currentSrc || img.src || '',
            loaded: !!(img.complete && img.naturalWidth > 0)
        });
    }

    for (const input of queryAllDeep('input[type="image" i]')) {
        if (decorativeVisual(input)) continue;
        const rect = rectOf(input);
        if (!largeEnough(rect, 32, 32, 1024)) continue;
        addVisual(input, 'input_image', rect, {
            src: input.currentSrc || input.src || input.getAttribute('src') || ''
        });
    }

    for (const canvas of queryAllDeep('canvas')) {
        if (decorativeVisual(canvas)) continue;
        const rect = rectOf(canvas);
        if (!largeEnough(rect, 48, 48, 2304)) continue;
        addVisual(canvas, 'canvas', rect);
    }

    for (const svg of queryAllDeep('svg')) {
        if (decorativeVisual(svg)) continue;
        const rect = rectOf(svg);
        if (!largeEnough(rect, 56, 56, 3136)) continue;
        addVisual(svg, 'svg', rect);
    }

    for (const video of queryAllDeep('video')) {
        if (decorativeVisual(video)) continue;
        const rect = rectOf(video);
        if (!largeEnough(rect, 96, 54, 5184)) continue;
        addVisual(video, 'video', rect, {
            src: video.currentSrc || video.src || video.getAttribute('poster') || '',
            loaded: !!(video.readyState > 0 || video.getAttribute('poster'))
        });
    }

    for (const embedded of queryAllDeep('object,embed')) {
        if (decorativeVisual(embedded)) continue;
        const rect = rectOf(embedded);
        if (!largeEnough(rect, 96, 72, 6912)) continue;
        addVisual(embedded, embedded.tagName.toLowerCase(), rect, {
            src: embedded.getAttribute('data') || embedded.getAttribute('src') || '',
            loaded: true
        });
    }

    for (const el of queryAllDeep('*')) {
        if (seen.has(el)) continue;
        if (decorativeVisual(el)) continue;
        const style = styleVisible(el);
        if (!style) continue;
        const bg = String(style.backgroundImage || '');
        if (!bg || bg === 'none' || !/url\\(/i.test(bg)) continue;
        const rect = rectOf(el);
        const semanticLabel = meaningfulVisualLabel(imageAlt(el));
        const largeVisual = semanticLabel
            ? largeEnough(rect, 48, 48, 2304)
            : largeEnough(rect, 96, 72, 6912);
        if (!largeVisual) continue;
        const repeat = `${style.backgroundRepeatX || style.backgroundRepeat} ${style.backgroundRepeatY || ''}`.toLowerCase();
        const bgSize = String(style.backgroundSize || '').toLowerCase();
        if (repeat.includes('repeat') && !bgSize.includes('cover') && !bgSize.includes('contain')) continue;
        addVisual(el, 'background', rect);
    }

    for (const el of queryAllDeep('*')) {
        if (seen.has(el)) continue;
        if (decorativeVisual(el)) continue;
        const style = styleVisible(el);
        if (!style) continue;
        const mask = maskImageOf(style);
        if (!mask || mask === 'none' || !/url\\(/i.test(mask)) continue;
        if (!hasMaskPaint(style)) continue;
        const rect = rectOf(el);
        if (!largeEnough(rect, 96, 72, 6912)) continue;
        addVisual(el, 'mask', rect);
    }

    for (const el of queryAllDeep('*')) {
        if (seen.has(el)) continue;
        if (decorativeVisual(el)) continue;
        const rect = rectOf(el);
        if (!rect) continue;
        for (const pseudo of ['::before', '::after']) {
            const info = pseudoVisualInfo(el, pseudo);
            if (!pseudoVisualLargeEnough(rect, info)) continue;
            addVisual(el, info.kind, rect, { pseudo: info.pseudo });
            break;
        }
    }

    rows.sort((a, b) => (a.y - b.y) || (a.x - b.x));
    return rows;
}"""

_COUNT_VISIBLE_IMAGES_JS = """() => {
    let count = 0;
    for (const img of document.querySelectorAll('img')) {
        const rect = img.getBoundingClientRect();
        const visible = rect.width > 20 && rect.height > 20
            && rect.bottom > 0 && rect.right > 0
            && rect.top < window.innerHeight && rect.left < window.innerWidth;
        if (visible && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) count += 1;
    }
    return count;
}"""

_TEXT_PREVIEW_JS = """() => {
    const el = document.body;
    return el ? (el.innerText || '') : '';
}"""

_PAGE_IMAGE_URLS_JS = """() => {
    const urls = new Set();
    document.querySelectorAll('img').forEach(el => {
        const current = el.currentSrc || el.src || el.getAttribute('src') || '';
        if (current) urls.add(current);
        const srcset = el.getAttribute('srcset') || '';
        for (const part of srcset.split(',')) {
            const candidate = part.trim().split(/\\s+/)[0];
            if (candidate) urls.add(candidate);
        }
    });
    document.querySelectorAll('source[srcset]').forEach(el => {
        const srcset = el.getAttribute('srcset') || '';
        for (const part of srcset.split(',')) {
            const candidate = part.trim().split(/\\s+/)[0];
            if (candidate) urls.add(candidate);
        }
    });
    for (const prop of ['og:image', 'twitter:image']) {
        const el = document.querySelector(`meta[property="${prop}"],meta[name="${prop}"]`);
        const content = el ? (el.getAttribute('content') || '') : '';
        if (content) urls.add(content);
    }
    return [...urls];
}"""

_SHOW_CLICK_PREVIEW_JS = """({ x, y }) => {
    const id = '__aicarus_click_preview__';
    document.getElementById(id)?.remove();
    const root = document.createElement('div');
    root.id = id;
    root.style.cssText = [
        'position:fixed',
        'left:0',
        'top:0',
        'width:0',
        'height:0',
        'z-index:2147483647',
        'pointer-events:none',
        'font-family:system-ui,sans-serif'
    ].join(';');

    function line(cssText) {
        const el = document.createElement('div');
        el.style.cssText = cssText;
        root.appendChild(el);
    }
    const xx = Number(x) || 0;
    const yy = Number(y) || 0;
    line(`position:fixed;left:${xx}px;top:0;width:1px;height:100vh;background:#ff1744;box-shadow:0 0 0 1px rgba(255,255,255,.75)`);
    line(`position:fixed;left:0;top:${yy}px;width:100vw;height:1px;background:#ff1744;box-shadow:0 0 0 1px rgba(255,255,255,.75)`);
    line(`position:fixed;left:${xx - 8}px;top:${yy - 8}px;width:16px;height:16px;border:2px solid #ff1744;border-radius:50%;background:rgba(255,255,255,.16);box-shadow:0 0 0 2px rgba(255,255,255,.85)`);
    const label = document.createElement('div');
    label.textContent = `${Math.round(xx)}, ${Math.round(yy)}`;
    label.style.cssText = `position:fixed;left:${xx + 12}px;top:${yy + 12}px;background:#ff1744;color:#fff;border:1px solid #fff;border-radius:4px;padding:2px 5px;font-size:11px;font-weight:700;line-height:1`;
    root.appendChild(label);
    document.documentElement.appendChild(root);
    return true;
}"""

_CLEAR_CLICK_PREVIEW_JS = """() => {
    document.getElementById('__aicarus_click_preview__')?.remove();
    return true;
}"""

_SHOW_TARGET_OVERLAY_JS = """(targets) => {
    document.getElementById('__aicarus_target_overlay__')?.remove();
    const root = document.createElement('div');
    root.id = '__aicarus_target_overlay__';
    root.style.cssText = [
        'position:fixed',
        'left:0',
        'top:0',
        'width:0',
        'height:0',
        'z-index:2147483647',
        'pointer-events:none',
        'font-family:Arial,system-ui,sans-serif'
    ].join(';');
    for (const target of targets || []) {
        const rect = target.rect || {};
        const index = target.index;
        if (!rect.width || !rect.height || index === undefined) continue;
        const box = document.createElement('div');
        box.style.cssText = [
            'position:fixed',
            `left:${rect.x}px`,
            `top:${rect.y}px`,
            `width:${rect.width}px`,
            `height:${rect.height}px`,
            'border:2px solid rgba(255,23,68,.9)',
            'box-sizing:border-box',
            'background:rgba(255,23,68,.04)'
        ].join(';');
        const badge = document.createElement('div');
        badge.textContent = String(index);
        badge.style.cssText = [
            'position:fixed',
            `left:${Math.max(0, Number(rect.x) || 0)}px`,
            `top:${Math.max(0, (Number(rect.y) || 0) - 18)}px`,
            'min-width:16px',
            'height:16px',
            'padding:1px 4px',
            'background:#ff1744',
            'color:white',
            'border:1px solid white',
            'border-radius:3px',
            'font-size:12px',
            'font-weight:700',
            'line-height:14px',
            'text-align:center',
            'box-shadow:0 1px 4px rgba(0,0,0,.35)'
        ].join(';');
        root.appendChild(box);
        root.appendChild(badge);
    }
    document.documentElement.appendChild(root);
    return true;
}"""

_CLEAR_TARGET_OVERLAY_JS = """() => {
    document.getElementById('__aicarus_target_overlay__')?.remove();
    return true;
}"""

_BROWSER_VIEWPORT_STATE_JS = """() => {
    const actionableRoles = [
        'button', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio',
        'checkbox', 'radio', 'switch', 'tab', 'option', 'treeitem',
        'slider', 'spinbutton', 'combobox'
    ];
    const actionableRoleSelector = actionableRoles.map(role => `[role="${role}"]`).join(',');
    const ariaDisabledRoleSelector = actionableRoles.map(role => `[role="${role}"][aria-disabled="true"]`).join(',');
    const cssDisabledControlSelector = [
        'button[class*="disabled" i]',
        'button[class*="unavailable" i]',
        'button[class*="readonly" i]',
        'button[class*="locked" i]',
        'button[style*="pointer-events" i]',
        'a[href][class*="disabled" i]',
        'a[href][class*="unavailable" i]',
        'a[href][class*="readonly" i]',
        'a[href][class*="locked" i]',
        'a[href][style*="pointer-events" i]',
        '[role="button"][class*="disabled" i]',
        '[role="button"][class*="unavailable" i]',
        '[role="button"][class*="readonly" i]',
        '[role="button"][class*="locked" i]',
        '[role="button"][style*="pointer-events" i]',
        '[role="link"][class*="disabled" i]',
        '[role="link"][class*="unavailable" i]',
        '[role="link"][class*="readonly" i]',
        '[role="link"][class*="locked" i]',
        '[role="link"][style*="pointer-events" i]'
    ].join(',');
    const disabledControlSelector = [
        'button:disabled',
        'button[aria-disabled="true"]',
        'input:disabled',
        'input[aria-disabled="true"]',
        'textarea:disabled',
        'textarea[aria-disabled="true"]',
        'select:disabled',
        'select[aria-disabled="true"]',
        'summary[aria-disabled="true"]',
        'a[aria-disabled="true"]',
        '[aria-disabled="true"] a[href]',
        '[aria-disabled="true"] button',
        '[aria-disabled="true"] input:not([type="hidden"])',
        '[aria-disabled="true"] textarea',
        '[aria-disabled="true"] select',
        '[aria-disabled="true"] summary',
        ariaDisabledRoleSelector,
        cssDisabledControlSelector
    ].filter(Boolean).join(',');
    const inertRoleSelector = actionableRoles.map(role => `[inert] [role="${role}"],[inert][role="${role}"]`).join(',');
    const inertControlSelector = [
        '[inert] a[href]',
        '[inert] button',
        '[inert] input:not([type="hidden"])',
        '[inert] textarea',
        '[inert] select',
        '[inert] summary',
        inertRoleSelector
    ].filter(Boolean).join(',');
    const editableSelector = '[contenteditable]:not([contenteditable="false"])';
    const clickableSelectors = [
        'a[href]', 'area[href]', 'button', actionableRoleSelector,
        'input:not([type="hidden"])', 'textarea', 'select', 'summary', editableSelector,
        '[onclick]', '[tabindex]:not([tabindex="-1"])'
    ];
    const clickableSelector = clickableSelectors.join(',');
    const targetSelector = ['a[href]', 'area[href]', 'button', 'summary', actionableRoleSelector, editableSelector].join(',');
    const loadingPlaceholderSelector = [
        '[class*="skeleton" i]',
        '[class*="shimmer" i]',
        '[class*="spinner" i]',
        '[class*="loader" i]',
        '[class*="loading" i]',
        '[aria-label*="loading" i]',
        '[aria-label*="spinner" i]',
        '[title*="loading" i]',
        '[title*="spinner" i]',
        '[data-state*="loading" i]',
        '[data-loading="true"]'
    ].join(',');
    const indicatorSelector = [
        'progress',
        'meter',
        '[role="progressbar"]',
        '[role="meter"]',
        'audio[controls]',
        'video[controls]',
        '[role="img"]',
        'svg',
        '[aria-busy="true"]',
        loadingPlaceholderSelector,
        disabledControlSelector,
        inertControlSelector
    ].filter(Boolean).join(',');
    const textOwnerSelector = [
        'h1,h2,h3,h4,h5,h6,p,li,dt,dd,pre,blockquote,figcaption,caption,th,td,label,legend,button,time',
        'mark,del,ins,s,strike,span[role],span[aria-live],span[aria-label]',
        'span[class*="badge" i],span[class*="status" i],span[class*="tag" i],span[class*="chip" i],span[class*="pill" i],span[class*="label" i]',
        'article,section,div',
        '[role="heading"],[role="paragraph"],[role="article"],[role="listitem"],[role="cell"],[role="text"]',
        '[role="tooltip"]',
        '[role="alert"],[role="status"],[role="note"],[role="log"],[role="timer"],[role="dialog"],[role="alertdialog"]',
        '[aria-live]:not([aria-live="off"])'
    ].join(',');
    const skipTags = new Set(['script', 'style', 'noscript', 'svg', 'canvas']);

    function queryAllDeep(selector, root = document) {
        const result = [];
        const visit = (node) => {
            if (!node || !node.querySelectorAll) return;
            result.push(...Array.from(node.querySelectorAll(selector)));
            for (const el of Array.from(node.querySelectorAll('*'))) {
                if (el.shadowRoot) visit(el.shadowRoot);
            }
        };
        visit(root);
        return Array.from(new Set(result));
    }

    function composedParent(el) {
        if (!el) return null;
        if (el.parentElement) return el.parentElement;
        const root = el.getRootNode && el.getRootNode();
        return root && root.host ? root.host : null;
    }

    function deepActiveElement() {
        let active = document.activeElement || null;
        while (active && active.shadowRoot && active.shadowRoot.activeElement) {
            active = active.shadowRoot.activeElement;
        }
        return active;
    }

    function containsComposed(root, node) {
        for (let current = node; current; current = composedParent(current)) {
            if (current === root) return true;
        }
        return false;
    }

    function focusedTarget(el) {
        const active = deepActiveElement();
        if (!active || !el) return false;
        return active === el || containsComposed(el, active);
    }

    function closestComposed(el, selector) {
        let cur = el;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            if (cur.matches && cur.matches(selector)) return cur;
            cur = composedParent(cur);
        }
        return cur && cur.matches && cur.matches(selector) ? cur : null;
    }

    function hasSkippedAncestor(el) {
        let cur = el;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            if (skipTags.has(tagNameOf(cur))) return true;
            cur = composedParent(cur);
        }
        return false;
    }

    function compact(value) {
        return String(value || '').replace(/\\s+/g, ' ').trim();
    }

    function tagNameOf(el) {
        return String((el && el.tagName) || '').toLowerCase();
    }

    function styleVisible(el) {
        if (!el || skipTags.has(tagNameOf(el))) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
        if (Number(style.opacity || '1') === 0) return false;
        return true;
    }

    function svgTextStyleVisible(el) {
        if (!el) return false;
        const tag = tagNameOf(el);
        if (['script', 'style', 'noscript', 'canvas'].includes(tag)) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
        if (Number(style.opacity || '1') === 0) return false;
        return true;
    }

    function interactionBlocked(el) {
        if (!el) return true;
        if (closestComposed(el, '[inert]')) return true;
        if (closestComposed(el, '[aria-disabled="true"]')) return true;
        if (el.disabled) return true;
        try {
            if (el.matches && el.matches(':disabled')) return true;
        } catch (_) {}
        return false;
    }

    function clipsOverflow(el) {
        try {
            const style = window.getComputedStyle(el);
            const values = [style.overflow, style.overflowX, style.overflowY].join(' ').toLowerCase();
            return /\\b(hidden|auto|scroll|clip)\\b/.test(values) || String(style.contain || '').includes('paint');
        } catch (_) {
            return false;
        }
    }

    function svgTextRect(el) {
        let raw = el.getBoundingClientRect();
        if (!raw || raw.width <= 1 || raw.height <= 1) return null;
        let left = Math.max(0, raw.left);
        let top = Math.max(0, raw.top);
        let right = Math.min(window.innerWidth || 0, raw.right);
        let bottom = Math.min(window.innerHeight || 0, raw.bottom);
        let cur = el;
        while (cur && cur !== document.documentElement) {
            if (!svgTextStyleVisible(cur)) return null;
            if (clipsOverflow(cur)) {
                const clip = cur.getBoundingClientRect();
                left = Math.max(left, clip.left);
                top = Math.max(top, clip.top);
                right = Math.min(right, clip.right);
                bottom = Math.min(bottom, clip.bottom);
            }
            cur = composedParent(cur);
        }
        raw = { left, top, right, bottom, width: right - left, height: bottom - top };
        if (raw.width <= 1 || raw.height <= 1) return null;
        return clippedRect(raw);
    }

    function containsComposed(root, node) {
        let cur = node;
        while (cur) {
            if (cur === root) return true;
            cur = composedParent(cur);
        }
        return false;
    }

    function alphaOf(color) {
        const text = String(color || '').trim().toLowerCase();
        if (!text || text === 'transparent') return 0;
        const rgba = text.match(/^rgba?\\(([^)]+)\\)$/);
        if (!rgba) return 1;
        const parts = rgba[1].split(',').map(part => part.trim());
        if (parts.length < 4) return 1;
        const alpha = Number(parts[3]);
        return Number.isFinite(alpha) ? Math.max(0, Math.min(1, alpha)) : 1;
    }

    function numericColorPart(value) {
        const text = String(value || '').trim();
        if (!text) return NaN;
        if (text.endsWith('%')) {
            const percent = Number(text.slice(0, -1));
            return Number.isFinite(percent) ? Math.round(Math.max(0, Math.min(100, percent)) * 2.55) : NaN;
        }
        const number = Number(text);
        return Number.isFinite(number) ? Math.max(0, Math.min(255, number)) : NaN;
    }

    function numericAlphaPart(value) {
        const text = String(value || '').trim();
        if (!text) return 1;
        if (text.endsWith('%')) {
            const percent = Number(text.slice(0, -1));
            return Number.isFinite(percent) ? Math.max(0, Math.min(1, percent / 100)) : 1;
        }
        const number = Number(text);
        return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 1;
    }

    function parseCssColor(color) {
        const text = String(color || '').trim().toLowerCase();
        if (!text || text === 'transparent') return null;
        const match = text.match(/^rgba?\\(([^)]+)\\)$/);
        if (!match) return null;
        let body = match[1].trim();
        let alpha = 1;
        if (body.includes('/')) {
            const split = body.split('/');
            body = split[0].trim();
            alpha = numericAlphaPart(split[1]);
        }
        const parts = body.includes(',')
            ? body.split(',').map(part => part.trim())
            : body.split(/\\s+/).filter(Boolean);
        if (parts.length < 3) return null;
        if (body.includes(',') && parts.length >= 4) alpha = numericAlphaPart(parts[3]);
        const r = numericColorPart(parts[0]);
        const g = numericColorPart(parts[1]);
        const b = numericColorPart(parts[2]);
        if (![r, g, b].every(Number.isFinite)) return null;
        const hex = '#' + [r, g, b].map(value => Math.round(value).toString(16).padStart(2, '0')).join('');
        return { r, g, b, alpha, hex };
    }

    function colorTone(info, { allowNeutral = false } = {}) {
        if (!info || info.alpha < 0.2) return '';
        const r = info.r / 255;
        const g = info.g / 255;
        const b = info.b / 255;
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const light = (max + min) / 2;
        const delta = max - min;
        if (delta < 0.08) {
            if (!allowNeutral) return '';
            if (light < 0.2) return 'black';
            if (light > 0.92) return 'white';
            return 'gray';
        }
        const saturation = light > 0.5 ? delta / (2 - max - min) : delta / (max + min);
        if (saturation < 0.22 && !allowNeutral) return '';
        let hue;
        if (max === r) hue = ((g - b) / delta) % 6;
        else if (max === g) hue = (b - r) / delta + 2;
        else hue = (r - g) / delta + 4;
        hue = (hue * 60 + 360) % 360;
        if (hue < 15 || hue >= 345) return 'red';
        if (hue < 45) return 'orange';
        if (hue < 70) return 'yellow';
        if (hue < 165) return 'green';
        if (hue < 200) return 'cyan';
        if (hue < 255) return 'blue';
        if (hue < 290) return 'purple';
        return 'pink';
    }

    function statusLikeElement(el) {
        if (!el || !el.getAttribute) return false;
        const role = explicitRole(el);
        const text = [
            role,
            el.className && typeof el.className === 'string' ? el.className : '',
            el.getAttribute('id'),
            el.getAttribute('aria-label'),
            el.getAttribute('title'),
            el.getAttribute('data-status'),
            el.getAttribute('data-state'),
            el.getAttribute('data-severity'),
            el.getAttribute('data-priority')
        ].filter(Boolean).join(' ').toLowerCase();
        return /\\b(status|state|badge|tag|chip|pill|label|alert|error|danger|warn|warning|success|fail|failed|critical|severity|priority|online|offline|available|unavailable|active|inactive|selected|current|legend|swatch|dot)\\b/.test(text);
    }

    function visualToneState(el) {
        const state = {};
        if (!el || !el.getAttribute) return state;
        let style;
        try {
            style = window.getComputedStyle(el);
        } catch (_) {
            return state;
        }
        if (!style) return state;
        const statusLike = statusLikeElement(el);
        const background = parseCssColor(style.backgroundColor);
        const backgroundTone = colorTone(background, { allowNeutral: statusLike });
        if (backgroundTone && background && background.alpha >= 0.25) state.background_tone = backgroundTone;
        const foreground = parseCssColor(style.color);
        const foregroundTone = colorTone(foreground, { allowNeutral: false });
        if (foregroundTone && statusLike) state.text_tone = foregroundTone;
        return state;
    }

    function borderColorInfo(style) {
        if (!style) return null;
        const sides = [
            ['Top', style.borderTopColor],
            ['Right', style.borderRightColor],
            ['Bottom', style.borderBottomColor],
            ['Left', style.borderLeftColor]
        ];
        for (const [side, color] of sides) {
            const width = Number(String(style[`border${side}Width`] || '0').replace('px', ''));
            if (!Number.isFinite(width) || width <= 0) continue;
            const info = parseCssColor(color);
            const tone = colorTone(info);
            if (tone) return { ...info, tone };
        }
        return null;
    }

    function colorSwatchName(el) {
        const labelled = referencedText(el, 'aria-labelledby');
        if (labelled) return { name: labelled, source: 'labelledby' };
        const aria = compact(el.getAttribute('aria-label'));
        if (aria) return { name: aria, source: 'aria' };
        const title = compact(el.getAttribute('title'));
        if (title) return { name: title, source: 'title' };
        return { name: '', source: '' };
    }

    function colorSwatchInfo(el) {
        if (!el || !el.getAttribute) return null;
        const tag = tagNameOf(el);
        if (['html', 'body', 'script', 'style', 'noscript', 'svg', 'canvas', 'img', 'video', 'audio', 'iframe', 'object', 'embed'].includes(tag)) return null;
        if (closestComposed(el, '[aria-hidden="true"]')) return null;
        if (!elementVisible(el)) return null;
        if (directTargetFor(el) || closestComposed(el, clickableSelector)) return null;
        if (compact(el.innerText || el.textContent || '')) return null;
        if (el.querySelector && el.querySelector('img,svg,canvas,video,audio,iframe,object,embed')) return null;
        const rect = rectOf(el);
        if (!rect || rect.width < 4 || rect.height < 4 || rect.width > 56 || rect.height > 56) return null;
        if (rect.width * rect.height > 1800) return null;
        const ratio = Math.max(rect.width, rect.height) / Math.max(1, Math.min(rect.width, rect.height));
        if (ratio > 6) return null;
        let style;
        try {
            style = window.getComputedStyle(el);
        } catch (_) {
            return null;
        }
        const background = parseCssColor(style.backgroundColor);
        let tone = colorTone(background);
        let color = background && background.alpha >= 0.35 ? background.hex : '';
        if (!tone || !color) {
            const border = borderColorInfo(style);
            if (!border) return null;
            tone = border.tone;
            color = border.hex;
        }
        const named = colorSwatchName(el);
        if (!named.name && !statusLikeElement(el)) return null;
        return { tone, color, name: named.name, source: named.source || (statusLikeElement(el) ? 'class' : '') };
    }

    function paintsOverPoint(el) {
        if (!el || el === document.documentElement || el === document.body) return false;
        const style = window.getComputedStyle(el);
        if (!style || style.visibility === 'hidden' || style.display === 'none') return false;
        const opacity = Number(style.opacity || '1');
        if (!Number.isFinite(opacity) || opacity <= 0.05) return false;
        if (['IMG', 'VIDEO', 'CANVAS', 'SVG', 'IFRAME', 'OBJECT', 'EMBED'].includes(el.tagName)) return true;
        if (alphaOf(style.backgroundColor) * opacity >= 0.85) return true;
        const bg = String(style.backgroundImage || '');
        return !!bg && bg !== 'none' && opacity >= 0.85;
    }

    let activeModalRootCache = null;
    function activeModalRoots() {
        if (activeModalRootCache !== null) return activeModalRootCache;
        activeModalRootCache = queryAllDeep('dialog,[role="dialog"],[role="alertdialog"]').filter(el => {
            if (!styleVisible(el)) return false;
            const ariaModal = compact(el.getAttribute('aria-modal')).toLowerCase() === 'true';
            let nativeModal = false;
            try {
                nativeModal = !!(el.matches && el.matches(':modal'));
            } catch (_) {}
            if (!ariaModal && !nativeModal) return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 1 && rect.height > 1 &&
                rect.bottom > 0 && rect.right > 0 &&
                rect.top < window.innerHeight && rect.left < window.innerWidth;
        });
        return activeModalRootCache;
    }

    function blockedByActiveModal(el) {
        const roots = activeModalRoots();
        return roots.length > 0 && !roots.some(root => root === el || containsComposed(root, el));
    }

    function clickHitVisible(el, rect) {
        if (blockedByActiveModal(el)) return false;
        const viewportWidth = window.innerWidth || 0;
        const viewportHeight = window.innerHeight || 0;
        const points = [
            [rect.left + rect.width / 2, rect.top + rect.height / 2],
            [rect.left + Math.min(4, rect.width / 2), rect.top + Math.min(4, rect.height / 2)],
            [rect.right - Math.min(4, rect.width / 2), rect.top + Math.min(4, rect.height / 2)],
            [rect.left + Math.min(4, rect.width / 2), rect.bottom - Math.min(4, rect.height / 2)],
            [rect.right - Math.min(4, rect.width / 2), rect.bottom - Math.min(4, rect.height / 2)]
        ];
        return points.some(([x, y]) => {
            const hit = document.elementFromPoint(
                Math.max(0, Math.min(viewportWidth - 1, x)),
                Math.max(0, Math.min(viewportHeight - 1, y))
            );
            return hit && (
                hit === el ||
                containsComposed(el, hit) ||
                (hit !== document.body && hit !== document.documentElement && containsComposed(hit, el))
            );
        });
    }

    function hitVisible(el, rect) {
        if (blockedByActiveModal(el)) return false;
        const style = window.getComputedStyle(el);
        if (style && style.pointerEvents === 'none') return true;
        if (closestComposed(el, '[inert]')) return true;
        const points = [
            [rect.left + rect.width / 2, rect.top + rect.height / 2],
            [rect.left + Math.min(4, rect.width / 2), rect.top + Math.min(4, rect.height / 2)],
            [rect.right - Math.min(4, rect.width / 2), rect.top + Math.min(4, rect.height / 2)],
            [rect.left + Math.min(4, rect.width / 2), rect.bottom - Math.min(4, rect.height / 2)],
            [rect.right - Math.min(4, rect.width / 2), rect.bottom - Math.min(4, rect.height / 2)]
        ];
        return points.some(([x, y]) => {
            const px = Math.max(0, Math.min((window.innerWidth || 1) - 1, x));
            const py = Math.max(0, Math.min((window.innerHeight || 1) - 1, y));
            const stack = document.elementsFromPoint ? document.elementsFromPoint(px, py) : [document.elementFromPoint(px, py)];
            for (const hit of stack) {
                if (!hit) continue;
                if (
                    hit === el ||
                    containsComposed(el, hit) ||
                    (hit !== document.body && hit !== document.documentElement && containsComposed(hit, el))
                ) return true;
                if (paintsOverPoint(hit)) return false;
            }
            return false;
        });
    }

    function clippedRawRect(raw, el) {
        let left = Math.max(0, raw.left);
        let top = Math.max(0, raw.top);
        let right = Math.min(window.innerWidth || 0, raw.right);
        let bottom = Math.min(window.innerHeight || 0, raw.bottom);
        let cur = el;
        while (cur && cur !== document.documentElement) {
            if (!styleVisible(cur)) return null;
            if (clipsOverflow(cur)) {
                const clip = cur.getBoundingClientRect();
                left = Math.max(left, clip.left);
                top = Math.max(top, clip.top);
                right = Math.min(right, clip.right);
                bottom = Math.min(bottom, clip.bottom);
            }
            cur = composedParent(cur);
        }
        const width = right - left;
        const height = bottom - top;
        if (width <= 1 || height <= 1) return null;
        const clipped = { left, top, right, bottom, width, height };
        return hitVisible(el, clipped) ? clipped : null;
    }

    function clippedGeometryRawRect(raw, el) {
        let left = Math.max(0, raw.left);
        let top = Math.max(0, raw.top);
        let right = Math.min(window.innerWidth || 0, raw.right);
        let bottom = Math.min(window.innerHeight || 0, raw.bottom);
        let cur = el;
        while (cur && cur !== document.documentElement) {
            if (!styleVisible(cur)) return null;
            if (clipsOverflow(cur)) {
                const clip = cur.getBoundingClientRect();
                left = Math.max(left, clip.left);
                top = Math.max(top, clip.top);
                right = Math.min(right, clip.right);
                bottom = Math.min(bottom, clip.bottom);
            }
            cur = composedParent(cur);
        }
        const width = right - left;
        const height = bottom - top;
        return width > 1 && height > 1 ? { left, top, right, bottom, width, height } : null;
    }

    function visibleClientRects(el) {
        return Array.from(el.getClientRects())
            .map(rect => clippedRawRect(rect, el))
            .filter(rect => rect && rect.width > 2 && rect.height > 2);
    }

    function displayContentsVisibleRects(el) {
        let style;
        try {
            style = window.getComputedStyle(el);
        } catch (_) {
            return [];
        }
        if (!style || style.display !== 'contents') return [];
        const rects = [];
        const pushRect = (owner, raw) => {
            if (!owner || !raw || raw.width <= 1 || raw.height <= 1) return;
            const rect = clippedRawRect(raw, owner);
            if (rect && rect.width > 2 && rect.height > 2) rects.push(rect);
        };
        for (const child of queryAllDeep('*', el)) {
            if (!child || !styleVisible(child)) continue;
            for (const raw of Array.from(child.getClientRects())) {
                pushRect(child, raw);
            }
        }
        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
            acceptNode(node) {
                const parent = node.parentElement;
                if (!parent || hasSkippedAncestor(parent) || !styleVisible(parent)) return NodeFilter.FILTER_REJECT;
                if (!(node.nodeValue || '').trim()) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        });
        while (walker.nextNode()) {
            const node = walker.currentNode;
            const parent = node.parentElement;
            if (!parent) continue;
            const range = document.createRange();
            range.selectNodeContents(node);
            for (const raw of Array.from(range.getClientRects())) {
                pushRect(parent, raw);
            }
            range.detach && range.detach();
        }
        return rects.sort((a, b) => (a.top - b.top) || (a.left - b.left));
    }

    function targetVisibleRects(el) {
        const rects = visibleClientRects(el);
        return rects.length ? rects : displayContentsVisibleRects(el);
    }

    function elementVisible(el) {
        if (!styleVisible(el)) return false;
        return visibleClientRects(el).length > 0;
    }

    function rectIntersectsViewport(rect) {
        return rect.width > 0 && rect.height > 0 &&
            rect.bottom > 0 && rect.right > 0 &&
            rect.top < window.innerHeight && rect.left < window.innerWidth;
    }

    function clippedRect(raw) {
        const left = Math.max(0, raw.left);
        const top = Math.max(0, raw.top);
        const right = Math.min(window.innerWidth || 0, raw.right);
        const bottom = Math.min(window.innerHeight || 0, raw.bottom);
        return {
            x: Math.round(left),
            y: Math.round(top),
            width: Math.max(0, Math.round(right - left)),
            height: Math.max(0, Math.round(bottom - top))
        };
    }

    function rectOf(el) {
        if (tagNameOf(el) === 'area') {
            const areaRect = areaRectOf(el);
            if (areaRect) return areaRect;
        }
        const rects = targetVisibleRects(el).sort((a, b) => (a.top - b.top) || (a.left - b.left));
        const rect = rects[0] || el.getBoundingClientRect();
        const clipped = clippedRect(rect);
        return {
            ...clipped,
            center: {
                x: Math.round(clipped.x + clipped.width / 2),
                y: Math.round(clipped.y + clipped.height / 2)
            },
            raw_y: Math.round(rect.top)
        };
    }

    function areaOwnerImage(area) {
        const map = area.closest && area.closest('map[name]');
        const name = map ? compact(map.getAttribute('name')) : '';
        if (!name) return null;
        for (const img of queryAllDeep('img[usemap]')) {
            const usemap = compact(img.getAttribute('usemap')).replace(/^#/, '');
            if (usemap === name && styleVisible(img) && clippedGeometryRawRect(img.getBoundingClientRect(), img)) return img;
        }
        return null;
    }

    function areaRectOf(area) {
        const img = areaOwnerImage(area);
        if (!img) return null;
        const imageRaw = img.getBoundingClientRect();
        const visibleImage = clippedGeometryRawRect(imageRaw, img);
        if (!visibleImage) return null;
        const coords = compact(area.getAttribute('coords'))
            .split(',')
            .map(value => Number(value.trim()))
            .filter(value => Number.isFinite(value));
        if (!coords.length) return null;
        const shape = compact(area.getAttribute('shape') || 'rect').toLowerCase();
        const scaleX = imageRaw.width / Math.max(1, img.naturalWidth || imageRaw.width || 1);
        const scaleY = imageRaw.height / Math.max(1, img.naturalHeight || imageRaw.height || 1);
        let left;
        let top;
        let right;
        let bottom;
        if (shape === 'circle' && coords.length >= 3) {
            const [cx, cy, radius] = coords;
            left = imageRaw.left + (cx - radius) * scaleX;
            top = imageRaw.top + (cy - radius) * scaleY;
            right = imageRaw.left + (cx + radius) * scaleX;
            bottom = imageRaw.top + (cy + radius) * scaleY;
        } else if (shape === 'poly' && coords.length >= 6) {
            const xs = coords.filter((_, index) => index % 2 === 0);
            const ys = coords.filter((_, index) => index % 2 === 1);
            left = imageRaw.left + Math.min(...xs) * scaleX;
            top = imageRaw.top + Math.min(...ys) * scaleY;
            right = imageRaw.left + Math.max(...xs) * scaleX;
            bottom = imageRaw.top + Math.max(...ys) * scaleY;
        } else if (coords.length >= 4) {
            const [x1, y1, x2, y2] = coords;
            left = imageRaw.left + Math.min(x1, x2) * scaleX;
            top = imageRaw.top + Math.min(y1, y2) * scaleY;
            right = imageRaw.left + Math.max(x1, x2) * scaleX;
            bottom = imageRaw.top + Math.max(y1, y2) * scaleY;
        } else {
            return null;
        }
        const raw = {
            left: Math.max(left, visibleImage.left),
            top: Math.max(top, visibleImage.top),
            right: Math.min(right, visibleImage.right),
            bottom: Math.min(bottom, visibleImage.bottom)
        };
        raw.width = raw.right - raw.left;
        raw.height = raw.bottom - raw.top;
        if (raw.width <= 2 || raw.height <= 2) return null;
        const centerX = Math.max(0, Math.min(window.innerWidth - 1, raw.left + raw.width / 2));
        const centerY = Math.max(0, Math.min(window.innerHeight - 1, raw.top + raw.height / 2));
        const hit = document.elementFromPoint(centerX, centerY);
        if (hit !== area && hit !== img) return null;
        const clipped = clippedRect(raw);
        return {
            ...clipped,
            center: {
                x: Math.round(clipped.x + clipped.width / 2),
                y: Math.round(clipped.y + clipped.height / 2)
            },
            raw_y: Math.round(raw.top)
        };
    }

    function mergeRects(left, right) {
        if (!left) return right;
        if (!right) return left;
        const x1 = Math.min(left.x, right.x);
        const y1 = Math.min(left.y, right.y);
        const x2 = Math.max(left.x + left.width, right.x + right.width);
        const y2 = Math.max(left.y + left.height, right.y + right.height);
        return {
            x: Math.round(x1),
            y: Math.round(y1),
            width: Math.round(x2 - x1),
            height: Math.round(y2 - y1)
        };
    }

    function rectOfTextNode(node) {
        const parent = node.parentElement;
        if (!parent) return null;
        const range = document.createRange();
        range.selectNodeContents(node);
        const rects = Array.from(range.getClientRects())
            .map(rect => readableTextRect(rect, parent))
            .filter(rect => rect && rect.width > 0 && rect.height > 0)
            .map(clippedRect);
        range.detach && range.detach();
        return rects.reduce((acc, rect) => mergeRects(acc, rect), null);
    }

    function readableTextRect(raw, parent) {
        const clipped = clippedRawRect(raw, parent);
        if (!clipped) return null;
        const minHeight = Math.min(8, Math.max(3, raw.height * 0.55));
        const minWidth = Math.min(4, Math.max(2, raw.width * 0.2));
        if (clipped.height < minHeight || clipped.width < minWidth) return null;
        return clipped;
    }

    function cssUnquote(value) {
        const raw = String(value || '').trim();
        const quote = raw[0];
        if ((quote !== '"' && quote !== "'") || raw[raw.length - 1] !== quote) return raw;
        return raw.slice(1, -1);
    }

    function formatCounterValue(value, styleName) {
        const style = compact(styleName || 'decimal').toLowerCase();
        const number = Number(value);
        if (!Number.isFinite(number)) return '';
        if (style === 'lower-alpha' || style === 'lower-latin') return alphaOrdinal(number, false);
        if (style === 'upper-alpha' || style === 'upper-latin') return alphaOrdinal(number, true);
        if (style === 'lower-roman') return romanOrdinal(number, false);
        if (style === 'upper-roman') return romanOrdinal(number, true);
        if (style === 'decimal-leading-zero') return String(number).padStart(2, '0');
        return String(number);
    }

    function parseCounterPairs(value, defaultValue) {
        const raw = compact(value);
        if (!raw || raw === 'none') return [];
        const tokens = raw.split(/\\s+/).filter(Boolean);
        const pairs = [];
        for (let index = 0; index < tokens.length; index += 1) {
            const name = tokens[index];
            if (!name || /^-?\\d+$/.test(name) || /^reversed\\(/i.test(name)) continue;
            let amount = defaultValue;
            const next = tokens[index + 1];
            if (next !== undefined && /^-?\\d+$/.test(next)) {
                amount = Number(next);
                index += 1;
            }
            pairs.push([name, amount]);
        }
        return pairs;
    }

    function counterStoreCurrent(store, name) {
        const stack = store.get(name);
        return stack && stack.length ? stack[stack.length - 1] : null;
    }

    function counterStoreValue(store, name) {
        const value = counterStoreCurrent(store, name);
        return value === null ? null : value;
    }

    function counterStoreValues(store, name) {
        const stack = store.get(name);
        return stack && stack.length ? [...stack] : [];
    }

    function pushCounter(store, localResets, name, value) {
        let stack = store.get(name);
        if (!stack) {
            stack = [];
            store.set(name, stack);
        }
        stack.push(value);
        localResets.push(name);
    }

    function setCounter(store, localResets, name, value) {
        let stack = store.get(name);
        if (!stack || !stack.length) {
            pushCounter(store, localResets, name, value);
            return;
        }
        stack[stack.length - 1] = value;
    }

    function incrementCounter(store, localResets, name, amount) {
        let stack = store.get(name);
        if (!stack || !stack.length) {
            pushCounter(store, localResets, name, 0);
            stack = store.get(name);
        }
        stack[stack.length - 1] += amount;
    }

    function applyCounterStyle(style, store, localResets) {
        for (const [name, value] of parseCounterPairs(style.counterReset, 0)) {
            pushCounter(store, localResets, name, value);
        }
        for (const [name, value] of parseCounterPairs(style.counterSet, 0)) {
            setCounter(store, localResets, name, value);
        }
        for (const [name, amount] of parseCounterPairs(style.counterIncrement, 1)) {
            incrementCounter(store, localResets, name, amount);
        }
    }

    function popCounterResets(store, localResets) {
        for (let index = localResets.length - 1; index >= 0; index -= 1) {
            const name = localResets[index];
            const stack = store.get(name);
            if (stack) stack.pop();
        }
    }

    function counterTraversalRoot(el) {
        const root = el && el.getRootNode && el.getRootNode();
        if (root && root !== document && root.children) return root;
        return document.body || document.documentElement;
    }

    function counterValueAt(el, name, pseudo) {
        if (name === 'list-item') {
            return orderedListValue(el);
        }
        const store = new Map();
        const root = counterTraversalRoot(el);
        function applyPseudoCounter(node, pseudoName, localResets = []) {
            try {
                applyCounterStyle(window.getComputedStyle(node, pseudoName), store, localResets);
            } catch (_) {}
        }
        function walk(node) {
            if (!node) return false;
            if (node.nodeType !== Node.ELEMENT_NODE) {
                for (const child of Array.from(node.children || [])) {
                    if (walk(child)) return true;
                }
                return false;
            }
            const localResets = [];
            try {
                applyCounterStyle(window.getComputedStyle(node), store, localResets);
            } catch (_) {}
            if (node === el && pseudo === '::before') {
                applyPseudoCounter(node, '::before', localResets);
                return true;
            }
            applyPseudoCounter(node, '::before', localResets);
            for (const child of Array.from(node.children || [])) {
                if (walk(child)) return true;
            }
            if (node === el && pseudo === '::after') {
                applyPseudoCounter(node, '::after', localResets);
                return true;
            }
            applyPseudoCounter(node, '::after', localResets);
            if (node === el && pseudo === '::marker') {
                return true;
            }
            popCounterResets(store, localResets);
            return false;
        }
        walk(root);
        return counterStoreValue(store, name);
    }

    function counterValuesAt(el, name, pseudo) {
        if (name === 'list-item') {
            const value = orderedListValue(el);
            return value === null ? [] : [value];
        }
        const store = new Map();
        const root = counterTraversalRoot(el);
        function applyPseudoCounter(node, pseudoName, localResets = []) {
            try {
                applyCounterStyle(window.getComputedStyle(node, pseudoName), store, localResets);
            } catch (_) {}
        }
        function walk(node) {
            if (!node) return false;
            if (node.nodeType !== Node.ELEMENT_NODE) {
                for (const child of Array.from(node.children || [])) {
                    if (walk(child)) return true;
                }
                return false;
            }
            const localResets = [];
            try {
                applyCounterStyle(window.getComputedStyle(node), store, localResets);
            } catch (_) {}
            if (node === el && pseudo === '::before') {
                applyPseudoCounter(node, '::before', localResets);
                return true;
            }
            applyPseudoCounter(node, '::before', localResets);
            for (const child of Array.from(node.children || [])) {
                if (walk(child)) return true;
            }
            if (node === el && pseudo === '::after') {
                applyPseudoCounter(node, '::after', localResets);
                return true;
            }
            applyPseudoCounter(node, '::after', localResets);
            if (node === el && pseudo === '::marker') {
                return true;
            }
            popCounterResets(store, localResets);
            return false;
        }
        walk(root);
        return counterStoreValues(store, name);
    }

    function splitCssArgs(value) {
        const args = [];
        let current = '';
        let quote = '';
        let depth = 0;
        for (const char of String(value || '')) {
            if (quote) {
                current += char;
                if (char === quote) quote = '';
                continue;
            }
            if (char === '"' || char === "'") {
                quote = char;
                current += char;
                continue;
            }
            if (char === '(') depth += 1;
            if (char === ')') depth = Math.max(0, depth - 1);
            if (char === ',' && depth === 0) {
                args.push(current.trim());
                current = '';
                continue;
            }
            current += char;
        }
        if (current.trim()) args.push(current.trim());
        return args;
    }

    function resolveCssCounterToken(fn, argsText, el, pseudo) {
        const args = splitCssArgs(argsText);
        const name = compact(args[0]);
        if (!name) return null;
        if (fn === 'counter') {
            const value = counterValueAt(el, name, pseudo);
            if (value === null) return null;
            return formatCounterValue(value, args[1] || 'decimal');
        }
        const separator = cssUnquote(args[1] || ' ');
        const styleName = args[2] || 'decimal';
        const values = counterValuesAt(el, name, pseudo);
        if (!values.length) return null;
        return values.map(value => formatCounterValue(value, styleName)).join(separator);
    }

    function resolveCssAttrToken(argsText, el) {
        if (!el || !el.getAttribute) return null;
        const args = splitCssArgs(argsText);
        const rawName = compact(args[0] || '');
        if (!rawName) return null;
        const attrName = cssUnquote(rawName.split(/\\s+/)[0] || '');
        if (!attrName || !/^[A-Za-z_:-][\\w:.-]*$/.test(attrName)) return null;
        const value = compact(el.getAttribute(attrName) || '');
        if (value) return value;
        if (args.length > 1) {
            const fallback = cssStringContent(args.slice(1).join(','), el, '');
            return fallback || '';
        }
        return '';
    }

    function cssStringContent(value, el = null, pseudo = '') {
        const raw = String(value || '').trim();
        if (!raw || ['none', 'normal', 'contents', 'initial', 'inherit', 'unset'].includes(raw)) return '';
        if (/^(url|image)\\(/i.test(raw)) return '';
        let textWithFunctions = raw;
        if (/\\bcounters?\\(/i.test(textWithFunctions)) {
            let missingCounter = false;
            textWithFunctions = textWithFunctions.replace(
                /(counters?)\\(\\s*([^()]*(?:\\([^)]*\\)[^()]*)*)\\s*\\)/gi,
                (_match, fn, args) => {
                    const resolved = resolveCssCounterToken(String(fn).toLowerCase(), args, el, pseudo);
                    if (resolved === null) {
                        missingCounter = true;
                        return '';
                    }
                    return '"' + String(resolved).replace(/["\\\\]/g, '') + '"';
                },
            );
            if (missingCounter) return '';
        }
        if (/\\battr\\(/i.test(textWithFunctions)) {
            let missingAttr = false;
            textWithFunctions = textWithFunctions.replace(
                /attr\\(\\s*([^()]*(?:\\([^)]*\\)[^()]*)*)\\s*\\)/gi,
                (_match, args) => {
                    const resolved = resolveCssAttrToken(args, el);
                    if (resolved === null) {
                        missingAttr = true;
                        return '';
                    }
                    return '"' + String(resolved).replace(/["\\\\]/g, '') + '"';
                },
            );
            if (missingAttr) return '';
        }
        if (textWithFunctions !== raw) {
            return cssStringContent(textWithFunctions, el, pseudo);
        }
        const pieces = [];
        const quoted = /(['"])((?:\\\\.|(?!\\1).)*)\\1/g;
        let match;
        while ((match = quoted.exec(raw)) !== null) {
            pieces.push(match[2].replace(/\\\\(["'\\\\])/g, '$1'));
        }
        const text = pieces.length ? pieces.join('') : raw;
        const compacted = compact(text.replace(/\\\\[0-9a-f]{1,6}\\s?/gi, ' '));
        if (!compacted) return '';
        if (/[\uE000-\uF8FF]/.test(compacted)) return '';
        if (!/[\\p{L}\\p{N}]/u.test(compacted)) return '';
        return compacted;
    }

    function pseudoTextFor(el, pseudo) {
        if (!el || hasSkippedAncestor(el) || !styleVisible(el)) return '';
        try {
            return applyTextTransform(cssStringContent(window.getComputedStyle(el, pseudo).content, el, pseudo), el, pseudo);
        } catch (_) {
            return '';
        }
    }

    function pseudoText(el) {
        return compact([pseudoTextFor(el, '::before'), pseudoTextFor(el, '::after')].filter(Boolean).join(' '));
    }

    function roleOf(el) {
        const role = compact(el.getAttribute('role')).toLowerCase();
        if (role) return role;
        const tag = el.tagName.toLowerCase();
        if (tag === 'a') return 'link';
        if (tag === 'area') return 'link';
        if (tag === 'button') return 'button';
        if (tag === 'select') return 'select';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'summary') return 'summary';
        if (el.isContentEditable) return 'textbox';
        if (tag === 'input') {
            const type = (el.getAttribute('type') || 'text').toLowerCase();
            if (['button', 'submit', 'reset', 'image'].includes(type)) return 'button';
            if (type === 'range') return 'slider';
            if (type === 'number') return 'spinbutton';
            if (['checkbox', 'radio', 'color', 'file'].includes(type)) return type;
            return 'textbox';
        }
        if (el.hasAttribute('onclick') || typeof el.onclick === 'function' || el.hasAttribute('popovertarget')) return 'button';
        if (el.hasAttribute('tabindex') && (el.tabIndex || 0) >= 0) return 'focusable';
        return tag;
    }

    function labelText(el) {
        if (!el.labels || !el.labels.length) return '';
        return compact(Array.from(el.labels).map(label =>
            visibleTextWithin(label, true) || label.innerText || label.textContent || label.getAttribute('aria-label') || ''
        ).join(' '));
    }

    function elementByIdInScope(el, id) {
        const root = el.getRootNode && el.getRootNode();
        if (root && root.getElementById) {
            const found = root.getElementById(id);
            if (found) return found;
        }
        if (root && root.querySelector) {
            try {
                const found = root.querySelector(`#${CSS.escape(id)}`);
                if (found) return found;
            } catch (_) {}
        }
        return document.getElementById(id);
    }

    function referencedText(el, attr) {
        const ids = compact(el.getAttribute(attr)).split(/\\s+/).filter(Boolean);
        if (!ids.length) return '';
        const text = ids.map(id => {
            const ref = elementByIdInScope(el, id);
            if (!ref || hasSkippedAncestor(ref)) return '';
            return visibleTextWithin(ref, true) || ref.innerText || ref.textContent || ref.getAttribute('aria-label') || ref.title || '';
        }).join(' ');
        return compact(text);
    }

    function referencedVisibleText(el, attr) {
        const ids = compact(el.getAttribute(attr)).split(/\\s+/).filter(Boolean);
        if (!ids.length) return '';
        const text = ids.map(id => {
            const ref = elementByIdInScope(el, id);
            if (!ref || hasSkippedAncestor(ref) || !styleVisible(ref)) return '';
            return visibleTextWithin(ref, true);
        }).join(' ');
        return compact(text);
    }

    function describedText(el) {
        const parts = [
            referencedVisibleText(el, 'aria-describedby'),
            el.getAttribute('aria-description')
        ].filter(Boolean);
        return compact(parts.join(' '));
    }

    function errorMessageText(el) {
        return referencedVisibleText(el, 'aria-errormessage');
    }

    function activeDescendantText(el) {
        const id = compact(el.getAttribute('aria-activedescendant'));
        if (!id) return '';
        const target = elementByIdInScope(el, id);
        if (!target || !elementVisible(target) || hasSkippedAncestor(target)) return '';
        return compact(
            referencedText(target, 'aria-labelledby') ||
            target.getAttribute('aria-label') ||
            visibleTextWithin(target, true) ||
            target.innerText ||
            target.textContent ||
            target.getAttribute('title') ||
            ''
        );
    }

    function imageText(el) {
        const img = el.matches && el.matches('img') ? el : el.querySelector && el.querySelector('img[alt],img[aria-label]');
        if (!img) return '';
        return compact(img.getAttribute('alt') || img.getAttribute('aria-label') || '');
    }

    function meaningfulGraphicText(value) {
        const text = compact(value);
        if (!text) return '';
        if (/[\uE000-\uF8FF]/.test(text)) return '';
        if (!/[\\p{L}\\p{N}]/u.test(text)) return '';
        return text;
    }

    function graphicVisible(el) {
        if (!el || closestComposed(el, '[aria-hidden="true"]')) return false;
        const role = compact(el.getAttribute('role')).toLowerCase();
        if (role === 'presentation' || role === 'none') return false;
        const style = window.getComputedStyle(el);
        if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
        if (Number(style.opacity || '1') === 0) return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 1 && rect.height > 1 &&
            rect.bottom > 0 && rect.right > 0 &&
            rect.top < window.innerHeight && rect.left < window.innerWidth;
    }

    function graphicText(el) {
        const roots = [];
        if (el && el.matches && el.matches('svg,[role="img"]')) roots.push(el);
        if (el && el.querySelectorAll) {
            roots.push(...Array.from(el.querySelectorAll('svg,[role="img"]')));
        }
        for (const graphic of roots) {
            if (!graphicVisible(graphic)) continue;
            const label = meaningfulGraphicText(
                graphic.getAttribute('aria-label') ||
                graphic.getAttribute('alt') ||
                graphic.getAttribute('title') ||
                ''
            );
            if (label) return label;
            const labelled = referencedText(graphic, 'aria-labelledby');
            const labelledText = meaningfulGraphicText(labelled);
            if (labelledText) return labelledText;
            const title = graphic.querySelector && graphic.querySelector('title');
            const titleText = meaningfulGraphicText(title ? title.textContent : '');
            if (titleText) return titleText;
        }
        return '';
    }

    function visualLabelVisible(el) {
        if (!el || closestComposed(el, '[aria-hidden="true"]')) return false;
        const role = compact(el.getAttribute('role')).toLowerCase();
        if (role === 'presentation' || role === 'none') return false;
        let style;
        try {
            style = window.getComputedStyle(el);
        } catch (_) {
            return false;
        }
        if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
        if (Number(style.opacity || '1') === 0) return false;
        const rect = el.getBoundingClientRect();
        return rect.width > 1 && rect.height > 1 &&
            rect.bottom > 0 && rect.right > 0 &&
            rect.top < window.innerHeight && rect.left < window.innerWidth;
    }

    function visualElementText(el) {
        const selector = 'img,canvas,video,object,embed,svg,[role="img"]';
        const roots = [];
        if (el && el.matches && el.matches(selector)) roots.push(el);
        if (el && el.querySelectorAll) {
            roots.push(...queryAllDeep(selector, el));
        }
        const seenVisuals = new Set();
        for (const visual of roots) {
            if (!visual || seenVisuals.has(visual) || !visualLabelVisible(visual)) continue;
            seenVisuals.add(visual);
            const direct = meaningfulGraphicText(
                visual.getAttribute('alt') ||
                visual.getAttribute('aria-label') ||
                visual.getAttribute('title') ||
                ''
            );
            if (direct) return direct;
            const labelled = meaningfulGraphicText(referencedText(visual, 'aria-labelledby'));
            if (labelled) return labelled;
            const svgTitle = visual.querySelector && visual.querySelector('title');
            const titleText = meaningfulGraphicText(svgTitle ? svgTitle.textContent : '');
            if (titleText) return titleText;
        }
        return '';
    }

    function formControlName(el) {
        if (!el.matches || !el.matches('input,textarea,select')) return '';
        const name = compact(el.getAttribute('name'));
        return /^[\\p{L}\\p{N}_.:-]{2,80}$/u.test(name) ? name.replace(/[_:-]+/g, ' ') : '';
    }

    function formSubmitText(el) {
        const tag = el.tagName;
        const type = (el.getAttribute('type') || '').toLowerCase();
        const submitLike = tag === 'BUTTON' || (tag === 'INPUT' && ['submit', 'image'].includes(type));
        if (!submitLike) return '';
        const form = el.form || el.closest('form');
        if (!form) return '';
        const fields = Array.from(form.querySelectorAll('input:not([type="hidden"]), textarea, select'))
            .filter(field => field !== el && elementVisible(field));
        for (const field of fields) {
            const fieldName = compact(labelText(field) || field.getAttribute('placeholder') || field.getAttribute('name') || '');
            if (fieldName) return `submit ${fieldName}`;
        }
        return type === 'submit' ? 'submit' : '';
    }

    function optionText(option) {
        return compact(
            option && (
                option.label ||
                option.textContent ||
                option.getAttribute && option.getAttribute('value') ||
                ''
            )
        );
    }

    function selectedOptionTexts(el) {
        if (!el || el.tagName !== 'SELECT') return [];
        return Array.from(el.selectedOptions || [])
            .map(optionText)
            .filter(Boolean);
    }

    function optionListText(values) {
        return values.map(text => compact(text)).filter(Boolean).join(' | ');
    }

    function visibleSelectOptions(el) {
        if (!el || el.tagName !== 'SELECT') return [];
        const options = Array.from(el.options || []);
        if (!options.length) return [];
        const declaredSize = Number(el.getAttribute('size') || 0);
        const idlSize = Number(el.size || 0);
        const listLike = !!el.multiple || declaredSize > 1 || idlSize > 1;
        if (!listLike) return [];
        const rawRect = el.getBoundingClientRect();
        const height = Math.max(1, el.clientHeight || rawRect.height || 0);
        const declaredRows = Math.max(declaredSize, idlSize, 0);
        const estimatedRows = declaredRows > 1
            ? declaredRows
            : Math.max(1, Math.round(height / 20));
        const visibleRows = Math.max(1, Math.min(options.length, estimatedRows));
        const optionHeight = Math.max(1, height / visibleRows);
        const start = Math.max(0, Math.min(options.length - 1, Math.floor((el.scrollTop || 0) / optionHeight)));
        const count = Math.max(1, Math.min(options.length - start, Math.ceil(height / optionHeight)));
        return options.slice(start, start + count);
    }

    function visibleSelectOptionTexts(el) {
        return visibleSelectOptions(el).map(optionText).filter(Boolean);
    }

    function optionDisabled(option) {
        if (!option) return false;
        const group = option.parentElement && tagNameOf(option.parentElement) === 'optgroup'
            ? option.parentElement
            : null;
        return !!option.disabled || !!(group && group.disabled);
    }

    function visibleDisabledSelectOptionTexts(el) {
        return visibleSelectOptions(el)
            .filter(optionDisabled)
            .map(optionText)
            .filter(Boolean);
    }

    function visibleName(el) {
        if (el.tagName === 'INPUT') {
            const type = (el.getAttribute('type') || 'text').toLowerCase();
            if (['button', 'submit', 'reset'].includes(type)) {
                const value = compact(el.value || el.getAttribute('value') || '');
                if (value) return value;
            }
        }
        if (el.tagName === 'SELECT') {
            const selected = optionListText(selectedOptionTexts(el));
            if (selected) return selected;
        }
        return compact([pseudoTextFor(el, '::before'), visibleTextWithin(el), pseudoTextFor(el, '::after')].filter(Boolean).join(' '));
    }

    function nativeControlNameCandidates(el) {
        if (!el.matches || !el.matches('input,textarea,select')) return null;
        const type = (el.getAttribute('type') || 'text').toLowerCase();
        if (el.tagName === 'INPUT' && ['button', 'submit', 'reset', 'image'].includes(type)) return null;
        return [
            ['labelledby', referencedText(el, 'aria-labelledby')],
            ['aria', el.getAttribute('aria-label')],
            ['label', labelText(el)],
            ['title', el.getAttribute('title')],
            ['placeholder', el.getAttribute('placeholder')],
            ['name', formControlName(el)],
            ['visible', visibleName(el)],
            ['value', el.value || el.getAttribute('value')]
        ];
    }

    function hrefTail(el) {
        const href = el.href || el.getAttribute('href') || '';
        if (!href) return '';
        try {
            const url = new URL(href, location.href);
            return decodeURIComponent((url.pathname.split('/').filter(Boolean).pop() || url.hostname || '').replace(/[-_]+/g, ' '));
        } catch (_) {
            return '';
        }
    }

    function targetName(el) {
        const contentEditable = !!el.isContentEditable;
        const nativeCandidates = nativeControlNameCandidates(el);
        const candidates = contentEditable ? [
            ['labelledby', referencedText(el, 'aria-labelledby')],
            ['aria', el.getAttribute('aria-label')],
            ['label', labelText(el)],
            ['title', el.getAttribute('title')],
            ['placeholder', el.getAttribute('placeholder')]
        ] : nativeCandidates || [
            ['visible', visibleName(el)],
            ['labelledby', referencedText(el, 'aria-labelledby')],
            ['aria', el.getAttribute('aria-label')],
            ['label', labelText(el)],
            ['alt', el.getAttribute('alt') || imageText(el)],
            ['graphic', graphicText(el)],
            ['visual', visualElementText(el)],
            ['title', el.getAttribute('title')],
            ['placeholder', el.getAttribute('placeholder')],
            ['name', formControlName(el)],
            ['value', el.value || el.getAttribute('value')],
            ['form', formSubmitText(el)],
            ['href_tail', hrefTail(el)]
        ];
        for (const [source, value] of candidates) {
            const text = compact(value);
            if ((source === 'visible' || source === 'graphic' || source === 'visual') && text && !/[\\p{L}\\p{N}]/u.test(text)) continue;
            if (text) return { name: text, source };
        }
        return { name: roleOf(el), source: 'fallback' };
    }

    function hrefOf(el) {
        const href = el.href || el.getAttribute('href') || '';
        if (!href) return '';
        try {
            return new URL(href, location.href).href;
        } catch (_) {
            return href;
        }
    }

    function visiblePlaceholder(el) {
        const placeholder = compact(el && el.getAttribute ? el.getAttribute('placeholder') : '');
        if (!placeholder) return '';
        if (compact(el.value || '')) return '';
        const tag = tagNameOf(el);
        if (tag === 'textarea') return placeholder;
        if (tag !== 'input') return '';
        const type = (el.getAttribute('type') || 'text').toLowerCase();
        return ['text', 'search', 'email', 'url', 'tel', 'password', 'number'].includes(type)
            ? placeholder
            : '';
    }

    function nativeValidationState(el) {
        const state = {};
        if (!el || !el.validity || !el.willValidate) return state;
        const validity = el.validity;
        if (!validity || validity.valid) return state;
        const reasons = [];
        for (const [key, reason] of [
            ['badInput', 'bad_input'],
            ['typeMismatch', 'type_mismatch'],
            ['patternMismatch', 'pattern_mismatch'],
            ['rangeOverflow', 'range_overflow'],
            ['rangeUnderflow', 'range_underflow'],
            ['stepMismatch', 'step_mismatch'],
            ['tooLong', 'too_long'],
            ['tooShort', 'too_short'],
            ['customError', 'custom_error']
        ]) {
            if (validity[key]) reasons.push(reason);
        }
        if (!reasons.length) return state;
        state.invalid = true;
        state.invalid_reason = reasons.join('|');
        return state;
    }

    function textVisualState(el) {
        const state = {};
        if (!el || !el.getAttribute) return state;
        for (let node = el; node && node.nodeType === 1; node = composedParent(node)) {
            const tag = tagNameOf(node);
            if (tag === 'del' || tag === 's' || tag === 'strike') state.deleted = true;
            if (tag === 'ins') state.inserted = true;
            if (tag === 'mark') state.marked = true;
            try {
                const style = window.getComputedStyle(node);
                const line = String(style.textDecorationLine || style.textDecoration || '').toLowerCase();
                if (line.includes('line-through')) state.deleted = true;
            } catch (_) {
                // Ignore style access failures on detached or inaccessible nodes.
            }
            if (node === document.body || node === document.documentElement) break;
        }
        return state;
    }

    function controlState(el) {
        const state = {};
        const tag = el.tagName;
        const type = (el.getAttribute('type') || 'text').toLowerCase();
        Object.assign(state, textVisualState(el));
        Object.assign(state, visualToneState(el));
        Object.assign(state, nativeValidationState(el));
        const focused = focusedTarget(el);
        if (focused) state.focused = true;
        for (const [attr, key] of [
            ['aria-checked', 'aria_checked'],
            ['aria-controls', 'aria_controls'],
            ['aria-current', 'aria_current'],
            ['aria-expanded', 'aria_expanded'],
            ['aria-haspopup', 'aria_haspopup'],
            ['aria-invalid', 'aria_invalid'],
            ['aria-autocomplete', 'aria_autocomplete'],
            ['aria-busy', 'aria_busy'],
            ['aria-level', 'aria_level'],
            ['aria-posinset', 'aria_posinset'],
            ['aria-setsize', 'aria_setsize'],
            ['aria-pressed', 'aria_pressed'],
            ['aria-required', 'aria_required'],
            ['aria-sort', 'aria_sort'],
            ['aria-selected', 'aria_selected'],
            ['aria-valuenow', 'aria_value_now'],
            ['aria-valuemin', 'aria_value_min'],
            ['aria-valuemax', 'aria_value_max'],
            ['aria-valuetext', 'aria_value_text']
        ]) {
            const value = compact(el.getAttribute(attr));
            if (value) state[key] = value;
        }
        const activeDescendant = activeDescendantText(el);
        if (activeDescendant) state.active_descendant = activeDescendant;
        const description = describedText(el);
        if (description) state.description = description;
        const errorMessage = errorMessageText(el);
        if (errorMessage) state.error_message = errorMessage;
        if (tag === 'SUMMARY') {
            const details = el.closest('details');
            if (details) state.open = !!details.open;
        }
        const popoverTarget = compact(el.getAttribute('popovertarget'));
        if (popoverTarget) {
            state.popover_target = popoverTarget;
            const action = compact(el.getAttribute('popovertargetaction'));
            if (action) state.popover_action = action;
            const target = elementByIdInScope(el, popoverTarget);
            if (target) {
                try {
                    state.popover_open = !!target.matches(':popover-open');
                } catch (_) {
                    state.popover_open = false;
                }
            }
        }
        if (el.hasAttribute('tabindex')) {
            state.tab_index = String(el.tabIndex);
        }
        if (el.isContentEditable) {
            const mode = compact(el.getAttribute('contenteditable') || 'true').toLowerCase();
            state.contenteditable = mode || 'true';
            const value = compact(visibleTextWithin(el));
            if (value) state.value = value;
        } else if (tag === 'INPUT') {
            if (el.required) state.required = true;
            if (el.readOnly) state.read_only = true;
            if (['checkbox', 'radio'].includes(type)) {
                state.checked = !!el.checked;
                if (type === 'checkbox' && el.indeterminate) state.indeterminate = true;
            } else if (type === 'range' || type === 'number') {
                const value = compact(el.value || '');
                if (value) state.value = value;
                const min = compact(el.min || el.getAttribute('min') || '');
                const max = compact(el.max || el.getAttribute('max') || '');
                if (min) state.value_min = min;
                if (max) state.value_max = max;
            } else if (type === 'file') {
                const files = Array.from(el.files || [])
                    .map(file => compact(file && file.name ? file.name : ''))
                    .filter(Boolean);
                if (files.length) {
                    state.file_count = String(files.length);
                    state.selected_files = optionListText(files);
                }
            } else if (!['password', 'hidden', 'file', 'button', 'submit', 'reset', 'image'].includes(type)) {
                const value = compact(el.value || '');
                if (value) state.value = value;
                if (focused && typeof el.selectionStart === 'number' && typeof el.selectionEnd === 'number') {
                    state.selection_start = String(el.selectionStart);
                    state.selection_end = String(el.selectionEnd);
                    if (el.selectionEnd > el.selectionStart) {
                        const selected = compact(String(el.value || '').slice(el.selectionStart, el.selectionEnd));
                        if (selected) state.text_selection = selected;
                    }
                }
            }
        } else if (tag === 'TEXTAREA') {
            if (el.required) state.required = true;
            if (el.readOnly) state.read_only = true;
            const value = compact(el.value || '');
            if (value) state.value = value;
            if (focused && typeof el.selectionStart === 'number' && typeof el.selectionEnd === 'number') {
                state.selection_start = String(el.selectionStart);
                state.selection_end = String(el.selectionEnd);
                if (el.selectionEnd > el.selectionStart) {
                    const selected = compact(String(el.value || '').slice(el.selectionStart, el.selectionEnd));
                    if (selected) state.text_selection = selected;
                }
            }
        } else if (tag === 'SELECT') {
            if (el.required) state.required = true;
            if (el.multiple) state.multiple = true;
            const selectedTexts = selectedOptionTexts(el);
            const selected = optionListText(selectedTexts);
            const visibleOptions = optionListText(visibleSelectOptionTexts(el));
            const disabledOptions = optionListText(visibleDisabledSelectOptionTexts(el));
            const value = compact(el.value || '');
            if (selected) state.selected_text = selected;
            if (selectedTexts.length > 1) state.selected_options = selected;
            if (visibleOptions) state.visible_options = visibleOptions;
            if (disabledOptions) state.disabled_options = disabledOptions;
            if (value && value !== selected && !el.multiple) state.value = value;
        }
        return state;
    }

    function textState(el) {
        const state = {};
        if (!el || !el.getAttribute) return state;
        Object.assign(state, textVisualState(el));
        Object.assign(state, visualToneState(el));
        for (const [attr, key] of [
            ['aria-current', 'aria_current'],
            ['aria-expanded', 'aria_expanded'],
            ['aria-invalid', 'aria_invalid'],
            ['aria-busy', 'aria_busy'],
            ['aria-pressed', 'aria_pressed'],
            ['aria-posinset', 'aria_posinset'],
            ['aria-setsize', 'aria_setsize'],
            ['aria-sort', 'aria_sort'],
            ['aria-selected', 'aria_selected'],
            ['aria-checked', 'aria_checked']
        ]) {
            const value = compact(el.getAttribute(attr));
            if (value) state[key] = value;
        }
        const live = compact(el.getAttribute('aria-live')).toLowerCase();
        if (live && live !== 'off') state.aria_live = live;
        if (tagNameOf(el) === 'time') {
            const dateTime = compact(el.getAttribute('datetime'));
            if (dateTime) state.datetime = dateTime;
        }
        const level = headingLevel(el);
        if (level) state.level = level;
        else {
            const ariaLevel = compact(el.getAttribute('aria-level'));
            if (ariaLevel) state.aria_level = ariaLevel;
        }
        const describes = describedTargetIndexes(el);
        if (describes) state.describes = describes;
        return state;
    }

    function headingLevel(el) {
        if (explicitRole(el) !== 'heading') return '';
        const level = compact(el.getAttribute('aria-level'));
        return /^[1-9][0-9]*$/.test(level) ? level : '';
    }

    function semanticContext(el) {
        const owner = closestComposed(
            el,
            [
                'dialog',
                'fieldset',
                '[role="dialog"]',
                '[role="alertdialog"]',
                '[role="radiogroup"]',
                '[role="group"]',
                '[role="toolbar"]',
                '[role="listbox"]',
                '[role="menu"]',
                '[role="tablist"]',
                '[role="tree"]'
            ].join(',')
        );
        if (!owner || owner === el || !elementVisible(owner)) return null;
        const tag = tagNameOf(owner);
        const role = tag === 'dialog' ? 'dialog' : (tag === 'fieldset' ? 'group' : explicitRole(owner));
        const legend = tag === 'fieldset'
            ? Array.from(owner.children || []).find(child => tagNameOf(child) === 'legend')
            : null;
        const legendText = legend ? visibleTextWithin(legend, true) : '';
        const name = compact(
            legendText ||
            referencedText(owner, 'aria-labelledby') ||
            owner.getAttribute('aria-label') ||
            owner.getAttribute('title') ||
            ''
        );
        if (!role && !name) return null;
        return { role: role || 'group', name, owner };
    }

    function noInfoFocusable(el, role, named, state) {
        if (role !== 'focusable' || named.source !== 'fallback') return false;
        if (el.hasAttribute('onclick') || typeof el.onclick === 'function') return false;
        const usefulStateKeys = Object.keys(state || {}).filter(key => key !== 'tab_index');
        if (usefulStateKeys.length) return false;
        return true;
    }

    function blockedControlReason(el) {
        if (!el) return '';
        if (closestComposed(el, '[inert]')) return 'inert';
        if (closestComposed(el, '[aria-disabled="true"]')) return 'aria_disabled';
        if (el.disabled) return 'disabled';
        try {
            if (el.matches && el.matches(':disabled')) return 'disabled';
        } catch (_) {}
        if (cssDisabledControlReason(el)) return 'css_disabled';
        return '';
    }

    function cssDisabledControlReason(el) {
        if (!el || !el.matches || !el.matches(cssDisabledControlSelector)) return '';
        const marker = [
            el.className && typeof el.className === 'string' ? el.className : '',
            el.getAttribute('id'),
            el.getAttribute('aria-label'),
            el.getAttribute('title'),
            el.getAttribute('data-state'),
            el.getAttribute('data-disabled'),
            el.getAttribute('data-readonly')
        ].filter(Boolean).join(' ').toLowerCase();
        if (/\\b(disabled|unavailable|readonly|read-only|locked)\\b/.test(marker)) return 'css_disabled';
        try {
            const style = window.getComputedStyle(el);
            if (style && style.pointerEvents === 'none') return 'css_disabled';
        } catch (_) {}
        return '';
    }

    function blockedControlElement(el) {
        if (!el || !blockedControlReason(el)) return false;
        const tag = tagNameOf(el);
        const role = explicitRole(el);
        const controlish =
            ['button', 'input', 'textarea', 'select', 'summary', 'a'].includes(tag) ||
            actionableRoles.includes(role) ||
            el.hasAttribute('onclick') ||
            (el.hasAttribute('tabindex') && (el.tabIndex || 0) >= 0);
        return controlish;
    }

    function loadingPlaceholderInfo(el) {
        if (!el || !el.matches || !el.matches(loadingPlaceholderSelector)) return null;
        if (directTargetFor(el) || closestComposed(el, clickableSelector)) return null;
        if (!elementVisible(el)) return null;
        if (el.querySelectorAll) {
            const child = Array.from(el.querySelectorAll(loadingPlaceholderSelector))
                .find(item => item !== el && elementVisible(item));
            if (child) return null;
        }
        if (compact(visibleTextWithin(el, true))) return null;
        const marker = [
            el.className && typeof el.className === 'string' ? el.className : '',
            el.getAttribute('aria-label'),
            el.getAttribute('title'),
            el.getAttribute('data-state'),
            el.getAttribute('data-loading')
        ].filter(Boolean).join(' ').toLowerCase();
        let variant = 'loading';
        if (/\\b(skeleton|shimmer)\\b/.test(marker)) variant = 'skeleton';
        else if (/\\b(spinner|loader)\\b/.test(marker)) variant = 'spinner';
        const labelled = referencedText(el, 'aria-labelledby');
        const aria = compact(el.getAttribute('aria-label'));
        const title = compact(el.getAttribute('title'));
        const named = compact(labelled || aria || title || '');
        return {
            variant,
            name: named || (variant === 'skeleton' ? 'loading placeholder' : 'loading'),
            source: labelled ? 'labelledby' : (aria ? 'aria' : (title ? 'title' : 'class'))
        };
    }

    function busyTextOwnerElement(el) {
        const textTags = new Set([
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'dt', 'dd',
            'pre', 'blockquote', 'figcaption', 'caption', 'th', 'td',
            'label', 'legend', 'time'
        ]);
        const textRoles = new Set([
            'heading', 'paragraph', 'listitem', 'cell', 'text',
            'alert', 'status', 'note', 'log', 'timer'
        ]);
        return textTags.has(tagNameOf(el)) || textRoles.has(explicitRole(el));
    }

    function indicatorKind(el) {
        if (blockedControlElement(el)) {
            return blockedControlReason(el) === 'inert' ? 'inert_control' : 'disabled_control';
        }
        if (loadingPlaceholderInfo(el)) return 'loading_placeholder';
        if (semanticGraphicElement(el)) return 'graphic';
        const role = explicitRole(el);
        if (role === 'meter') return 'meter';
        if (role === 'progressbar') return 'progressbar';
        const tag = tagNameOf(el);
        if (tag === 'meter') return 'meter';
        if (tag === 'progress') return 'progressbar';
        if (tag === 'audio' || tag === 'video') return tag;
        if (compact(el.getAttribute('aria-busy')).toLowerCase() === 'true') return 'busy';
        return role || tag;
    }

    function semanticGraphicElement(el) {
        return !!el && (explicitRole(el) === 'img' || tagNameOf(el) === 'svg');
    }

    function nearestSemanticGraphic(el) {
        const graphic = closestComposed(el, '[role="img"],svg');
        return graphic && semanticGraphicElement(graphic) ? graphic : null;
    }

    function semanticGraphicName(el) {
        const kind = indicatorKind(el);
        if (kind !== 'graphic') return '';
        return meaningfulGraphicText(graphicText(el));
    }

    function indicatorName(el) {
        return compact(
            semanticGraphicName(el) ||
            referencedText(el, 'aria-labelledby') ||
            el.getAttribute('aria-label') ||
            labelText(el) ||
            el.getAttribute('title') ||
            ''
        );
    }

    function numericElementAttr(el, attr) {
        if (!el || !el.hasAttribute || !el.hasAttribute(attr)) return '';
        const value = Number(el.getAttribute(attr));
        return Number.isFinite(value) ? String(value) : compact(el.getAttribute(attr));
    }

    function mediaNumber(value) {
        const number = Number(value);
        if (!Number.isFinite(number)) return '';
        return String(Math.round(number * 1000) / 1000);
    }

    function indicatorState(el) {
        const state = controlState(el);
        const tag = tagNameOf(el);
        const blockedReason = blockedControlReason(el);
        if (blockedControlElement(el)) {
            state.blocked = blockedReason;
            if (blockedReason === 'inert') {
                state.inert = true;
            } else {
                state.disabled = true;
                if (blockedReason === 'aria_disabled') state.aria_disabled = true;
            }
            if (tag === 'input') {
                state.input_type = (el.getAttribute('type') || 'text').toLowerCase();
            }
        }
        if (tag === 'progress') {
            if (el.hasAttribute('value')) state.value = String(el.value);
            else state.indeterminate = true;
            const max = numericElementAttr(el, 'max') || (el.max ? String(el.max) : '');
            if (max) state.value_max = max;
            state.value_min = state.value_min || '0';
        } else if (tag === 'meter') {
            state.value = String(el.value);
            state.value_min = numericElementAttr(el, 'min') || String(el.min);
            state.value_max = numericElementAttr(el, 'max') || String(el.max);
            for (const attr of ['low', 'high', 'optimum']) {
                const value = numericElementAttr(el, attr);
                if (value) state[`value_${attr}`] = value;
            }
        } else if (tag === 'audio' || tag === 'video') {
            state.controls = true;
            state.paused = !!el.paused;
            state.muted = !!el.muted;
            const currentTime = mediaNumber(el.currentTime);
            const duration = mediaNumber(el.duration);
            if (currentTime) state.current_time = currentTime;
            if (duration) state.duration = duration;
            if (el.loop) state.loop = true;
            if (el.autoplay) state.autoplay = true;
            const playbackRate = mediaNumber(el.playbackRate);
            if (playbackRate && playbackRate !== '1') state.playback_rate = playbackRate;
            const volume = mediaNumber(el.volume);
            if (volume && volume !== '1') state.volume = volume;
        }
        return state;
    }

    function meaningfulIndicator(item) {
        if (compact(item.name) || compact(item.text)) return true;
        return [
            'value', 'checked', 'selected_text', 'selected_options', 'visible_options',
            'value_min', 'value_max',
            'aria_value_now', 'aria_value_min', 'aria_value_max', 'aria_value_text',
            'aria_busy',
            'background_tone', 'text_tone', 'tone', 'color',
            'variant',
            'indeterminate', 'controls', 'paused', 'muted', 'current_time', 'duration'
        ]
            .some(key => item[key] !== undefined && String(item[key]) !== '');
    }

    function visibleClickable(el) {
        if (interactionBlocked(el)) return false;
        if (blockedByActiveModal(el)) return false;
        if (tagNameOf(el) === 'area') return !!areaRectOf(el);
        const rects = targetVisibleRects(el);
        if (!rects.length) return false;
        const style = window.getComputedStyle(el);
        if (style.pointerEvents === 'none') return false;
        return rects.some(rect => clickHitVisible(el, rect));
    }

    function domainLike(name) {
        return /^[a-z0-9.-]+\\.[a-z]{2,}/i.test(name || '');
    }

    function metaLike(name) {
        const s = String(name || '').toLowerCase();
        return ['hide', 'vote', 'login', 'new', 'past', 'ask', 'show', 'jobs', 'submit', 'comments'].includes(s)
            || /^\\d+\\s+(comments?|points?)$/.test(s)
            || /^\\d+\\s+(minutes?|hours?|days?) ago$/.test(s);
    }

    function groupOf(item) {
        if ([
            'textbox', 'button', 'select', 'combobox', 'checkbox', 'radio', 'switch',
            'summary', 'tab', 'option', 'treeitem', 'menuitem', 'menuitemcheckbox',
            'menuitemradio', 'slider', 'spinbutton', 'color', 'file', 'focusable'
        ].includes(item.role)) return 'control';
        if (item.role === 'link' && item.source === 'graphic') return 'control';
        if (
            item.role === 'link' &&
            item.name.length >= 8 &&
            item.rect.width >= 70 &&
            item.rect.height >= 14 &&
            !domainLike(item.name) &&
            !metaLike(item.name)
        ) return 'primary';
        return 'minor';
    }

    const representedContextOwners = new WeakSet();
    const seen = new Set();
    const candidateElements = [];
    const candidates = [];
    for (const el of queryAllDeep(clickableSelector)) {
        const target = closestComposed(el, targetSelector) || el;
        if (seen.has(target)) continue;
        seen.add(target);
        if (!visibleClickable(target)) continue;
        const named = targetName(target);
        const role = roleOf(target);
        const state = controlState(target);
        const placeholder = visiblePlaceholder(target);
        if (placeholder && named.source !== 'placeholder') state.placeholder = placeholder;
        if (noInfoFocusable(target, role, named, state)) continue;
        const rect = rectOf(target);
        const item = {
            dom_index: candidates.length,
            tag: target.tagName.toLowerCase(),
            role,
            name: named.name,
            source: named.source,
            rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
            center: rect.center,
            href: hrefOf(target),
            input_type: target.tagName === 'INPUT' ? (target.getAttribute('type') || 'text').toLowerCase() : '',
            disabled: interactionBlocked(target)
        };
        Object.assign(item, state);
        const context = semanticContext(target);
        if (context) {
            if (context.role) item.context_role = context.role;
            if (context.name && context.name !== item.name) item.context = context.name;
            if (context.owner) representedContextOwners.add(context.owner);
        }
        item.group = groupOf(item);
        candidates.push(item);
        candidateElements.push(target);
    }

    const selected = candidates;
    const selectedElements = candidateElements;
    const paired = selected.map((item, idx) => ({ item, el: selectedElements[idx] }));
    paired.sort((a, b) => (a.item.rect.y - b.item.rect.y) || (a.item.rect.x - b.item.rect.x) || (a.item.dom_index - b.item.dom_index));
    const targetIndexByElement = new Map();
    paired.forEach((pair, index) => {
        pair.item.index = index;
        targetIndexByElement.set(pair.el, index);
    });
    const selectedTargets = paired.map(pair => pair.item);
    const selectedTargetElements = paired.map(pair => pair.el);
    const describedTargetsByElement = new Map();

    function registerDescribedTargets(target, index) {
        const ids = compact(target.getAttribute('aria-describedby')).split(/\\s+/).filter(Boolean);
        for (const id of ids) {
            const ref = elementByIdInScope(target, id);
            if (!ref || hasSkippedAncestor(ref) || !elementVisible(ref)) continue;
            const values = describedTargetsByElement.get(ref) || [];
            if (!values.includes(index)) values.push(index);
            describedTargetsByElement.set(ref, values);
        }
    }

    paired.forEach((pair, index) => registerDescribedTargets(pair.el, index));

    function describedTargetIndexes(el) {
        for (let cur = el; cur && cur !== document.body && cur !== document.documentElement; cur = composedParent(cur)) {
            const values = describedTargetsByElement.get(cur);
            if (values && values.length) return values.join('|');
        }
        return '';
    }

    function ownerKind(el) {
        const tag = (el.tagName || '').toLowerCase();
        const role = compact(el.getAttribute('role')).toLowerCase();
        if (['alert', 'status', 'note', 'log', 'timer', 'dialog', 'alertdialog'].includes(role)) return role;
        if (role === 'tooltip') return 'tooltip';
        if (role === 'heading') return 'heading';
        if (role === 'paragraph') return 'paragraph';
        if (role === 'listitem') return 'list_item';
        if (role === 'cell') return 'table_cell';
        if (role === 'text') return 'text';
        if (/^h[1-6]$/.test(tag)) return 'heading';
        if (tag === 'p') return 'paragraph';
        if (tag === 'li') return 'list_item';
        if (tag === 'td' || tag === 'th') return 'table_cell';
        if (tag === 'blockquote') return 'quote';
        if (tag === 'pre') return 'pre';
        if (tag === 'time') return 'time';
        if (tag === 'label' || tag === 'legend') return 'label';
        if (tag === 'button') return 'button';
        if (tag === 'article' || tag === 'section') return 'section';
        if (tag === 'div') return 'text';
        if (role) return role;
        return 'text';
    }

    function directTargetFor(el) {
        let cur = el;
        while (cur && cur !== document.body && cur !== document.documentElement) {
            if (targetIndexByElement.has(cur)) return cur;
            cur = composedParent(cur);
        }
        return null;
    }

    function labelControlTarget(label) {
        if (!label || tagNameOf(label) !== 'label') return null;
        const control = label.control || null;
        return control ? directTargetFor(control) : null;
    }

    function labelledTargetOwner(el) {
        const label = closestComposed(el, 'label');
        return labelControlTarget(label) ? label : null;
    }

    function appendText(parts, text) {
        const value = String(text || '').replace(/\\s+/g, ' ');
        if (!value.trim()) return;
        const last = parts[parts.length - 1];
        if (last && last.type === 'text') last.text = String(last.text || '') + value;
        else parts.push({ type: 'text', text: value });
    }

    function textFragmentNeedsSpace(previous, next) {
        const left = String(previous || '');
        const right = String(next || '');
        if (!left || !right) return false;
        if (/\\s$/.test(left) || /^\\s/.test(right)) return false;
        if (/^[,.;:!?%)}\\]'"\u2019\u201d]/.test(right)) return false;
        if (/[(\\[{'"\\u2018\\u201c]$/.test(left)) return false;
        return true;
    }

    function joinTextFragments(chunks) {
        let text = '';
        for (const chunk of chunks) {
            const value = String(chunk || '').replace(/\\s+/g, ' ');
            if (!value.trim()) continue;
            if (textFragmentNeedsSpace(text, value)) text += ' ';
            text += value;
        }
        return text;
    }

    function appendSeparatedText(parts, text) {
        const value = String(text || '').replace(/\\s+/g, ' ');
        if (!value.trim()) return;
        const last = parts[parts.length - 1];
        const lastText = last && last.type === 'text' ? String(last.text || '') : '';
        if (
            last &&
            last.type === 'text' &&
            textFragmentNeedsSpace(lastText, value)
        ) {
            last.text = String(last.text || '') + ' ';
        }
        appendText(parts, value);
    }

    function trimEdgeTextParts(parts) {
        while (parts.length && parts[0].type === 'text') {
            parts[0].text = String(parts[0].text || '').replace(/^\\s+/, '');
            if (parts[0].text) break;
            parts.shift();
        }
        while (parts.length && parts[parts.length - 1].type === 'text') {
            const last = parts[parts.length - 1];
            last.text = String(last.text || '').replace(/\\s+$/, '');
            if (last.text) break;
            parts.pop();
        }
        return parts;
    }

    function appendRef(parts, target) {
        const index = targetIndexByElement.get(target);
        if (index === undefined) return false;
        const last = parts[parts.length - 1];
        if (last && last.type === 'ref' && last.target === index) return true;
        parts.push({ type: 'ref', target: index });
        return true;
    }

    function appendPseudo(parts, el, pseudo, coveredPseudoElements) {
        const text = pseudoTextFor(el, pseudo);
        if (!text) return;
        if (pseudo === '::before') {
            appendSeparatedText(parts, /[(\\[{'"\\u2018\\u201c]$/.test(text) ? text : `${text} `);
        } else {
            appendSeparatedText(parts, /^[,.;:!?%)}\\]'"\u2019\u201d]/.test(text) ? text : ` ${text}`);
        }
        coveredPseudoElements.push(el);
    }

    function alphaOrdinal(value, upper) {
        let n = Math.max(1, Number(value) || 1);
        let out = '';
        while (n > 0) {
            n -= 1;
            out = String.fromCharCode((upper ? 65 : 97) + (n % 26)) + out;
            n = Math.floor(n / 26);
        }
        return out;
    }

    function romanOrdinal(value, upper) {
        let n = Math.max(1, Math.min(3999, Number(value) || 1));
        const pairs = [
            [1000, 'm'], [900, 'cm'], [500, 'd'], [400, 'cd'],
            [100, 'c'], [90, 'xc'], [50, 'l'], [40, 'xl'],
            [10, 'x'], [9, 'ix'], [5, 'v'], [4, 'iv'], [1, 'i']
        ];
        let out = '';
        for (const [amount, token] of pairs) {
            while (n >= amount) {
                out += token;
                n -= amount;
            }
        }
        return upper ? out.toUpperCase() : out;
    }

    function numericAttr(el, attr) {
        const raw = compact(el && el.getAttribute ? el.getAttribute(attr) : '');
        if (!raw) return NaN;
        const value = Number(raw);
        return Number.isFinite(value) ? value : NaN;
    }

    function orderedListValue(li) {
        const parent = composedParent(li);
        if (!parent || tagNameOf(parent) !== 'ol') return null;
        const items = Array.from(parent.children || []).filter(child => tagNameOf(child) === 'li');
        const explicit = numericAttr(li, 'value');
        if (Number.isFinite(explicit)) return explicit;
        const index = Math.max(0, items.indexOf(li));
        const reversed = parent.hasAttribute('reversed');
        if (reversed) {
            const start = numericAttr(parent, 'start');
            const base = Number.isFinite(start) ? start : items.length;
            return base - index;
        }
        const start = numericAttr(parent, 'start');
        const base = Number.isFinite(start) ? start : 1;
        return base + index;
    }

    function listMarkerText(el) {
        if (tagNameOf(el) !== 'li') return '';
        try {
            const customMarker = cssStringContent(window.getComputedStyle(el, '::marker').content, el, '::marker');
            if (customMarker) return customMarker;
        } catch (_) {}
        const value = orderedListValue(el);
        if (value === null) return '';
        const parent = composedParent(el);
        const style = window.getComputedStyle(el);
        const markerTypeRaw = compact(el.getAttribute('type') || parent?.getAttribute('type') || style.listStyleType || 'decimal');
        const markerType = markerTypeRaw.toLowerCase();
        if (markerType === 'none') return '';
        if (markerType === 'lower-alpha' || markerTypeRaw === 'a') return `${alphaOrdinal(value, false)}.`;
        if (markerType === 'upper-alpha' || markerTypeRaw === 'A') return `${alphaOrdinal(value, true)}.`;
        if (markerType === 'lower-roman' || markerType === 'i') return `${romanOrdinal(value, false)}.`;
        if (markerType === 'upper-roman' || markerTypeRaw === 'I') return `${romanOrdinal(value, true)}.`;
        if (markerType === 'decimal-leading-zero') return `${String(value).padStart(2, '0')}.`;
        return `${value}.`;
    }

    function textUnits(text) {
        const units = [];
        const re = /\\s*\\S+\\s*/gu;
        let match;
        while ((match = re.exec(text)) !== null) {
            const raw = match[0];
            const visibleChars = Array.from(raw.trimEnd());
            if (visibleChars.length > 32) {
                let offset = match.index;
                for (const char of visibleChars) {
                    units.push({ start: offset, end: offset + char.length, text: char });
                    offset += char.length;
                }
                const trailing = raw.slice(offset - match.index);
                if (/^\\s+$/.test(trailing) && units.length) {
                    units[units.length - 1].text += trailing;
                }
            } else {
                units.push({ start: match.index, end: match.index + raw.length, text: raw });
            }
        }
        return units;
    }

    function visibleTextFromTextNode(node) {
        const parent = node.parentElement;
        if (!styleVisible(parent)) return '';
        const text = node.nodeValue || '';
        if (!text.trim()) return '';
        const range = document.createRange();
        const chunks = [];
        for (const unit of textUnits(text)) {
            range.setStart(node, unit.start);
            range.setEnd(node, unit.end);
            if (Array.from(range.getClientRects()).some(rect => readableTextRect(rect, parent))) {
                chunks.push(unit.text);
            }
        }
        range.detach && range.detach();
        return applyTextTransform(chunks.join(''), parent);
    }

    function applyTextTransform(text, el, pseudo = null) {
        const value = String(text || '');
        if (!value || !el) return value;
        let transform = '';
        try {
            transform = String(window.getComputedStyle(el, pseudo).textTransform || '').toLowerCase();
        } catch (_) {
            return value;
        }
        if (!transform || transform === 'none') return value;
        if (transform.includes('uppercase')) return value.toLocaleUpperCase();
        if (transform.includes('lowercase')) return value.toLocaleLowerCase();
        if (transform.includes('capitalize')) {
            return value.replace(/(^|[^\\p{L}\\p{N}])(\\p{L})/gu, (_match, prefix, letter) =>
                `${prefix}${String(letter).toLocaleUpperCase()}`
            );
        }
        return value;
    }

    function visibleTextWithin(root, includeRootPseudo = false) {
        const chunks = [];
        if (includeRootPseudo) {
            const ownBefore = pseudoTextFor(root, '::before');
            if (ownBefore) chunks.push(ownBefore);
        }
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
            acceptNode(node) {
                const parent = node.parentElement;
                if (!parent || hasSkippedAncestor(parent) || !styleVisible(parent)) return NodeFilter.FILTER_REJECT;
                if (nearestSemanticGraphic(parent)) return NodeFilter.FILTER_REJECT;
                if (!(node.nodeValue || '').trim()) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        });
        while (walker.nextNode()) {
            const text = visibleTextFromTextNode(walker.currentNode);
            if (text) chunks.push(text);
        }
        if (root.querySelectorAll) {
            for (const el of Array.from(root.querySelectorAll('*'))) {
                const before = pseudoTextFor(el, '::before');
                const after = pseudoTextFor(el, '::after');
                if (before) chunks.push(before);
                if (after) chunks.push(after);
            }
        }
        if (includeRootPseudo) {
            const ownAfter = pseudoTextFor(root, '::after');
            if (ownAfter) chunks.push(ownAfter);
        }
        return joinTextFragments(chunks);
    }

    function buildParts(root, parts, coveredTextNodes, coveredPseudoElements) {
        const marker = listMarkerText(root);
        if (marker) appendText(parts, `${marker} `);
        appendPseudo(parts, root, '::before', coveredPseudoElements);
        for (const node of Array.from(root.childNodes || [])) {
            if (node.nodeType === Node.TEXT_NODE) {
                const text = visibleTextFromTextNode(node);
                if (text) {
                    appendText(parts, text);
                    coveredTextNodes.push(node);
                }
                continue;
            }
            if (node.nodeType !== Node.ELEMENT_NODE) continue;
            const el = node;
            if (skipTags.has(tagNameOf(el)) || !styleVisible(el)) continue;
            if (nearestSemanticGraphic(el)) continue;
            const target = directTargetFor(el);
            if (target) {
                appendRef(parts, target);
                continue;
            }
            buildParts(el, parts, coveredTextNodes, coveredPseudoElements);
        }
        appendPseudo(parts, root, '::after', coveredPseudoElements);
    }

    function collectTextNodesDeep(root, out = []) {
        if (!root) return out;
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
            acceptNode(node) {
                const parent = node.parentElement;
                if (!parent || hasSkippedAncestor(parent) || !styleVisible(parent)) return NodeFilter.FILTER_REJECT;
                if (nearestSemanticGraphic(parent)) return NodeFilter.FILTER_REJECT;
                if (!(node.nodeValue || '').trim()) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
            }
        });
        while (walker.nextNode()) out.push(walker.currentNode);
        if (root.querySelectorAll) {
            for (const el of Array.from(root.querySelectorAll('*'))) {
                if (el.shadowRoot) collectTextNodesDeep(el.shadowRoot, out);
            }
        }
        return out;
    }

    function hasVisibleChildTextOwner(owner) {
        return queryAllDeep(textOwnerSelector, owner).some(child => child !== owner && elementVisible(child));
    }

    function hasMeaningfulParts(parts) {
        const text = parts.filter(part => part.type === 'text').map(part => part.text).join(' ');
        if (text && !/^[|()\\[\\]·•»«.,:;，。、；：\\s]+$/.test(text)) return true;
        if (parts.some(part => part.type === 'ref') && parts.length > 1) return true;
        return parts.some(part => {
            if (part.type !== 'ref') return false;
            const target = selectedTargets[part.target];
            return target && target.source === 'visible' && target.group === 'primary';
        });
    }

    function scrollableAxis(style, axis) {
        const value = String(axis === 'x' ? style.overflowX : style.overflowY || style.overflow || '').toLowerCase();
        return /\\b(auto|scroll|overlay)\\b/.test(value);
    }

    function scrollRegionName(el) {
        const labelledBy = compact(el.getAttribute('aria-labelledby'));
        if (labelledBy) {
            const text = labelledBy.split(/\\s+/).map(id => document.getElementById(id)?.innerText || '').join(' ');
            if (compact(text)) return compact(text);
        }
        return compact(el.getAttribute('aria-label') || el.getAttribute('title') || '');
    }

    function collectScrollRegions() {
        const regions = [];
        const seen = new Set();
        for (const el of queryAllDeep('*')) {
            const tag = tagNameOf(el);
            if (tag === 'html' || tag === 'body' || skipTags.has(tag) || seen.has(el)) continue;
            seen.add(el);
            if (!elementVisible(el)) continue;
            const style = window.getComputedStyle(el);
            const canScrollY = scrollableAxis(style, 'y') && el.scrollHeight > el.clientHeight + 2;
            const canScrollX = scrollableAxis(style, 'x') && el.scrollWidth > el.clientWidth + 2;
            if (!canScrollY && !canScrollX) continue;
            const rect = rectOf(el);
            if (!rect.width || !rect.height || rect.width < 80 || rect.height < 40 || rect.width * rect.height < 3200) continue;
            regions.push({
                index: regions.length,
                tag,
                role: roleOf(el),
                name: scrollRegionName(el),
                rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                center: rect.center,
                scroll_x: Math.round(el.scrollLeft || 0),
                scroll_y: Math.round(el.scrollTop || 0),
                viewport_width: Math.round(el.clientWidth || 0),
                viewport_height: Math.round(el.clientHeight || 0),
                scroll_width: Math.round(el.scrollWidth || 0),
                scroll_height: Math.round(el.scrollHeight || 0),
                can_scroll_up: canScrollY && (el.scrollTop || 0) > 0,
                can_scroll_down: canScrollY && (el.scrollTop || 0) + (el.clientHeight || 0) + 2 < (el.scrollHeight || 0),
                can_scroll_left: canScrollX && (el.scrollLeft || 0) > 0,
                can_scroll_right: canScrollX && (el.scrollLeft || 0) + (el.clientWidth || 0) + 2 < (el.scrollWidth || 0)
            });
        }
        regions.sort((a, b) => (a.rect.y - b.rect.y) || (a.rect.x - b.rect.x));
        regions.forEach((region, index) => { region.index = index; });
        return regions;
    }

    const blocks = [];
    const coveredTextNodes = new WeakSet();
    const coveredPseudoElements = new WeakSet();
    const representedTableCells = new WeakSet();
    const representedTableCaptions = new WeakSet();
    const representedIndicators = new WeakSet();

    function nearestRepresentedTableCell(el) {
        const cell = closestComposed(el, `td,th,${ariaCellSelector}`);
        return cell && representedTableCells.has(cell) ? cell : null;
    }

    function nearestRepresentedTableCaption(el) {
        const caption = closestComposed(el, 'caption');
        return caption && representedTableCaptions.has(caption) ? caption : null;
    }

    function representedTableOwner(el) {
        return nearestRepresentedTableCell(el) || nearestRepresentedTableCaption(el);
    }

    function nearestRepresentedIndicator(el) {
        const indicator = closestComposed(el, indicatorSelector);
        return indicator && representedIndicators.has(indicator) ? indicator : null;
    }

    function labelIndicatorTarget(label) {
        if (!label || tagNameOf(label) !== 'label') return null;
        const control = label.control || null;
        return control && representedIndicators.has(control) ? control : null;
    }

    function labelledIndicatorOwner(el) {
        const label = closestComposed(el, 'label');
        return labelIndicatorTarget(label) ? label : null;
    }

    function representedContextLabelOwner(el) {
        if (!el || tagNameOf(el) !== 'legend') return null;
        const owner = composedParent(el);
        return owner && representedContextOwners.has(owner) ? owner : null;
    }

    const structuredTableSelector = 'table,[role="table"],[role="grid"],[role="treegrid"]';
    const ariaCellSelector = '[role="cell"],[role="gridcell"],[role="columnheader"],[role="rowheader"]';

    function explicitRole(el) {
        return compact(el && el.getAttribute ? el.getAttribute('role') : '').toLowerCase();
    }

    function numericAttrValue(el, attr) {
        const raw = compact(el && el.getAttribute ? el.getAttribute(attr) : '');
        if (!raw) return null;
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
    }

    function tableName(table) {
        const caption = table.querySelector && table.querySelector('caption');
        const captionText = caption ? visibleTextWithin(caption, true) : '';
        return compact(captionText || referencedText(table, 'aria-labelledby') || table.getAttribute('aria-label') || table.getAttribute('title') || '');
    }

    function rowOwner(row) {
        return closestComposed(row, structuredTableSelector);
    }

    function cellOwnerRow(cell) {
        return closestComposed(cell, 'tr,[role="row"]');
    }

    function rowsForStructuredTable(table) {
        if (tagNameOf(table) === 'table') return Array.from(table.rows || []);
        return queryAllDeep('[role="row"]', table).filter(row => rowOwner(row) === table);
    }

    function cellsForStructuredRow(row) {
        if (row.cells) return Array.from(row.cells || []);
        return queryAllDeep(ariaCellSelector, row).filter(cell => cellOwnerRow(cell) === row);
    }

    function tableHasVisibleChildTable(table) {
        return Array.from(table.querySelectorAll(structuredTableSelector)).some(child =>
            child !== table && rowOwner(child) !== table && elementVisible(child)
        );
    }

    function cellHeader(cell) {
        const role = explicitRole(cell);
        return tagNameOf(cell) === 'th' || role === 'columnheader' || role === 'rowheader';
    }

    function collectTables() {
        const tables = [];
        const seen = new Set();
        for (const table of queryAllDeep(structuredTableSelector)) {
            if (seen.has(table) || !elementVisible(table)) continue;
            seen.add(table);
            if (tableHasVisibleChildTable(table)) continue;
            const tableRect = rectOf(table);
            if (!tableRect || !tableRect.width || !tableRect.height) continue;
            const rows = [];
            for (const row of rowsForStructuredTable(table)) {
                const cells = [];
                let rowRect = rectOf(row);
                if (!rowRect || !rowRect.width || !rowRect.height) rowRect = null;
                for (const cell of cellsForStructuredRow(row)) {
                    if (!elementVisible(cell)) continue;
                    const cellRect = rectOf(cell);
                    if (!cellRect || !cellRect.width || !cellRect.height) continue;
                    const parts = [];
                    const pendingCoveredTextNodes = [];
                    const pendingCoveredPseudoElements = [];
                    buildParts(cell, parts, pendingCoveredTextNodes, pendingCoveredPseudoElements);
                    trimEdgeTextParts(parts);
                    const ariaRowIndex = numericAttrValue(cell, 'aria-rowindex');
                    const ariaColIndex = numericAttrValue(cell, 'aria-colindex');
                    const ariaSort = compact(cell.getAttribute('aria-sort')).toLowerCase();
                    const ariaBusy = compact(cell.getAttribute('aria-busy')).toLowerCase();
                    const role = explicitRole(cell);
                    const visualState = {
                        ...textVisualState(cell),
                        ...visualToneState(cell)
                    };
                    if (!hasMeaningfulParts(parts) && !Object.keys(visualState).length) continue;
                    representedTableCells.add(cell);
                    pendingCoveredTextNodes.forEach(node => coveredTextNodes.add(node));
                    pendingCoveredPseudoElements.forEach(el => coveredPseudoElements.add(el));
                    const cellItem = {
                        row: rows.length,
                        col: Number.isFinite(cell.cellIndex) && cell.cellIndex >= 0 ? cell.cellIndex : cells.length,
                        tag: tagNameOf(cell),
                        role,
                        header: cellHeader(cell),
                        rowspan: cell.rowSpan || 1,
                        colspan: cell.colSpan || 1,
                        rect: { x: cellRect.x, y: cellRect.y, width: cellRect.width, height: cellRect.height },
                        parts
                    };
                    if (ariaRowIndex !== null) cellItem.row_index = ariaRowIndex;
                    if (ariaColIndex !== null) cellItem.col_index = ariaColIndex;
                    if (ariaSort) cellItem.aria_sort = ariaSort;
                    if (ariaBusy) cellItem.aria_busy = ariaBusy;
                    Object.assign(cellItem, visualState);
                    cells.push(cellItem);
                    rowRect = mergeRects(rowRect, cellRect);
                }
                if (!cells.length) continue;
                if (!rowRect || !rowRect.width || !rowRect.height) continue;
                const rowItem = {
                    index: rows.length,
                    rect: { x: rowRect.x, y: rowRect.y, width: rowRect.width, height: rowRect.height },
                    cells
                };
                const ariaRowIndex = numericAttrValue(row, 'aria-rowindex');
                if (ariaRowIndex !== null) rowItem.row_index = ariaRowIndex;
                rows.push(rowItem);
            }
            if (!rows.length) continue;
            const caption = table.querySelector && table.querySelector('caption');
            if (caption && elementVisible(caption)) representedTableCaptions.add(caption);
            tables.push({
                index: tables.length,
                tag: tagNameOf(table),
                role: roleOf(table),
                name: tableName(table),
                rect: { x: tableRect.x, y: tableRect.y, width: tableRect.width, height: tableRect.height },
                rows
            });
        }
        tables.sort((a, b) => (a.rect.y - b.rect.y) || (a.rect.x - b.rect.x));
        tables.forEach((table, index) => { table.index = index; });
        return tables;
    }

    function collectIndicators() {
        const indicators = [];
        const seen = new Set();
        for (const el of queryAllDeep(indicatorSelector)) {
            const kind = indicatorKind(el);
            if (seen.has(el) || !(kind === 'graphic' ? graphicVisible(el) : elementVisible(el))) continue;
            seen.add(el);
            let disabledName = null;
            let loadingInfo = null;
            if (kind === 'graphic') {
                if (!semanticGraphicName(el)) continue;
                const clickableAncestor = closestComposed(el, clickableSelector);
                if (clickableAncestor && targetIndexByElement.has(clickableAncestor)) continue;
            } else if (kind === 'loading_placeholder') {
                loadingInfo = loadingPlaceholderInfo(el);
                if (!loadingInfo) continue;
                if (targetIndexByElement.has(el) || closestComposed(el, clickableSelector)) continue;
                if (representedTableOwner(el)) continue;
            } else if (kind === 'busy') {
                const clickableAncestor = closestComposed(el, clickableSelector);
                if (targetIndexByElement.has(el) || (clickableAncestor && targetIndexByElement.has(clickableAncestor))) continue;
                if (representedTableOwner(el)) continue;
                if (busyTextOwnerElement(el)) continue;
            } else if (kind === 'disabled_control' || kind === 'inert_control') {
                disabledName = targetName(el);
                if (disabledName.source === 'fallback') {
                    disabledName = { name: '', source: '' };
                }
            }
            const rect = rectOf(el);
            if (!rect || !rect.width || !rect.height) continue;
            const state = indicatorState(el);
            const placeholder = visiblePlaceholder(el);
            if (placeholder && (!disabledName || disabledName.source !== 'placeholder')) {
                state.placeholder = placeholder;
            }
            const text = kind === 'graphic' || kind === 'busy' || kind === 'loading_placeholder' || kind === 'disabled_control' || kind === 'inert_control'
                ? ''
                : compact(visibleTextWithin(el, true));
            const item = {
                index: indicators.length,
                kind,
                tag: tagNameOf(el),
                role: kind === 'disabled_control' || kind === 'inert_control' ? roleOf(el) : '',
                name: loadingInfo ? loadingInfo.name : (disabledName ? disabledName.name : indicatorName(el)),
                source: loadingInfo ? loadingInfo.source : (disabledName ? disabledName.source : ''),
                text,
                rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                ...(loadingInfo ? { variant: loadingInfo.variant } : {}),
                ...state
            };
            if (!meaningfulIndicator(item)) continue;
            if (kind !== 'busy') representedIndicators.add(el);
            indicators.push(item);
        }
        for (const el of queryAllDeep('*')) {
            if (seen.has(el)) continue;
            const swatch = colorSwatchInfo(el);
            if (!swatch) continue;
            const rect = rectOf(el);
            if (!rect || !rect.width || !rect.height) continue;
            const item = {
                index: indicators.length,
                kind: 'color_swatch',
                tag: tagNameOf(el),
                role: '',
                name: swatch.name,
                source: swatch.source,
                rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                tone: swatch.tone,
                color: swatch.color
            };
            if (!meaningfulIndicator(item)) continue;
            seen.add(el);
            representedIndicators.add(el);
            indicators.push(item);
        }
        indicators.sort((a, b) => (a.rect.y - b.rect.y) || (a.rect.x - b.rect.x));
        indicators.forEach((item, index) => { item.index = index; });
        return indicators;
    }

    function collectSvgTextBlocks() {
        const svgBlocks = [];
        const seen = new Set();
        for (const el of queryAllDeep('svg text')) {
            if (seen.has(el)) continue;
            seen.add(el);
            if (closestComposed(el, '[aria-hidden="true"],[role="presentation"],[role="none"]')) continue;
            if (!svgTextStyleVisible(el)) continue;
            const text = meaningfulGraphicText(el.textContent || '');
            if (!text) continue;
            const rect = svgTextRect(el);
            if (!rect || !rect.width || !rect.height) continue;
            const parts = [];
            const target = directTargetFor(el);
            if (target) {
                appendRef(parts, target);
            } else {
                appendSeparatedText(parts, text);
            }
            trimEdgeTextParts(parts);
            if (!hasMeaningfulParts(parts)) continue;
            svgBlocks.push({
                index: svgBlocks.length,
                kind: 'svg_text',
                tag: 'text',
                rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                parts
            });
        }
        return svgBlocks;
    }

    const tables = collectTables();
    const indicators = collectIndicators();
    for (const owner of queryAllDeep(textOwnerSelector)) {
        if (!elementVisible(owner)) continue;
        if (nearestSemanticGraphic(owner)) continue;
        if (targetIndexByElement.has(owner)) continue;
        if (representedTableOwner(owner)) continue;
        if (nearestRepresentedIndicator(owner)) continue;
        if (labelControlTarget(owner)) continue;
        if (labelIndicatorTarget(owner)) continue;
        if (representedContextLabelOwner(owner)) continue;
        if (hasVisibleChildTextOwner(owner)) continue;
        const directClickableAncestor = closestComposed(owner, clickableSelector);
        if (directClickableAncestor && directClickableAncestor !== owner && targetIndexByElement.has(directClickableAncestor)) continue;
        const parts = [];
        const pendingCoveredTextNodes = [];
        const pendingCoveredPseudoElements = [];
        buildParts(owner, parts, pendingCoveredTextNodes, pendingCoveredPseudoElements);
        trimEdgeTextParts(parts);
        if (!hasMeaningfulParts(parts)) continue;
        const rect = rectOf(owner);
        if (!rect.width || !rect.height) continue;
        pendingCoveredTextNodes.forEach(node => coveredTextNodes.add(node));
        pendingCoveredPseudoElements.forEach(el => coveredPseudoElements.add(el));
        blocks.push({
            index: blocks.length,
            kind: ownerKind(owner),
            tag: owner.tagName.toLowerCase(),
            rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
            ...textState(owner),
            parts
        });
    }

    const orphanGroups = new Map();
    for (const node of collectTextNodesDeep(document.body || document.documentElement)) {
        if (coveredTextNodes.has(node)) continue;
        const parent = node.parentElement;
        if (!parent || hasSkippedAncestor(parent) || !styleVisible(parent)) continue;
        if (directTargetFor(parent)) continue;
        if (representedTableOwner(parent)) continue;
        if (nearestRepresentedIndicator(parent)) continue;
        if (labelledTargetOwner(parent)) continue;
        if (labelledIndicatorOwner(parent)) continue;
        if (representedContextLabelOwner(parent)) continue;
        const clickableAncestor = closestComposed(parent, clickableSelector);
        if (clickableAncestor && targetIndexByElement.has(clickableAncestor)) continue;
        const text = visibleTextFromTextNode(node);
        if (!text) continue;
        const rect = rectOfTextNode(node);
        if (!rect || !rect.width || !rect.height) continue;
        let group = orphanGroups.get(parent);
        if (!group) {
            group = { owner: parent, rect: null, parts: [] };
            orphanGroups.set(parent, group);
        }
        appendSeparatedText(group.parts, text);
        group.rect = mergeRects(group.rect, rect);
    }
    for (const group of orphanGroups.values()) {
        trimEdgeTextParts(group.parts);
        if (!group.rect || !hasMeaningfulParts(group.parts)) continue;
        blocks.push({
            index: blocks.length,
            kind: ownerKind(group.owner),
            tag: group.owner.tagName.toLowerCase(),
            rect: group.rect,
            ...textState(group.owner),
            parts: group.parts
        });
    }
    for (const el of queryAllDeep('*')) {
        if (coveredPseudoElements.has(el)) continue;
        if (hasSkippedAncestor(el) || !styleVisible(el)) continue;
        if (nearestSemanticGraphic(el)) continue;
        if (directTargetFor(el)) continue;
        if (representedTableOwner(el)) continue;
        if (nearestRepresentedIndicator(el)) continue;
        if (labelledTargetOwner(el)) continue;
        if (labelledIndicatorOwner(el)) continue;
        if (representedContextLabelOwner(el)) continue;
        const clickableAncestor = closestComposed(el, clickableSelector);
        if (clickableAncestor && targetIndexByElement.has(clickableAncestor)) continue;
        const text = pseudoText(el);
        if (!text) continue;
        const rect = rectOf(el);
        if (!rect.width || !rect.height) continue;
        const parts = [];
        appendSeparatedText(parts, text);
        trimEdgeTextParts(parts);
        if (!hasMeaningfulParts(parts)) continue;
        blocks.push({
            index: blocks.length,
            kind: ownerKind(el),
            tag: tagNameOf(el),
            rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
            ...textState(el),
            parts
        });
    }
    for (const block of collectSvgTextBlocks()) {
        block.index = blocks.length;
        blocks.push(block);
    }

    function addReferencedTargetsFromParts(parts, out) {
        for (const part of parts || []) {
            if (part && part.type === 'ref' && Number.isFinite(Number(part.target))) {
                out.add(Number(part.target));
            }
        }
    }

    function referencedTargetIndexes() {
        const refs = new Set();
        for (const block of blocks) addReferencedTargetsFromParts(block.parts, refs);
        for (const table of tables) {
            for (const row of table.rows || []) {
                for (const cell of row.cells || []) {
                    addReferencedTargetsFromParts(cell.parts, refs);
                }
            }
        }
        return refs;
    }

    function addUnrepresentedVisibleLinkTextBlocks() {
        const refs = referencedTargetIndexes();
        selectedTargets.forEach((target, index) => {
            if (refs.has(index)) return;
            if (!target || target.role !== 'link' || target.source !== 'visible') return;
            if (!compact(target.name)) return;
            const rect = target.rect || {};
            if (!rect.width || !rect.height) return;
            const el = selectedTargetElements[index];
            if (!el || hasSkippedAncestor(el) || !styleVisible(el)) return;
            blocks.push({
                index: blocks.length,
                kind: ownerKind(el),
                tag: tagNameOf(el),
                rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                ...textState(el),
                parts: [{ type: 'ref', target: index }]
            });
            refs.add(index);
        });
    }

    addUnrepresentedVisibleLinkTextBlocks();
    blocks.sort((a, b) => (a.rect.y - b.rect.y) || (a.rect.x - b.rect.x));
    blocks.forEach((block, index) => { block.index = index; });

    const doc = document.documentElement;
    const body = document.body;
    const scrollY = Math.max(window.scrollY || 0, doc.scrollTop || 0, body ? body.scrollTop || 0 : 0);
    const viewportHeight = window.innerHeight || doc.clientHeight || 0;
    const pageHeight = Math.max(doc.scrollHeight || 0, body ? body.scrollHeight || 0 : 0);
    return {
        viewport: { width: window.innerWidth || 0, height: window.innerHeight || 0 },
        scroll: {
            y: Math.round(scrollY),
            viewport_height: Math.round(viewportHeight),
            page_height: Math.round(pageHeight),
            can_scroll_up: scrollY > 0,
            can_scroll_down: scrollY + viewportHeight + 2 < pageHeight
        },
        click_targets: selectedTargets,
        target_candidates: candidates.length,
        tables,
        indicators,
        text_blocks: blocks,
        scroll_regions: collectScrollRegions()
    };
}"""

_CLICK_TARGETS_JS = """(limit) => {
    function textOf(el) {
        const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
        if (text) return text.slice(0, 120);
        const img = el.matches('img') ? el : el.querySelector('img');
        if (img) return (img.alt || img.getAttribute('aria-label') || '').slice(0, 120);
        return (el.getAttribute('aria-label') || el.title || '').slice(0, 120);
    }
    function meta(el, index) {
        const rect = el.getBoundingClientRect();
        const img = el.matches('img') ? el : el.querySelector('img');
        return {
            index,
            tag: el.tagName.toLowerCase(),
            text: textOf(el),
            href: el.href || '',
            src: img ? (img.currentSrc || img.src || '') : '',
            alt: img ? (img.alt || '') : '',
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            x: Math.round(rect.x),
            y: Math.round(rect.y)
        };
    }
    const selectors = ['a[href]', 'button', '[role="button"]', 'input[type="button"]', 'input[type="submit"]', 'img'];
    const seen = new Set();
    const targets = [];
    for (const el of document.querySelectorAll(selectors.join(','))) {
        const target = el.closest('a[href],button,[role="button"]') || el;
        if (seen.has(target)) continue;
        seen.add(target);
        const rect = target.getBoundingClientRect();
        const visible = rect.width > 20 && rect.height > 20
            && rect.bottom > 0 && rect.right > 0
            && rect.top < window.innerHeight && rect.left < window.innerWidth;
        if (!visible) continue;
        targets.push(target);
        if (targets.length >= limit) break;
    }
    return targets.map((el, index) => meta(el, index));
}"""

_CLICK_TARGET_JS = """({ index, limit }) => {
    const selectors = ['a[href]', 'button', '[role="button"]', 'input[type="button"]', 'input[type="submit"]', 'img'];
    const seen = new Set();
    const targets = [];
    for (const el of document.querySelectorAll(selectors.join(','))) {
        const target = el.closest('a[href],button,[role="button"]') || el;
        if (seen.has(target)) continue;
        seen.add(target);
        const rect = target.getBoundingClientRect();
        const visible = rect.width > 20 && rect.height > 20
            && rect.bottom > 0 && rect.right > 0
            && rect.top < window.innerHeight && rect.left < window.innerWidth;
        if (!visible) continue;
        targets.push(target);
        if (targets.length >= limit) break;
    }
    const target = targets[index];
    if (!target) return { ok: false, error: `No visible click target at index ${index}`, count: targets.length };
    const rect = target.getBoundingClientRect();
    const img = target.matches('img') ? target : target.querySelector('img');
    const info = {
        index,
        tag: target.tagName.toLowerCase(),
        text: (target.innerText || target.textContent || img?.alt || '').replace(/\\s+/g, ' ').trim().slice(0, 120),
        href: target.href || '',
        src: img ? (img.currentSrc || img.src || '') : '',
        alt: img ? (img.alt || '') : '',
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        x: Math.round(rect.x),
        y: Math.round(rect.y)
    };
    target.scrollIntoView({ block: 'center', inline: 'center' });
    target.click();
    return { ok: true, target: info, count: targets.length };
}"""
