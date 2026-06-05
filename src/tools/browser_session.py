"""Shared Playwright browser session for browser_control.

The module keeps a single persistent Chromium context so browser tools can
share cookies, history, scroll position, and response-cached images.
"""

from __future__ import annotations

import base64
import queue
import hashlib
import io
import logging
import mimetypes
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

logger = logging.getLogger("AICQ.tools.browser")
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


@dataclass
class BrowserImage:
    ref: str
    url: str
    path: str
    mime: str
    size_bytes: int
    sha256: str
    page_url: str


def _image_extension(mime: str, url: str) -> str:
    mime = mime.split(";", 1)[0].strip().lower()
    if mime in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[mime]
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".ico"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return mimetypes.guess_extension(mime) or ".bin"


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


class BrowserSession:
    def __init__(self) -> None:
        self.owner_thread_id = threading.get_ident()
        self._pw: Any | None = None
        self._browser: Any | None = None
        self.context: Any | None = None
        self.page: Any | None = None
        self.profile_dir = BROWSER_PROFILE_DIR / f"thread_{self.owner_thread_id}"
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

        ref = f"brimg_{digest[:12]}"
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
        except PlaywrightTimeoutError as exc:
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
        result = {
            "url": page.url,
            "title": page.title() or "",
            "scroll": self.scroll_state(),
            "click_targets": self.click_targets(),
            "visible_images": self.visible_images(),
            "cached_images": [asdict(item) for item in self.cached_images()],
            "cached_by_url": dict(self.cached_by_url),
            "response_errors": self.response_errors[-10:],
            "events": events or [],
        }
        if include_screenshot:
            png = _resize_png(page.screenshot(full_page=False, type="png"))
            result["_multimodal_parts"] = [
                {
                    "mime_type": "image/png",
                    "display_name": "browser_viewport.png",
                    "data": png,
                }
            ]
        return result

    def scroll_state(self) -> dict[str, Any]:
        return dict(self.require_page().evaluate(_SCROLL_STATE_JS) or {})

    def visible_images(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.require_page().evaluate(_VISIBLE_IMAGES_JS, limit) or [])

    def click_targets(self, limit: int = 30) -> list[dict[str, Any]]:
        return list(self.require_page().evaluate(_CLICK_TARGETS_JS, limit) or [])

    def click_target(self, index: int) -> dict[str, Any]:
        page = self.require_page()
        return dict(page.evaluate(_CLICK_TARGET_JS, {"index": int(index), "limit": 60}) or {})

    def cached_images(self) -> list[BrowserImage]:
        return sorted(self.cached_by_sha.values(), key=lambda item: item.size_bytes, reverse=True)

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


def make_image_data_url(image_ref: str) -> str | None:
    session = get_browser_session()
    item = session.read_image_file(image_ref)
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

_VISIBLE_IMAGES_JS = """(limit) => {
    const rows = [];
    for (const img of document.querySelectorAll('img')) {
        const rect = img.getBoundingClientRect();
        const visible = rect.width > 20 && rect.height > 20
            && rect.bottom > 0 && rect.right > 0
            && rect.top < window.innerHeight && rect.left < window.innerWidth;
        if (!visible) continue;
        rows.push({
            src: img.currentSrc || img.src || '',
            alt: img.alt || '',
            loaded: !!(img.complete && img.naturalWidth > 0),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            x: Math.round(rect.x),
            y: Math.round(rect.y)
        });
        if (rows.length >= limit) break;
    }
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
