"""Test browser viewport screenshots and image response caching.

This is a standalone probe for the proposed browser tool flow:

    open URL -> show viewport screenshot -> scroll/click/back/forward
    -> cache image response bytes -> expose send-ready image refs

Usage:
    python scripts/test_browser_image_pipeline.py --url https://example.com
    python scripts/test_browser_image_pipeline.py --url https://www.pixiv.net/tags/%E8%90%8C/artworks --channel msedge
    python scripts/test_browser_image_pipeline.py --url https://www.pixiv.net/tags/%E8%90%8C/artworks --channel msedge --headful
    python scripts/test_browser_image_pipeline.py --self-test --channel msedge --actions "click:0,back,forward"

Outputs:
    logs/browser_image_pipeline/
      manifest.json
      report.md
      screenshots/*.png
      images/*.{jpg,png,webp,gif}
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
}


@dataclass
class ScreenshotInfo:
    ref: str
    path: str
    scroll_y: int
    viewport_height: int
    page_height: int
    can_scroll_up: bool
    can_scroll_down: bool


@dataclass
class CachedImage:
    ref: str
    url: str
    path: str
    mime: str
    size_bytes: int
    sha256: str
    page_url: str


@dataclass
class ActionResult:
    index: int
    action: str
    ok: bool
    url: str
    title: str
    screenshot_ref: str
    detail: dict[str, Any]
    error: str = ""


def _image_extension(mime: str, url: str) -> str:
    mime = mime.split(";", 1)[0].strip().lower()
    if mime in MIME_EXTENSIONS:
        return MIME_EXTENSIONS[mime]
    path = urlparse(url).path
    guessed = Path(path).suffix.lower()
    if guessed in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}:
        return ".jpg" if guessed == ".jpeg" else guessed
    return mimetypes.guess_extension(mime) or ".bin"


def _make_self_test_page(out_dir: Path) -> str:
    fixture_dir = out_dir / "fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    image_path = fixture_dir / "sample.png"
    html_path = fixture_dir / "index.html"

    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (640, 420), (245, 247, 250))
        draw = ImageDraw.Draw(img)
        draw.rectangle((36, 36, 604, 384), outline=(24, 92, 160), width=8)
        draw.rectangle((80, 96, 280, 260), fill=(239, 92, 92))
        draw.ellipse((340, 96, 540, 260), fill=(62, 172, 118))
        draw.text((82, 310), "browser image cache self-test", fill=(20, 24, 33))
        img.save(image_path, format="PNG")
    except Exception:
        # 1x1 PNG fallback. It may be filtered by --min-image-bytes unless
        # self-test lowers the threshold.
        raw_png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
            "YGD4DwABBAEAgh+E8wAAAABJRU5ErkJggg=="
        )
        image_path.write_bytes(base64.b64decode(raw_png))

    html_path.write_text(
        """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Browser Image Pipeline Self Test</title>
<style>
body { margin: 0; font-family: Arial, sans-serif; background: #f4f4f1; color: #20242a; }
header { position: sticky; top: 0; padding: 18px 28px; background: #ffffff; border-bottom: 1px solid #ccc; }
section { min-height: 760px; padding: 28px; }
img { width: 420px; max-width: 80vw; border: 4px solid #20242a; display: block; margin: 24px 0; }
a { color: #005bac; font-weight: 700; }
.band-a { background: #f4f4f1; }
.band-b { background: #e9f1f6; }
</style>
<header>Browser Image Pipeline Self Test</header>
<section class="band-a">
  <h1>Viewport 0</h1>
  <p>The first screenshot should include this image.</p>
  <a href="#details"><img src="sample.png" alt="self-test image first view"></a>
</section>
<section class="band-b" id="details">
  <h1>Viewport 1</h1>
  <p>After scrolling, the second screenshot should include this repeated image.</p>
  <img src="sample.png?second=1" alt="self-test image second view">
</section>
</html>
""",
        encoding="utf-8",
    )
    return html_path.resolve().as_uri()


def _scroll_state(page) -> dict[str, int | bool]:
    state = page.evaluate(
        """() => {
            const doc = document.documentElement;
            const body = document.body;
            const scrollY = Math.max(window.scrollY || 0, doc.scrollTop || 0, body ? body.scrollTop || 0 : 0);
            const viewportHeight = window.innerHeight || doc.clientHeight || 0;
            const pageHeight = Math.max(
                doc.scrollHeight || 0,
                body ? body.scrollHeight || 0 : 0,
                doc.offsetHeight || 0,
                body ? body.offsetHeight || 0 : 0
            );
            return {
                scroll_y: Math.round(scrollY),
                viewport_height: Math.round(viewportHeight),
                page_height: Math.round(pageHeight),
                can_scroll_up: scrollY > 0,
                can_scroll_down: scrollY + viewportHeight + 2 < pageHeight
            };
        }"""
    )
    return dict(state or {})


def _visible_images(page, limit: int) -> list[dict[str, Any]]:
    return page.evaluate(
        """(limit) => {
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
                    width: Math.round(rect.width),
                    height: Math.round(rect.height),
                    x: Math.round(rect.x),
                    y: Math.round(rect.y)
                });
                if (rows.length >= limit) break;
            }
            return rows;
        }""",
        limit,
    )


def _visible_click_targets(page, limit: int) -> list[dict[str, Any]]:
    return page.evaluate(
        """(limit) => {
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
            const selectors = [
                'a[href]',
                'button',
                '[role="button"]',
                'input[type="button"]',
                'input[type="submit"]',
                'img'
            ];
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
        }""",
        limit,
    )


def _click_visible_target(page, index: int, limit: int) -> dict[str, Any]:
    return page.evaluate(
        """({ index, limit }) => {
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
            const selectors = [
                'a[href]',
                'button',
                '[role="button"]',
                'input[type="button"]',
                'input[type="submit"]',
                'img'
            ];
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
            if (!target) {
                return { ok: false, error: `No visible click target at index ${index}`, count: targets.length };
            }
            const info = meta(target, index);
            target.scrollIntoView({ block: 'center', inline: 'center' });
            target.click();
            return { ok: true, target: info, count: targets.length };
        }""",
        {"index": index, "limit": limit},
    )


def _count_visible_loaded_images(page) -> int:
    return int(
        page.evaluate(
            """() => {
                let count = 0;
                for (const img of document.querySelectorAll('img')) {
                    const rect = img.getBoundingClientRect();
                    const visible = rect.width > 20 && rect.height > 20
                        && rect.bottom > 0 && rect.right > 0
                        && rect.top < window.innerHeight && rect.left < window.innerWidth;
                    if (visible && img.complete && img.naturalWidth > 0 && img.naturalHeight > 0) {
                        count += 1;
                    }
                }
                return count;
            }"""
        )
        or 0
    )


def _wait_for_ready(page, args: argparse.Namespace, phase: str) -> list[str]:
    events: list[str] = []
    if args.wait_for_selector:
        try:
            page.wait_for_selector(args.wait_for_selector, timeout=args.wait_timeout_ms)
            events.append(f"{phase}: selector matched {args.wait_for_selector!r}")
        except Exception as exc:
            events.append(f"{phase}: selector wait failed: {exc}")

    if args.wait_for_visible_images > 0:
        deadline = time.perf_counter() + (args.wait_timeout_ms / 1000)
        last_count = 0
        while time.perf_counter() < deadline:
            try:
                last_count = _count_visible_loaded_images(page)
            except Exception:
                last_count = 0
            if last_count >= args.wait_for_visible_images:
                events.append(f"{phase}: visible loaded images reached {last_count}")
                break
            page.wait_for_timeout(250)
        else:
            events.append(
                f"{phase}: visible loaded images only {last_count}, "
                f"wanted {args.wait_for_visible_images}"
            )

    if args.wait_after_load_ms > 0:
        page.wait_for_timeout(args.wait_after_load_ms)
        events.append(f"{phase}: fixed wait {args.wait_after_load_ms}ms")
    return events


def _wait_for_action_settle(page, args: argparse.Namespace, phase: str) -> list[str]:
    events: list[str] = []
    if args.wait_until != "commit":
        try:
            page.wait_for_load_state(args.wait_until, timeout=args.wait_timeout_ms)
            events.append(f"{phase}: load state reached {args.wait_until}")
        except Exception as exc:
            events.append(f"{phase}: load state wait skipped/failed: {exc}")
    events.extend(_wait_for_ready(page, args, phase))
    return events


def _parse_actions(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [part.strip() for part in re.split(r"[,;]+", raw) if part.strip()]


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Browser Image Pipeline Report",
        "",
        f"- URL: {manifest['input_url']}",
        f"- Final URL: {manifest['final_url']}",
        f"- Title: {manifest['title'] or '(none)'}",
        f"- Screenshots: {len(manifest['screenshots'])}",
        f"- Cached images: {len(manifest['cached_images'])}",
        "",
        "## Screenshots",
        "",
    ]
    for shot in manifest["screenshots"]:
        lines.extend(
            [
                f"### {shot['ref']}",
                "",
                f"- scroll_y: {shot['scroll_y']}",
                f"- can_scroll_down: {shot['can_scroll_down']}",
                f"![]({Path(shot['path']).as_posix()})",
                "",
            ]
        )

    lines.extend(["## Cached Images", ""])
    if not manifest["cached_images"]:
        lines.append("No image responses were cached.")
    for image in manifest["cached_images"]:
        rel_path = Path(image["path"]).as_posix()
        lines.extend(
            [
                f"### {image['ref']}",
                "",
                f"- mime: `{image['mime']}`",
                f"- size: `{image['size_bytes']}` bytes",
                f"- url: {image['url']}",
                f"![]({rel_path})",
                "",
            ]
        )

    lines.extend(["## Actions", ""])
    if not manifest.get("actions"):
        lines.append("No explicit actions were executed.")
    for action in manifest.get("actions", []):
        status = "ok" if action["ok"] else "failed"
        lines.extend(
            [
                f"### {action['index']}: {action['action']} ({status})",
                "",
                f"- URL: {action['url']}",
                f"- Screenshot: `{action['screenshot_ref']}`",
            ]
        )
        if action.get("error"):
            lines.append(f"- Error: `{action['error']}`")
        if action.get("detail"):
            lines.append(f"- Detail: `{json.dumps(action['detail'], ensure_ascii=False)}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Playwright screenshots and image response caching.")
    parser.add_argument("--url", help="Page URL to open.")
    parser.add_argument("--self-test", action="store_true", help="Generate and open a local fixture page.")
    parser.add_argument("--out-dir", default="logs/browser_image_pipeline")
    parser.add_argument("--profile-dir", default="cache/browser_profile_test")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument(
        "--wait-until",
        choices=["commit", "domcontentloaded", "load", "networkidle"],
        default="domcontentloaded",
        help="Playwright navigation wait strategy.",
    )
    parser.add_argument("--wait-after-load-ms", type=int, default=1500)
    parser.add_argument("--wait-timeout-ms", type=int, default=10000)
    parser.add_argument("--wait-for-selector", help="Wait for a selector before the first screenshot.")
    parser.add_argument(
        "--wait-for-visible-images",
        type=int,
        default=0,
        help="Wait until at least this many visible img elements have loaded before screenshots.",
    )
    parser.add_argument("--scroll-steps", type=int, default=2)
    parser.add_argument("--scroll-pixels", type=int, default=700)
    parser.add_argument(
        "--actions",
        default="",
        help=(
            "Comma/semicolon separated actions. Supported: "
            "scroll[:pixels], click:N, click-selector:CSS, back, forward, wait[:ms], screenshot[:name]. "
            "When set, this replaces the default --scroll-steps loop."
        ),
    )
    parser.add_argument("--viewport-width", type=int, default=1280)
    parser.add_argument("--viewport-height", type=int, default=900)
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--min-image-bytes", type=int, default=1024)
    parser.add_argument("--visible-image-limit", type=int, default=20)
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--channel", choices=["chrome", "msedge"])
    parser.add_argument("--executable-path")
    parser.add_argument("--no-persistent", action="store_true", help="Use a temporary browser context.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    screenshots_dir = out_dir / "screenshots"
    images_dir = out_dir / "images"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        args.url = _make_self_test_page(out_dir)
        args.min_image_bytes = min(args.min_image_bytes, 1)
    if not args.url:
        print("Either --url or --self-test is required.", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install playwright", file=sys.stderr)
        return 2

    cached_by_sha: dict[str, CachedImage] = {}
    cached_by_url: dict[str, str] = {}
    screenshots: list[ScreenshotInfo] = []
    response_errors: list[str] = []
    wait_events: list[str] = []
    active_page_url = ""

    def cache_response(response) -> None:
        if len(cached_by_sha) >= args.max_images:
            return
        headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            return
        try:
            body = response.body()
        except Exception as exc:
            response_errors.append(f"{response.url}: {exc}")
            return
        if len(body) < args.min_image_bytes:
            return
        digest = hashlib.sha256(body).hexdigest()
        if digest in cached_by_sha:
            cached_by_url[response.url] = cached_by_sha[digest].ref
            return

        ext = _image_extension(content_type, response.url)
        ref = digest[:12]
        filename = f"{ref}{ext}"
        path = images_dir / filename
        path.write_bytes(body)
        try:
            source_page_url = response.frame.url
        except Exception:
            source_page_url = active_page_url

        cached = CachedImage(
            ref=ref,
            url=response.url,
            path=str(path),
            mime=content_type,
            size_bytes=len(body),
            sha256=digest,
            page_url=source_page_url,
        )
        cached_by_sha[digest] = cached
        cached_by_url[response.url] = ref
        print(f"[IMAGE] cached {ref} {len(body)} bytes {content_type} {response.url[:100]}")

    def take_screenshot(page, ref: str) -> None:
        state = _scroll_state(page)
        path = screenshots_dir / f"{ref}.png"
        page.screenshot(path=str(path), full_page=False, type="png")
        screenshots.append(
            ScreenshotInfo(
                ref=ref,
                path=str(path),
                scroll_y=int(state.get("scroll_y", 0)),
                viewport_height=int(state.get("viewport_height", 0)),
                page_height=int(state.get("page_height", 0)),
                can_scroll_up=bool(state.get("can_scroll_up", False)),
                can_scroll_down=bool(state.get("can_scroll_down", False)),
            )
        )
        print(
            f"[SHOT] {ref} y={state.get('scroll_y')} "
            f"page={state.get('page_height')} can_down={state.get('can_scroll_down')}"
        )

    def snapshot_after_action(
        page,
        action_index: int,
        action: str,
        ok: bool,
        detail: dict[str, Any],
        error: str = "",
    ) -> None:
        safe_action = re.sub(r"[^a-zA-Z0-9_.-]+", "_", action).strip("_")[:40] or "action"
        screenshot_ref = f"action_{action_index:02d}_{safe_action}"
        try:
            wait_events.extend(_wait_for_action_settle(page, args, f"action_{action_index:02d}_{action}"))
        except Exception as exc:
            detail = dict(detail)
            detail["wait_error"] = str(exc)
        try:
            take_screenshot(page, screenshot_ref)
        except Exception as exc:
            ok = False
            error = error or str(exc)
        action_results.append(
            ActionResult(
                index=action_index,
                action=action,
                ok=ok,
                url=page.url,
                title=page.title() or "",
                screenshot_ref=screenshot_ref,
                detail=detail,
                error=error,
            )
        )

    def run_action(page, action_index: int, raw_action: str) -> None:
        name, _, value = raw_action.partition(":")
        name = name.strip().lower()
        value = value.strip()
        try:
            if name == "scroll":
                pixels = int(value) if value else args.scroll_pixels
                page.mouse.wheel(0, pixels)
                snapshot_after_action(page, action_index, raw_action, True, {"pixels": pixels})
            elif name == "click":
                if not value:
                    raise ValueError("click action requires an index, for example click:0")
                clicked = _click_visible_target(page, int(value), args.visible_image_limit)
                ok = bool(clicked.get("ok"))
                snapshot_after_action(
                    page,
                    action_index,
                    raw_action,
                    ok,
                    dict(clicked),
                    str(clicked.get("error", "")),
                )
            elif name == "click-selector":
                if not value:
                    raise ValueError("click-selector action requires a CSS selector")
                page.click(value, timeout=args.wait_timeout_ms)
                snapshot_after_action(page, action_index, raw_action, True, {"selector": value})
            elif name == "back":
                response = page.go_back(wait_until=args.wait_until, timeout=args.timeout_ms)
                snapshot_after_action(
                    page,
                    action_index,
                    raw_action,
                    True,
                    {"status": response.status if response else None},
                )
            elif name == "forward":
                response = page.go_forward(wait_until=args.wait_until, timeout=args.timeout_ms)
                snapshot_after_action(
                    page,
                    action_index,
                    raw_action,
                    True,
                    {"status": response.status if response else None},
                )
            elif name == "wait":
                ms = int(value) if value else args.wait_after_load_ms
                page.wait_for_timeout(ms)
                snapshot_after_action(page, action_index, raw_action, True, {"ms": ms})
            elif name == "screenshot":
                snapshot_after_action(page, action_index, raw_action, True, {"name": value or ""})
            else:
                raise ValueError(f"unknown action {raw_action!r}")
        except Exception as exc:
            snapshot_after_action(page, action_index, raw_action, False, {}, str(exc))

    started = time.perf_counter()
    final_url = ""
    title = ""
    visible_images: list[dict[str, Any]] = []
    click_targets: list[dict[str, Any]] = []
    action_results: list[ActionResult] = []

    with sync_playwright() as pw:
        launch_kwargs: dict[str, Any] = {"headless": not args.headful}
        if args.channel:
            launch_kwargs["channel"] = args.channel
        if args.executable_path:
            launch_kwargs["executable_path"] = args.executable_path

        browser = None
        context = None
        if args.no_persistent:
            browser = pw.chromium.launch(**launch_kwargs)
            context = browser.new_context(
                viewport={"width": args.viewport_width, "height": args.viewport_height},
                user_agent=DEFAULT_UA,
                locale="zh-CN",
            )
        else:
            context = pw.chromium.launch_persistent_context(
                user_data_dir=str(Path(args.profile_dir)),
                viewport={"width": args.viewport_width, "height": args.viewport_height},
                user_agent=DEFAULT_UA,
                locale="zh-CN",
                **launch_kwargs,
            )

        try:
            page = context.new_page()
            page.set_default_timeout(args.timeout_ms)
            page.on("response", cache_response)
            try:
                page.goto(args.url, wait_until=args.wait_until, timeout=args.timeout_ms)
            except PlaywrightTimeoutError:
                print("[WARN] initial navigation timed out; continuing with current page state")
            wait_events.extend(_wait_for_ready(page, args, "initial"))

            final_url = page.url
            title = page.title() or ""
            take_screenshot(page, "viewport_00")

            actions = _parse_actions(args.actions)
            if actions:
                for action_index, action in enumerate(actions, start=1):
                    run_action(page, action_index, action)
            else:
                for index in range(max(0, args.scroll_steps)):
                    state = _scroll_state(page)
                    if not state.get("can_scroll_down"):
                        break
                    page.mouse.wheel(0, args.scroll_pixels)
                    wait_events.extend(_wait_for_ready(page, args, f"scroll_{index + 1:02d}"))
                    take_screenshot(page, f"viewport_{index + 1:02d}")

            visible_images = _visible_images(page, args.visible_image_limit)
            click_targets = _visible_click_targets(page, args.visible_image_limit)
            for item in cached_by_sha.values():
                if not item.page_url:
                    item.page_url = page.url
        finally:
            context.close()
            if browser is not None:
                browser.close()

    cached_images = sorted(cached_by_sha.values(), key=lambda item: item.size_bytes, reverse=True)
    send_candidates = [
        {
            "command": "image",
            "params": {
                "image_ref": item.ref,
            },
            "debug": {
                "url": item.url,
                "mime": item.mime,
                "size_bytes": item.size_bytes,
                "path": item.path,
            },
        }
        for item in cached_images[:10]
    ]

    manifest: dict[str, Any] = {
        "input_url": args.url,
        "final_url": final_url,
        "title": title,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "screenshots": [asdict(item) for item in screenshots],
        "cached_images": [asdict(item) for item in cached_images],
        "cached_by_url": cached_by_url,
        "visible_images": visible_images,
        "click_targets": click_targets,
        "send_candidates": send_candidates,
        "actions": [asdict(item) for item in action_results],
        "wait": {
            "wait_until": args.wait_until,
            "wait_after_load_ms": args.wait_after_load_ms,
            "wait_timeout_ms": args.wait_timeout_ms,
            "wait_for_selector": args.wait_for_selector,
            "wait_for_visible_images": args.wait_for_visible_images,
            "events": wait_events,
        },
        "response_errors": response_errors[:20],
        "notes": [
            "Screenshots model the default browser tool visual return.",
            "click_targets model browser_click(index=...) candidates in the current viewport.",
            "actions model browser_scroll/browser_click/browser_back/browser_forward tool calls.",
            "cached_images model browser_image_ref records for future send_message(image_ref=...).",
            "send_candidates are not sent; they show the intended send_message payload shape.",
        ],
    }

    manifest_path = out_dir / "manifest.json"
    report_path = out_dir / "report.md"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(report_path, manifest)

    print("")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report:   {report_path}")
    print(f"Screenshots:    {len(screenshots)}")
    print(f"Cached images:  {len(cached_images)}")
    if send_candidates:
        print(f"Top send ref:   {send_candidates[0]['params']['image_ref']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
