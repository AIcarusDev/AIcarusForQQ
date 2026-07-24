"""Isolated experiment for cache-first browser image materialization.

This script intentionally does not import or modify the production browser or
QQ send paths. It starts a local HTTP fixture, observes image resources through
Chrome DevTools Protocol, exposes a minimal projection to a simulated LLM, and
materializes only the selected original image.

Run:
    python scripts/browser_image_materialization_check.py
    python scripts/browser_image_materialization_check.py --json
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from playwright.sync_api import Browser, CDPSession, Error, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from browser.runtime_paths import system_chrome_path  # noqa: E402
from browser.image_resources import (  # noqa: E402
    BrowserImageArtifactStore,
    BrowserImageResourceRegistry,
)
from browser.session import BrowserSession  # noqa: E402


INTERNAL_ONLY_FIELDS = {
    "request_id",
    "frame_id",
    "page_url",
    "observed_at",
    "materialized_image_ref",
}
MODEL_IMAGE_FIELDS = {
    "resource_ref",
    "source_url",
    "alt",
    "rect",
    "natural_size",
}


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int], label: str) -> bytes:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, size[0] - 9, size[1] - 9), outline=(255, 255, 255), width=4)
    draw.text((18, 18), label, fill=(255, 255, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


TARGET_BYTES = _png_bytes((640, 420), (153, 38, 92), "TARGET ORIGINAL")
DECOY_BYTES = _png_bytes((96, 96), (34, 105, 125), "DECOY")


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    counters: dict[str, int] = {}

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_bytes(self, body: bytes, content_type: str, *, cache: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600, immutable")
            self.send_header("ETag", f'"{hashlib.sha256(body).hexdigest()}"')
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = self.path.split("?", 1)[0]
        type(self).counters[path] = type(self).counters.get(path, 0) + 1
        if path == "/":
            html = b"""<!doctype html>
<meta charset="utf-8">
<title>Browser image materialization fixture</title>
<style>
  body { margin: 24px; font-family: sans-serif; }
  .target { position: relative; width: 640px; }
  .target img { display: block; width: 640px; height: 420px; }
  .overlay {
    position: absolute; left: 0; right: 0; bottom: 0; height: 54px;
    display: flex; align-items: center; justify-content: flex-end; gap: 18px;
    padding: 0 18px; box-sizing: border-box;
    color: white; background: rgba(0, 0, 0, 0.75);
  }
  .decoy { width: 96px; height: 96px; margin-top: 20px; }
</style>
<div class="target">
  <img id="target" src="/target.png" alt="selected artwork">
  <div class="overlay" aria-label="page controls">like share more</div>
</div>
<img id="decoy" class="decoy" src="/decoy.png" alt="small avatar">
"""
            self._send_bytes(html, "text/html; charset=utf-8")
            return
        if path == "/target.png":
            self._send_bytes(TARGET_BYTES, "image/png", cache=True)
            return
        if path == "/decoy.png":
            self._send_bytes(DECOY_BYTES, "image/png", cache=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


class FixtureServer:
    def __init__(self) -> None:
        FixtureHandler.counters = {}
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/"

    def request_count(self, path: str) -> int:
        return FixtureHandler.counters.get(path, 0)

    def __enter__(self) -> "FixtureServer":
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@dataclass(frozen=True)
class InternalImageResource:
    resource_ref: str
    source_url: str
    page_url: str
    request_id: str
    frame_id: str
    observed_at: float
    alt: str
    rect: dict[str, int]
    natural_width: int
    natural_height: int
    materialized_image_ref: str | None = None

    def model_projection(self, *, expose_source_url: bool = True) -> dict[str, Any]:
        projection: dict[str, Any] = {
            "resource_ref": self.resource_ref,
            "alt": self.alt,
            "rect": dict(self.rect),
            "natural_size": [self.natural_width, self.natural_height],
        }
        if expose_source_url:
            projection["source_url"] = self.source_url
        return projection


class SimulatedLlm:
    """Deterministic stand-in that can only inspect model-visible fields."""

    def choose(self, model_world: dict[str, Any]) -> str:
        images = model_world.get("images")
        if not isinstance(images, list) or not images:
            raise AssertionError("model received no image candidates")
        for image in images:
            leaked = INTERNAL_ONLY_FIELDS.intersection(image)
            if leaked:
                raise AssertionError(f"internal fields leaked to model: {sorted(leaked)}")
            unexpected = set(image).difference(MODEL_IMAGE_FIELDS)
            if unexpected:
                raise AssertionError(f"unexpected model image fields: {sorted(unexpected)}")

        def area(item: dict[str, Any]) -> int:
            size = item.get("natural_size") or [0, 0]
            return int(size[0]) * int(size[1])

        selected = max(images, key=area)
        return str(selected["resource_ref"])


@dataclass(frozen=True)
class MaterializedImage:
    image_ref: str
    path: str
    mime: str
    size_bytes: int
    sha256: str
    strategy: str


class OriginalImageMaterializer:
    def __init__(self, cdp: CDPSession, output_dir: Path) -> None:
        self._cdp = cdp
        self._output_dir = output_dir

    @staticmethod
    def _decode_content(payload: dict[str, Any]) -> bytes:
        content = str(payload.get("body", payload.get("content", "")))
        if payload.get("base64Encoded"):
            return base64.b64decode(content)
        return content.encode("latin-1")

    def _from_response_body(self, resource: InternalImageResource) -> bytes:
        payload = self._cdp.send(
            "Network.getResponseBody",
            {"requestId": resource.request_id},
        )
        return self._decode_content(payload)

    def _from_page_resource(self, resource: InternalImageResource) -> bytes:
        payload = self._cdp.send(
            "Page.getResourceContent",
            {"frameId": resource.frame_id, "url": resource.source_url},
        )
        return self._decode_content(payload)

    def materialize(
        self,
        resource: InternalImageResource,
        *,
        strategies: tuple[str, ...] = ("response_body", "page_resource"),
    ) -> MaterializedImage:
        errors: list[str] = []
        data: bytes | None = None
        used_strategy = ""
        for strategy in strategies:
            try:
                if strategy == "response_body":
                    data = self._from_response_body(resource)
                elif strategy == "page_resource":
                    data = self._from_page_resource(resource)
                else:
                    raise ValueError(f"unknown materialization strategy: {strategy}")
            except Exception as exc:
                errors.append(f"{strategy}: {type(exc).__name__}: {exc}")
                continue
            if data:
                used_strategy = strategy
                break
            errors.append(f"{strategy}: empty body")

        if not data:
            raise RuntimeError("original image unavailable; " + " | ".join(errors))

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                detected_format = str(image.format or "").upper()
        except Exception as exc:
            raise RuntimeError(f"materialized bytes are not a valid image: {exc}") from exc
        if detected_format != "PNG":
            raise RuntimeError(f"unexpected fixture image format: {detected_format}")

        digest = hashlib.sha256(data).hexdigest()
        image_ref = digest[:12]
        output_path = self._output_dir / f"{image_ref}.png"
        output_path.write_bytes(data)
        return MaterializedImage(
            image_ref=image_ref,
            path=str(output_path),
            mime="image/png",
            size_bytes=len(data),
            sha256=digest,
            strategy=used_strategy,
        )


class MethodFailingCdp:
    """Real CDP proxy used to force production fallback branches."""

    def __init__(self, cdp: CDPSession, failed_methods: set[str]) -> None:
        self._cdp = cdp
        self._failed_methods = set(failed_methods)

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if method in self._failed_methods:
            raise RuntimeError(f"forced {method} failure")
        return self._cdp.send(method, params or {})


def _launch_browser(playwright: Any) -> tuple[Browser, str]:
    try:
        return playwright.chromium.launch(headless=True), "playwright-chromium"
    except Error:
        executable = system_chrome_path()
        return (
            playwright.chromium.launch(
                executable_path=executable,
                headless=True,
                args=["--no-first-run", "--no-default-browser-check"],
            ),
            executable,
        )


def _collect_resources(page: Any, responses: dict[str, dict[str, str]]) -> list[InternalImageResource]:
    rows = page.locator("img").evaluate_all(
        """images => images.map(image => {
            const rect = image.getBoundingClientRect();
            return {
                source_url: image.currentSrc || image.src,
                alt: image.alt || "",
                rect: {
                    x: Math.round(rect.x),
                    y: Math.round(rect.y),
                    width: Math.round(rect.width),
                    height: Math.round(rect.height)
                },
                natural_width: image.naturalWidth,
                natural_height: image.naturalHeight
            };
        })"""
    )
    resources: list[InternalImageResource] = []
    for row in rows:
        source_url = str(row["source_url"])
        response = responses.get(source_url)
        if response is None:
            raise AssertionError(f"missing CDP response metadata for {source_url}")
        resource_ref = "br_" + hashlib.sha256(
            f"{response['frame_id']}:{source_url}".encode("utf-8")
        ).hexdigest()[:12]
        resources.append(
            InternalImageResource(
                resource_ref=resource_ref,
                source_url=source_url,
                page_url=page.url,
                request_id=response["request_id"],
                frame_id=response["frame_id"],
                observed_at=0.0,
                alt=str(row["alt"]),
                rect={key: int(value) for key, value in row["rect"].items()},
                natural_width=int(row["natural_width"]),
                natural_height=int(row["natural_height"]),
            )
        )
    return resources


def run_experiment() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    with FixtureServer() as server, tempfile.TemporaryDirectory(
        prefix="aicq-browser-materialization-"
    ) as temp_dir, sync_playwright() as playwright:
        browser, browser_source = _launch_browser(playwright)
        try:
            context = browser.new_context(viewport={"width": 900, "height": 700})
            page = context.new_page()
            cdp = context.new_cdp_session(page)
            cdp.send(
                "Network.enable",
                {
                    "maxTotalBufferSize": 16 * 1024 * 1024,
                    "maxResourceBufferSize": 8 * 1024 * 1024,
                },
            )
            cdp.send("Page.enable")
            responses: dict[str, dict[str, str]] = {}

            def on_response(event: dict[str, Any]) -> None:
                response = event.get("response") or {}
                mime = str(response.get("mimeType") or "")
                if not mime.startswith("image/"):
                    return
                responses[str(response["url"])] = {
                    "request_id": str(event["requestId"]),
                    "frame_id": str(event["frameId"]),
                }

            cdp.on("Network.responseReceived", on_response)
            page.goto(server.url, wait_until="networkidle")
            resources = _collect_resources(page, responses)
            target = next(item for item in resources if item.source_url.endswith("/target.png"))
            decoy = next(item for item in resources if item.source_url.endswith("/decoy.png"))

            model_world = {
                "page": {"url": page.url, "title": page.title()},
                "images": [item.model_projection() for item in resources],
            }
            selected_ref = SimulatedLlm().choose(model_world)
            if selected_ref != target.resource_ref:
                raise AssertionError("simulated LLM did not choose the largest target image")
            checks.append({"name": "model_field_boundary_and_selection", "ok": True})

            output_dir = Path(temp_dir) / "artifacts"
            output_dir.mkdir()
            materializer = OriginalImageMaterializer(cdp, output_dir)

            requests_before = server.request_count("/target.png")
            from_response = materializer.materialize(target, strategies=("response_body",))
            if Path(from_response.path).read_bytes() != TARGET_BYTES:
                raise AssertionError("response-body materialization changed original bytes")
            if server.request_count("/target.png") != requests_before:
                raise AssertionError("response-body materialization caused another HTTP request")
            checks.append(
                {
                    "name": "exact_response_body_without_network",
                    "ok": True,
                    "strategy": from_response.strategy,
                }
            )

            Path(from_response.path).unlink()
            requests_before = server.request_count("/target.png")
            from_resource = materializer.materialize(target, strategies=("page_resource",))
            if Path(from_resource.path).read_bytes() != TARGET_BYTES:
                raise AssertionError("page-resource materialization changed original bytes")
            if server.request_count("/target.png") != requests_before:
                raise AssertionError("page-resource materialization caused another HTTP request")
            checks.append(
                {
                    "name": "page_resource_cache_without_network",
                    "ok": True,
                    "strategy": from_resource.strategy,
                }
            )

            artifact_names = sorted(path.name for path in output_dir.iterdir())
            expected_name = f"{hashlib.sha256(TARGET_BYTES).hexdigest()[:12]}.png"
            if artifact_names != [expected_name]:
                raise AssertionError(
                    f"only the selected original may persist: {artifact_names}"
                )
            if decoy.materialized_image_ref is not None:
                raise AssertionError("unselected resource was unexpectedly materialized")
            checks.append({"name": "selected_resource_only", "ok": True})

            invalid = InternalImageResource(
                **{
                    **asdict(target),
                    "resource_ref": "br_invalid",
                    "request_id": "invalid-request",
                    "frame_id": "invalid-frame",
                    "source_url": server.url + "missing.png",
                }
            )
            before_failure = artifact_names
            try:
                materializer.materialize(
                    invalid,
                    strategies=("response_body", "page_resource"),
                )
            except RuntimeError as exc:
                failure_message = str(exc)
            else:
                raise AssertionError("invalid resource unexpectedly materialized")
            after_failure = sorted(path.name for path in output_dir.iterdir())
            if after_failure != before_failure:
                raise AssertionError("failure created an artifact or screenshot fallback")
            checks.append(
                {
                    "name": "explicit_failure_without_screenshot_fallback",
                    "ok": True,
                    "error": failure_message,
                }
            )

            # Exercise the production registry/materializer against the same
            # real Chrome state.  The deterministic LLM above remains the only
            # simulated component.
            production_dir = Path(temp_dir) / "production-artifacts"
            production_registry = BrowserImageResourceRegistry()
            for observed in resources:
                production_registry.observe_request(
                    observed.request_id,
                    observed.source_url,
                )
                production_registry.observe_response(
                    request_id=observed.request_id,
                    frame_id=observed.frame_id,
                    url=observed.source_url,
                    mime="image/png",
                    status=200,
                )

            production_resources = []
            for index, observed in enumerate(resources):
                registered = production_registry.register(
                    source_url=observed.source_url,
                    page_url=observed.page_url,
                    identity=f"main:{index}",
                    alt=observed.alt,
                    rect=observed.rect,
                    natural_size=(observed.natural_width, observed.natural_height),
                )
                if registered is None:
                    raise AssertionError("production registry rejected a real DOM image")
                production_resources.append(registered)
            production_target = next(
                item for item in production_resources
                if item.source_url.endswith("/target.png")
            )

            production_session = BrowserSession()
            production_session.image_resources = production_registry
            production_session.image_artifacts = BrowserImageArtifactStore(production_dir)
            production_session._cdp_by_request_id[production_target.request_id] = cdp
            production_session._cdp_by_frame_id[production_target.frame_id] = cdp

            requests_before = server.request_count("/target.png")
            production_response = production_session.materialize_resources(
                [production_target.resource_ref]
            )[0]
            if production_response["strategy"] != "response_body":
                raise AssertionError("production did not prefer exact observed response bytes")
            if server.request_count("/target.png") != requests_before:
                raise AssertionError("production response-body path caused another request")

            production_session._cdp_by_request_id[production_target.request_id] = MethodFailingCdp(
                cdp,
                {"Network.getResponseBody"},
            )
            production_session._cdp_by_frame_id[production_target.frame_id] = (
                production_session._cdp_by_request_id[production_target.request_id]
            )
            production_page_cache = production_session.materialize_resources(
                [production_target.resource_ref]
            )[0]
            if production_page_cache["strategy"] != "page_resource":
                raise AssertionError("production did not use Page.getResourceContent fallback")
            if server.request_count("/target.png") != requests_before:
                raise AssertionError("production page-resource path caused another request")

            browser_network_cdp = MethodFailingCdp(
                cdp,
                {"Network.getResponseBody", "Page.getResourceContent"},
            )
            production_session._cdp_by_request_id[production_target.request_id] = browser_network_cdp
            production_session._cdp_by_frame_id[production_target.frame_id] = browser_network_cdp
            production_browser_cache = production_session.materialize_resources(
                [production_target.resource_ref]
            )[0]
            if production_browser_cache["strategy"] != "browser_network":
                raise AssertionError("production did not reach the Chrome network-stack fallback")
            if server.request_count("/target.png") != requests_before:
                raise AssertionError("Chrome cache fallback unexpectedly transferred the image again")
            if production_browser_cache["confirmation_reasons"] != ["resource_identity_unproven"]:
                raise AssertionError("browser-network fallback did not expose an explicit risk reason")

            persisted_files = sorted(path.name for path in production_dir.iterdir())
            if len(persisted_files) != 2 or not any(name.endswith(".json") for name in persisted_files):
                raise AssertionError(f"production persisted unexpected artifacts: {persisted_files}")
            production_bytes = production_session.image_artifacts.read(
                production_response["image_ref"]
            )
            if production_bytes is None or production_bytes[0] != TARGET_BYTES:
                raise AssertionError("production artifact differs from source bytes")
            production_session._cdp_by_page[id(page)] = cdp
            production_session._main_frame_id_by_cdp[id(cdp)] = production_target.frame_id
            missing_resource = production_registry.register(
                source_url=server.url + "missing.png",
                page_url=page.url,
                identity="main:missing",
                alt="missing",
                rect={"x": 0, "y": 0, "width": 320, "height": 200},
                natural_size=(320, 200),
            )
            if missing_resource is None:
                raise AssertionError("production registry rejected failure fixture")
            before_failure_files = sorted(path.name for path in production_dir.iterdir())
            try:
                production_session.materialize_resources([missing_resource.resource_ref])
            except Exception as exc:
                production_failure = str(exc)
            else:
                raise AssertionError("production materialized a missing resource")
            after_failure_files = sorted(path.name for path in production_dir.iterdir())
            if after_failure_files != before_failure_files:
                raise AssertionError("production failure persisted an artifact or screenshot")
            checks.append({
                "name": "production_cache_fallback_order_and_exact_bytes",
                "ok": True,
                "strategies": [
                    production_response["strategy"],
                    production_page_cache["strategy"],
                    production_browser_cache["strategy"],
                ],
            })
            checks.append({
                "name": "production_failure_without_ref_or_screenshot",
                "ok": True,
                "error": production_failure,
            })

            return {
                "ok": True,
                "browser": browser_source,
                "fixture_url": server.url,
                "model_world": model_world,
                "selected_resource_ref": selected_ref,
                "materialized": asdict(from_resource),
                "http_requests": dict(FixtureHandler.counters),
                "checks": checks,
            }
        finally:
            browser.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test cache-first, model-selected browser image materialization."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the complete machine-readable experiment report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = run_experiment()
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("PASS: browser image lazy-materialization experiment")
        for check in report["checks"]:
            print(f"  - {check['name']}: ok")
        print(f"  selected: {report['selected_resource_ref']}")
        print(f"  image_ref: {report['materialized']['image_ref']}")
        print(f"  HTTP requests: {report['http_requests']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
