from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import app_state  # noqa: E402
import browser  # noqa: E402
import llm.prompt.user_prompt_builder as user_prompt_builder  # noqa: E402
from export_browser_world_samples import _sanitize, _summarize, _xml_text_and_image_count  # noqa: E402
from llm.session import ChatSession  # noqa: E402
from tools.browser_control import execute  # noqa: E402
from browser.session import browser_world_snapshot  # noqa: E402
from browser.world_prompt import render_browser_world_content  # noqa: E402

OUT = ROOT / "output" / "browser_prompt_samples"

REAL_SITE_SAMPLES: list[dict[str, Any]] = [
    {
        "slug": "example",
        "url": "https://example.com/",
        "category": "baseline",
    },
    {
        "slug": "hacker_news",
        "url": "https://news.ycombinator.com/",
        "category": "news_list",
    },
    {
        "slug": "wikipedia",
        "url": "https://www.wikipedia.org/",
        "category": "portal",
        "visible_images": 1,
    },
    {
        "slug": "github_trending",
        "url": "https://github.com/trending",
        "category": "code_list",
    },
    {
        "slug": "python_docs",
        "url": "https://docs.python.org/3/",
        "category": "docs",
    },
    {
        "slug": "mdn_js",
        "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        "category": "docs",
    },
    {
        "slug": "reddit",
        "url": "https://www.reddit.com/",
        "category": "social_feed",
    },
    {
        "slug": "amazon_search",
        "url": "https://www.amazon.com/s?k=laptop",
        "category": "commerce_search",
        "visible_images": 1,
    },
    {
        "slug": "adult_pornhub",
        "url": "https://www.pornhub.com/",
        "category": "adult_public_landing",
    },
    {
        "slug": "adult_xvideos",
        "url": "https://www.xvideos.com/",
        "category": "adult_public_landing",
    },
    {
        "slug": "adult_xhamster",
        "url": "https://xhamster.com/",
        "category": "adult_public_landing",
    },
    {
        "slug": "adult_nhentai",
        "url": "https://nhentai.net/",
        "category": "adult_anime_public_landing",
    },
    {
        "slug": "adult_ehentai",
        "url": "https://e-hentai.org/",
        "category": "adult_anime_public_landing",
    },
]


def _sample_by_slug(slug: str) -> dict[str, Any] | None:
    return next((sample for sample in REAL_SITE_SAMPLES if sample["slug"] == slug), None)


def _configure_runtime() -> None:
    app_state.config = {
        "vision": True,
        "generation": {
            "final_reminder": False,
            "world_multimodal_image_limit": -1,
        },
        "browser_control": {
            "multimodal_image_limit": -1,
        },
    }
    app_state.GEN = app_state.config["generation"]


def _sample_session(slug: str) -> ChatSession:
    session = ChatSession()
    session.set_conversation_meta("group", "browser_prompt_samples", "Browser prompt samples", 2)
    session._timezone = ZoneInfo("Asia/Shanghai")
    session._qq_id = "10000"
    session._qq_name = "BrowserTestBot"
    session._qq_card = "BrowserTestBot"
    session._model_name = "prompt-export"
    session._style_prompt = ""
    session._social_tips_group = ""
    session.build_dynamic_prompt_blocks = lambda now=None: {  # type: ignore[method-assign]
        "current_time": "现在是2026年的夏天，6月10日，凌晨0点0分",
        "memory": "",
        "goals": "",
    }
    session.add_to_context({
        "role": "user",
        "message_id": f"prompt-sample-{slug}",
        "timestamp": datetime.now(session._timezone).isoformat(),
        "sender_id": "20000",
        "sender_name": "PromptTester",
        "content_type": "text",
        "content": "请观察当前浏览器视口，并判断这个页面有哪些可见信息和可操作目标。",
    })
    return session


def _render_prompt_from_browser_content(slug: str, browser_content: str | list) -> str | list:
    session = _sample_session(slug)
    original = browser.build_browser_world_content
    try:
        browser.build_browser_world_content = lambda: browser_content
        return user_prompt_builder.build_main_user_prompt(session, consume_unread=False)
    finally:
        browser.build_browser_world_content = original


def _part_manifest(prompt: str | list) -> tuple[str, list[dict[str, Any]], int]:
    if isinstance(prompt, str):
        return prompt, [{"index": 0, "type": "text", "chars": len(prompt)}], 0

    text_parts: list[str] = []
    manifest: list[dict[str, Any]] = []
    image_count = 0
    for index, part in enumerate(prompt):
        if not isinstance(part, dict):
            manifest.append({"index": index, "type": type(part).__name__})
            continue
        part_type = str(part.get("type") or "")
        if part_type == "text":
            text = str(part.get("text") or "")
            text_parts.append(text)
            manifest.append({
                "index": index,
                "type": "text",
                "chars": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
        elif part_type == "image_url":
            image_count += 1
            url = str((part.get("image_url") or {}).get("url") or "")
            mime = ""
            if url.startswith("data:"):
                mime = url[5:].split(";", 1)[0]
            digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
            placeholder = f"\n<image_part index=\"{index}\" mime=\"{mime}\" chars=\"{len(url)}\" digest=\"{digest[:12]}\"/>\n"
            text_parts.append(placeholder)
            manifest.append({
                "index": index,
                "type": "image_url",
                "mime": mime,
                "chars": len(url),
                "sha256": digest,
            })
        else:
            manifest.append({"index": index, "type": part_type, "keys": sorted(part.keys())})
    return "".join(text_parts), manifest, image_count


def _strip_image_payloads(snapshot: dict[str, Any]) -> dict[str, Any]:
    stripped = json.loads(json.dumps(_sanitize(snapshot), ensure_ascii=False))
    if isinstance(stripped.get("viewport"), dict):
        stripped["viewport"].pop("data", None)
    for image in stripped.get("images") or []:
        if isinstance(image, dict):
            image.pop("data", None)
    return stripped


def _remove_previous_outputs(slug: str) -> None:
    for path in OUT.glob(f"{slug}.*"):
        if path.is_file():
            path.unlink()


def _write_outputs(sample: dict[str, Any], snapshot: dict[str, Any], browser_content: str | list, prompt: str | list) -> dict[str, Any]:
    slug = str(sample["slug"])
    _remove_previous_outputs(slug)
    xml, browser_image_parts = _xml_text_and_image_count(browser_content)
    prompt_text, prompt_manifest, prompt_image_parts = _part_manifest(prompt)
    summary = _summarize(slug, snapshot, xml, browser_image_parts)
    summary.update({
        "category": sample.get("category", ""),
        "prompt_text_chars": len(prompt_text),
        "prompt_parts": len(prompt_manifest),
        "prompt_image_parts": prompt_image_parts,
        "browser_image_parts": browser_image_parts,
    })

    world_path = OUT / f"{slug}.world.xml"
    prompt_path = OUT / f"{slug}.prompt.txt"
    snapshot_path = OUT / f"{slug}.snapshot.json"
    world_path.write_text(xml, encoding="utf-8")
    prompt_path.write_text(prompt_text, encoding="utf-8")
    snapshot_path.write_text(
        json.dumps(_strip_image_payloads(snapshot), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    parts_path = OUT / f"{slug}.prompt.parts.json"
    parts_path.write_text(
        json.dumps(prompt_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path = OUT / f"{slug}.summary.json"
    summary.update({
        "world_path": str(world_path),
        "prompt_path": str(prompt_path),
        "snapshot_path": str(snapshot_path),
        "prompt_parts_path": str(parts_path),
        "summary_path": str(summary_path),
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def export_prompt_samples(selected: set[str] | None = None) -> dict[str, Any]:
    _configure_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    try:
        for sample in REAL_SITE_SAMPLES:
            slug = str(sample["slug"])
            if selected is not None and slug not in selected:
                continue

            print(f"OPEN {slug}: {sample['url']}", flush=True)
            try:
                result = execute(
                    action="open",
                    url=str(sample["url"]),
                    wait_until="domcontentloaded",
                    wait_ms=int(sample.get("wait_ms", 900)),
                    visible_images=int(sample.get("visible_images", 0)),
                    timeout_ms=int(sample.get("timeout_ms", 30_000)),
                )
                if isinstance(result, dict) and result.get("error"):
                    errors.append({"slug": slug, "error": str(result.get("error"))})
                    print(f"ERROR {slug}: {result.get('error')}", flush=True)
                    continue

                snapshot = browser_world_snapshot()
                if not isinstance(snapshot, dict):
                    errors.append({"slug": slug, "error": "no snapshot"})
                    print(f"ERROR {slug}: no snapshot", flush=True)
                    continue

                browser_content = render_browser_world_content(snapshot)
                prompt = _render_prompt_from_browser_content(slug, browser_content)
                summary = _write_outputs(sample, snapshot, browser_content, prompt)
                summaries.append(summary)
                print(
                    "WROTE "
                    f"{slug}: targets={summary['click_targets']} "
                    f"text={summary['text_blocks']} "
                    f"images={summary['visible_images']} "
                    f"prompt_images={summary['prompt_image_parts']} "
                    f"world={summary['world_path']}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({"slug": slug, "error": repr(exc)})
                print(f"ERROR {slug}: {exc!r}", flush=True)
    finally:
        try:
            execute(action="close")
        except Exception as exc:
            errors.append({"slug": "close", "error": repr(exc)})

    result = {"summaries": summaries, "errors": errors}
    (OUT / "_index.summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export full browser prompt samples from real sites.")
    parser.add_argument("samples", nargs="*", help="Optional sample slugs. Defaults to all samples.")
    parser.add_argument("--list", action="store_true", help="List sample slugs and exit.")
    args = parser.parse_args()

    if args.list:
        for sample in REAL_SITE_SAMPLES:
            print(f"{sample['slug']}\t{sample['category']}\t{sample['url']}")
        return 0

    selected = set(args.samples) if args.samples else None
    unknown = sorted(slug for slug in (selected or set()) if _sample_by_slug(slug) is None)
    if unknown:
        print(f"Unknown samples: {', '.join(unknown)}", file=sys.stderr)
        return 2
    result = export_prompt_samples(selected)
    print(json.dumps({"count": len(result["summaries"]), "errors": result["errors"]}, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
