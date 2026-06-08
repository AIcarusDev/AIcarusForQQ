"""Test the Tavily-free flow: SERP candidates -> browser-opened pages.

The goal is to evaluate whether a browser-driven search tool can return enough
candidate links for the model to choose a page and then open it with a URL
browsing/extraction tool.

Usage:
    python scripts/test_serp_browse_pipeline.py --query "测试" --engine bing
    python scripts/test_serp_browse_pipeline.py --query "galgame ..." --engine all --open-top 3

Outputs:
    - JSON report with SERP candidates and browsed page previews.
    - Markdown report for quick manual inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from test_serp_search_quality import SearchResult, run_case  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class BrowsedPage:
    source_engine: str
    source_rank: int
    source_title: str
    source_url: str
    final_url: str
    final_domain: str
    page_title: str
    text_preview: str
    keyword_hits: int
    keyword_total: int
    score: float
    latency_ms: int
    error: str = ""


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _keywords_from_query(query: str) -> list[str]:
    raw = re.split(r"[\s,，。；;、|/]+", query)
    keywords: list[str] = []
    seen: set[str] = set()
    for item in raw:
        item = item.strip().strip('"').strip("'")
        if not item:
            continue
        # Keep short CJK words such as "义妹"; drop one-letter ASCII fragments.
        if len(item) <= 1 and item.isascii():
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        keywords.append(item)
    return keywords


def _keyword_score(text: str, keywords: list[str]) -> tuple[int, int, float]:
    lowered = text.lower()
    hits = 0
    for keyword in keywords:
        if keyword.lower() in lowered:
            hits += 1
    total = len(keywords)
    return hits, total, round((hits / total) * 100, 1) if total else 0.0


def _extract_body_text(page, max_chars: int) -> str:
    body_text = page.evaluate(
        """() => {
            const blocked = ['script', 'style', 'noscript', 'svg', 'canvas'];
            for (const tag of blocked) {
                document.querySelectorAll(tag).forEach((el) => el.remove());
            }
            return document.body ? document.body.innerText : '';
        }"""
    )
    text = _normalize_space(str(body_text or ""))
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + f"... [truncated, {len(text)} chars]"
    return text


def browse_candidate(
    context,
    candidate: SearchResult,
    engine: str,
    keywords: list[str],
    timeout_ms: int,
    max_chars: int,
) -> BrowsedPage:
    started = time.perf_counter()
    page = None
    try:
        page = context.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(candidate.url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(800)
        final_url = page.url
        page_title = _normalize_space(page.title() or "")
        text_preview = _extract_body_text(page, max_chars)
        score_text = " ".join([
            candidate.title,
            candidate.snippet,
            final_url,
            page_title,
            text_preview,
        ])
        hits, total, score = _keyword_score(score_text, keywords)
        return BrowsedPage(
            source_engine=engine,
            source_rank=candidate.rank,
            source_title=candidate.title,
            source_url=candidate.url,
            final_url=final_url,
            final_domain=_domain(final_url),
            page_title=page_title,
            text_preview=text_preview,
            keyword_hits=hits,
            keyword_total=total,
            score=score,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        return BrowsedPage(
            source_engine=engine,
            source_rank=candidate.rank,
            source_title=candidate.title,
            source_url=candidate.url,
            final_url="",
            final_domain="",
            page_title="",
            text_preview="",
            keyword_hits=0,
            keyword_total=len(keywords),
            score=0.0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=str(exc),
        )
    finally:
        if page is not None:
            page.close()


def _dedupe_candidates(candidates: list[tuple[str, SearchResult]]) -> list[tuple[str, SearchResult]]:
    deduped: list[tuple[str, SearchResult]] = []
    seen: set[str] = set()
    for engine, item in candidates:
        key = item.url.rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((engine, item))
    return deduped


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# SERP Browse Pipeline Report",
        "",
        f"- Query: `{report['query']}`",
        f"- Engines: `{', '.join(report['engines'])}`",
        f"- Keywords: `{', '.join(report['keywords'])}`",
        "",
        "## SERP Candidates",
        "",
        "| Engine | Rank | Title | URL | Snippet |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for item in report["candidates"]:
        snippet = str(item["snippet"]).replace("|", "\\|")
        title = str(item["title"]).replace("|", "\\|")
        lines.append(
            f"| {item['engine']} | {item['rank']} | {title} | {item['url']} | {snippet} |"
        )

    lines.extend([
        "",
        "## Browsed Pages",
        "",
        "| Score | Engine | Rank | Final Domain | Page Title | Final URL |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ])
    for item in report["browsed_pages"]:
        page_title = str(item["page_title"]).replace("|", "\\|")
        status = f"{item['score']:.1f}"
        if item["error"]:
            status = "ERR"
        lines.append(
            f"| {status} | {item['source_engine']} | {item['source_rank']} | "
            f"{item['final_domain']} | {page_title} | {item['final_url'] or item['source_url']} |"
        )

    for item in report["browsed_pages"]:
        lines.extend([
            "",
            f"### {item['source_engine']} #{item['source_rank']}: {item['page_title'] or item['source_title']}",
            "",
            f"- Source URL: {item['source_url']}",
            f"- Final URL: {item['final_url'] or '(failed)'}",
            f"- Keyword hits: {item['keyword_hits']}/{item['keyword_total']}",
            f"- Latency: {item['latency_ms']}ms",
        ])
        if item["error"]:
            lines.append(f"- Error: `{item['error']}`")
        if item["text_preview"]:
            lines.extend(["", item["text_preview"]])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SERP candidates followed by browser page extraction.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--engine", choices=["google", "duckduckgo", "bing", "baidu", "all"], default="all")
    parser.add_argument("--limit", type=int, default=8, help="SERP candidates per engine.")
    parser.add_argument("--open-top", type=int, default=5, help="Number of deduped candidates to browse.")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--market", default="en-US")
    parser.add_argument("--setlang", default="en-US")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--page-chars", type=int, default=1600)
    parser.add_argument("--headful", action="store_true")
    parser.add_argument("--channel", choices=["chrome", "msedge"])
    parser.add_argument("--executable-path")
    parser.add_argument("--output", default="logs/serp_browse_pipeline.json")
    parser.add_argument("--markdown", default="logs/serp_browse_pipeline.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engines = ["google", "duckduckgo", "bing", "baidu"] if args.engine == "all" else [args.engine]
    keywords = _keywords_from_query(args.query)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install playwright", file=sys.stderr)
        return 2

    all_candidates: list[tuple[str, SearchResult]] = []
    browsed_pages: list[BrowsedPage] = []

    with sync_playwright() as pw:
        launch_kwargs: dict[str, Any] = {"headless": not args.headful}
        if args.channel:
            launch_kwargs["channel"] = args.channel
        if args.executable_path:
            launch_kwargs["executable_path"] = args.executable_path
        browser = pw.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale=args.setlang,
        )

        case = {"query": args.query, "expected_terms": keywords, "expected_domains": []}
        for engine in engines:
            report = run_case(
                context,
                engine,
                case,
                args.limit,
                args.timeout_ms,
                args.market,
                args.setlang,
            )
            print(
                f"[SERP] {engine}: ok={report.ok} blocked={report.blocked} "
                f"results={report.result_count} score={report.score:.1f}"
            )
            for result in report.results:
                all_candidates.append((engine, result))
            if args.delay > 0:
                time.sleep(args.delay)

        deduped = _dedupe_candidates(all_candidates)
        for engine, candidate in deduped[: max(0, args.open_top)]:
            page_report = browse_candidate(
                context,
                candidate,
                engine,
                keywords,
                args.timeout_ms,
                args.page_chars,
            )
            browsed_pages.append(page_report)
            status = "ERR" if page_report.error else f"{page_report.score:.1f}"
            print(
                f"[OPEN] {engine} #{candidate.rank}: score={status} "
                f"domain={page_report.final_domain or '-'}"
            )
            if args.delay > 0:
                time.sleep(args.delay)

        browser.close()

    candidate_rows = [
        {
            "engine": engine,
            "rank": item.rank,
            "title": item.title,
            "url": item.url,
            "snippet": item.snippet,
            "domain": item.domain,
        }
        for engine, item in all_candidates
    ]
    report_data = {
        "query": args.query,
        "engines": engines,
        "keywords": keywords,
        "candidates": candidate_rows,
        "browsed_pages": [asdict(item) for item in browsed_pages],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(Path(args.markdown), report_data)

    print(f"\nWrote JSON: {output}")
    print(f"Wrote Markdown: {args.markdown}")
    if browsed_pages:
        best = max(browsed_pages, key=lambda item: item.score)
        print(
            f"Best opened page: score={best.score:.1f} "
            f"hits={best.keyword_hits}/{best.keyword_total} url={best.final_url or best.source_url}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
