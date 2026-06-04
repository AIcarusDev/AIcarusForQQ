"""Probe browser-based SERP search quality with Playwright.

This script is intentionally standalone. It does not call Tavily or mutate the
runtime tool registry. It opens public search result pages, parses organic
results, and emits a small quality report that is useful when deciding whether
SERP scraping is good enough for the candidate-URL stage of web_search.

Usage:
    python scripts/test_serp_search_quality.py --engine duckduckgo --limit 5
    python scripts/test_serp_search_quality.py --engine bing --query "Playwright Python screenshot"
    python scripts/test_serp_search_quality.py --engine all --output logs/serp_quality.json --markdown logs/serp_quality.md

Notes:
    - Google and DuckDuckGo are more likely to show captcha/consent/interstitial pages.
    - DuckDuckGo uses html.duckduckgo.com, which is lighter and easier to parse.
    - Bing and Baidu use regular result pages and may vary more by region/account/IP.
    - SERP scraping can be blocked or rate-limited; blocked pages are reported
      instead of treated as valid zero-result searches.
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
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "query": "AIcarusForQQ GitHub",
        "expected_terms": ["AIcarusForQQ", "GitHub"],
        "expected_domains": ["github.com"],
    },
    {
        "query": "Playwright Python page screenshot full_page",
        "expected_terms": ["Playwright", "Python", "screenshot"],
        "expected_domains": ["playwright.dev"],
    },
    {
        "query": "Quart Python websocket documentation",
        "expected_terms": ["Quart", "websocket", "Python"],
        "expected_domains": ["quart.palletsprojects.com"],
    },
    {
        "query": "SQLite FTS5 documentation bm25",
        "expected_terms": ["SQLite", "FTS5", "bm25"],
        "expected_domains": ["sqlite.org"],
    },
    {
        "query": "OpenAI API official docs tools function calling",
        "expected_terms": ["OpenAI", "API", "tools", "function"],
        "expected_domains": ["platform.openai.com"],
    },

]


@dataclass
class SearchResult:
    rank: int
    title: str
    url: str
    snippet: str
    domain: str


@dataclass
class QualityReport:
    engine: str
    query: str
    ok: bool
    blocked: bool
    latency_ms: int
    result_count: int
    unique_domain_count: int
    expected_term_hits: int
    expected_term_total: int
    expected_domain_hits: int
    expected_domain_total: int
    score: float
    error: str
    results: list[SearchResult]


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


def _unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    if "duckduckgo.com" not in parsed.netloc:
        return url
    params = parse_qs(parsed.query)
    uddg = params.get("uddg")
    if uddg:
        return unquote(uddg[0])
    return url


def _looks_blocked(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "captcha",
        "unusual traffic",
        "verify you are human",
        "are you a robot",
        "access denied",
        "too many requests",
        "rate limit",
        "异常流量",
        "自动程序",
        "确认这些请求是由您",
    ]
    return any(marker in lowered for marker in markers)


def _score(case: dict[str, Any], results: list[SearchResult], blocked: bool) -> tuple[float, dict[str, int]]:
    expected_terms = [str(t).lower() for t in case.get("expected_terms", []) if str(t).strip()]
    expected_domains = [str(d).lower() for d in case.get("expected_domains", []) if str(d).strip()]
    haystacks = [
        f"{r.title} {r.snippet} {r.url}".lower()
        for r in results
    ]
    domains = {r.domain for r in results if r.domain}

    term_hits = 0
    for term in expected_terms:
        if any(term in text for text in haystacks):
            term_hits += 1

    domain_hits = 0
    for expected in expected_domains:
        if any(domain == expected or domain.endswith("." + expected) for domain in domains):
            domain_hits += 1

    if blocked:
        return 0.0, {
            "term_hits": term_hits,
            "term_total": len(expected_terms),
            "domain_hits": domain_hits,
            "domain_total": len(expected_domains),
        }

    result_score = min(len(results), 5) / 5
    diversity_score = min(len(domains), 4) / 4
    term_score = (term_hits / len(expected_terms)) if expected_terms else 1.0
    domain_score = (domain_hits / len(expected_domains)) if expected_domains else 1.0
    total = (
        result_score * 0.25
        + diversity_score * 0.15
        + term_score * 0.35
        + domain_score * 0.25
    )
    return round(total * 100, 1), {
        "term_hits": term_hits,
        "term_total": len(expected_terms),
        "domain_hits": domain_hits,
        "domain_total": len(expected_domains),
    }


def _dedupe_results(results: list[SearchResult], limit: int) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    seen: set[str] = set()
    for item in results:
        key = item.url.rstrip("/") or f"{item.title}|{item.snippet}"
        if not key or key in seen:
            continue
        seen.add(key)
        item.rank = len(deduped) + 1
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _parse_duckduckgo(page, limit: int) -> list[SearchResult]:
    raw = page.evaluate(
        """() => Array.from(document.querySelectorAll('.result')).map((node) => {
            const link = node.querySelector('a.result__a');
            const snippet = node.querySelector('.result__snippet');
            return {
                title: link ? (link.innerText || link.textContent || link.getAttribute('aria-label') || '') : '',
                url: link ? link.href : '',
                snippet: snippet ? (snippet.innerText || snippet.textContent || '') : ''
            };
        })"""
    )
    results = []
    for item in raw or []:
        url = _unwrap_duckduckgo_url(str(item.get("url") or ""))
        results.append(SearchResult(
            rank=0,
            title=_normalize_space(str(item.get("title") or "")),
            url=url,
            snippet=_normalize_space(str(item.get("snippet") or "")),
            domain=_domain(url),
        ))
    return _dedupe_results(results, limit)


def _parse_bing(page, limit: int) -> list[SearchResult]:
    raw = page.evaluate(
        """() => Array.from(document.querySelectorAll('li.b_algo')).map((node) => {
            const link = node.querySelector('h2 a');
            const snippet = node.querySelector('.b_caption p, p');
            return {
                title: link ? (link.innerText || link.textContent || link.getAttribute('aria-label') || '') : '',
                url: link ? link.href : '',
                snippet: snippet ? (snippet.innerText || snippet.textContent || '') : ''
            };
        })"""
    )
    results = []
    for item in raw or []:
        url = str(item.get("url") or "")
        results.append(SearchResult(
            rank=0,
            title=_normalize_space(str(item.get("title") or "")),
            url=url,
            snippet=_normalize_space(str(item.get("snippet") or "")),
            domain=_domain(url),
        ))
    return _dedupe_results(results, limit)


def _parse_baidu(page, limit: int) -> list[SearchResult]:
    raw = page.evaluate(
        """() => Array.from(document.querySelectorAll('.result, .c-container')).map((node) => {
            const link = node.querySelector('h3 a, .t a, a');
            const snippet = node.querySelector('.c-abstract, .content-right_8Zs40, .c-span-last, span');
            return {
                title: link ? (link.innerText || link.textContent || link.getAttribute('aria-label') || '') : '',
                url: link ? link.href : '',
                snippet: snippet ? (snippet.innerText || snippet.textContent || '') : ''
            };
        })"""
    )
    results = []
    for item in raw or []:
        url = str(item.get("url") or "")
        results.append(SearchResult(
            rank=0,
            title=_normalize_space(str(item.get("title") or "")),
            url=url,
            snippet=_normalize_space(str(item.get("snippet") or "")),
            domain=_domain(url),
        ))
    return _dedupe_results(results, limit)


def _parse_google(page, limit: int) -> list[SearchResult]:
    raw = page.evaluate(
        """() => {
            const rows = [];
            const containers = Array.from(document.querySelectorAll('div.g, div[data-sokoban-container], div.MjjYud'));
            for (const node of containers) {
                const link = node.querySelector('a:has(h3), a[jsname][href], a[href]');
                const titleEl = node.querySelector('h3');
                const snippetEl = node.querySelector('.VwiC3b, .IsZvec, [data-sncf], .yXK7lf, span');
                if (!link || !titleEl) continue;
                rows.push({
                    title: titleEl.innerText || titleEl.textContent || '',
                    url: link.href || '',
                    snippet: snippetEl ? (snippetEl.innerText || snippetEl.textContent || '') : ''
                });
            }
            return rows;
        }"""
    )
    results = []
    for item in raw or []:
        url = str(item.get("url") or "")
        results.append(SearchResult(
            rank=0,
            title=_normalize_space(str(item.get("title") or "")),
            url=url,
            snippet=_normalize_space(str(item.get("snippet") or "")),
            domain=_domain(url),
        ))
    return _dedupe_results(results, limit)


def _search_url(engine: str, query: str, market: str, setlang: str) -> str:
    encoded = quote_plus(query)
    if engine == "google":
        # pws=0 reduces personalization; num requests more candidates but Google
        # may still return fewer organic results.
        return (
            f"https://www.google.com/search?q={encoded}"
            f"&hl={quote_plus(setlang)}&gl={quote_plus(market.split('-')[-1] or 'US')}"
            f"&num=10&pws=0"
        )
    if engine == "duckduckgo":
        return f"https://html.duckduckgo.com/html/?q={encoded}&kl={quote_plus(market.lower())}"
    if engine == "bing":
        return (
            f"https://www.bing.com/search?q={encoded}"
            f"&mkt={quote_plus(market)}&setlang={quote_plus(setlang)}&cc=US"
        )
    if engine == "baidu":
        return f"https://www.baidu.com/s?wd={encoded}"
    raise ValueError(f"unsupported engine: {engine}")


def run_case(
    browser,
    engine: str,
    case: dict[str, Any],
    limit: int,
    timeout_ms: int,
    market: str,
    setlang: str,
) -> QualityReport:
    query = str(case["query"])
    started = time.perf_counter()
    page = None
    try:
        page = browser.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto(_search_url(engine, query, market, setlang), wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(800)
        text = page.locator("body").inner_text(timeout=3000)
        blocked = _looks_blocked(text)
        if engine == "google":
            results = _parse_google(page, limit)
        elif engine == "duckduckgo":
            results = _parse_duckduckgo(page, limit)
        elif engine == "baidu":
            results = _parse_baidu(page, limit)
        else:
            results = _parse_bing(page, limit)
        latency_ms = int((time.perf_counter() - started) * 1000)
        score, parts = _score(case, results, blocked)
        return QualityReport(
            engine=engine,
            query=query,
            ok=bool(results) and not blocked,
            blocked=blocked,
            latency_ms=latency_ms,
            result_count=len(results),
            unique_domain_count=len({r.domain for r in results if r.domain}),
            expected_term_hits=parts["term_hits"],
            expected_term_total=parts["term_total"],
            expected_domain_hits=parts["domain_hits"],
            expected_domain_total=parts["domain_total"],
            score=score,
            error="",
            results=results,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return QualityReport(
            engine=engine,
            query=query,
            ok=False,
            blocked=False,
            latency_ms=latency_ms,
            result_count=0,
            unique_domain_count=0,
            expected_term_hits=0,
            expected_term_total=len(case.get("expected_terms", []) or []),
            expected_domain_hits=0,
            expected_domain_total=len(case.get("expected_domains", []) or []),
            score=0.0,
            error=str(exc),
            results=[],
        )
    finally:
        if page is not None:
            page.close()


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = list(DEFAULT_CASES)
    if args.cases:
        loaded = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("--cases must point to a JSON list")
        cases = loaded
    if args.query:
        cases = [{"query": q, "expected_terms": [], "expected_domains": []} for q in args.query]
    return cases


def _write_markdown(path: Path, reports: list[QualityReport]) -> None:
    lines = [
        "# SERP Search Quality Report",
        "",
        "| Engine | Query | Score | Results | Domains | Terms | Expected Domains | Latency | Status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for report in reports:
        status = "blocked" if report.blocked else ("ok" if report.ok else f"error: {report.error[:60]}")
        terms = f"{report.expected_term_hits}/{report.expected_term_total}"
        domains = f"{report.expected_domain_hits}/{report.expected_domain_total}"
        lines.append(
            f"| {report.engine} | {report.query} | {report.score:.1f} | "
            f"{report.result_count} | {report.unique_domain_count} | {terms} | "
            f"{domains} | {report.latency_ms}ms | {status} |"
        )
    lines.append("")
    for report in reports:
        lines.append(f"## {report.engine}: {report.query}")
        if report.error:
            lines.append(f"- Error: `{report.error}`")
        if report.blocked:
            lines.append("- Blocked/captcha markers were detected.")
        for item in report.results:
            snippet = item.snippet.replace("\n", " ")
            lines.append(f"{item.rank}. [{item.title}]({item.url})")
            if snippet:
                lines.append(f"   {snippet}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test browser-based SERP search quality.")
    parser.add_argument("--engine", choices=["google", "duckduckgo", "bing", "baidu", "all"], default="duckduckgo")
    parser.add_argument("--query", action="append", help="Query to test. Can be repeated.")
    parser.add_argument("--cases", help="Path to JSON list of {query, expected_terms, expected_domains}.")
    parser.add_argument("--limit", type=int, default=5, help="Organic results per query.")
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between searches.")
    parser.add_argument("--market", default="en-US", help="Search market/region, for example en-US or zh-CN.")
    parser.add_argument("--setlang", default="en-US", help="Search UI/result language.")
    parser.add_argument("--headful", action="store_true", help="Show browser window.")
    parser.add_argument("--channel", choices=["chrome", "msedge"], help="Use an installed browser channel.")
    parser.add_argument("--executable-path", help="Path to a Chrome/Edge executable.")
    parser.add_argument("--output", default="logs/serp_quality.json")
    parser.add_argument("--markdown", default="logs/serp_quality.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = _load_cases(args)
    engines = ["google", "duckduckgo", "bing", "baidu"] if args.engine == "all" else [args.engine]

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed. Run: pip install playwright", file=sys.stderr)
        return 2

    reports: list[QualityReport] = []
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
            locale="zh-CN",
        )
        for engine in engines:
            for case in cases:
                report = run_case(
                    context,
                    engine,
                    case,
                    args.limit,
                    args.timeout_ms,
                    args.market,
                    args.setlang,
                )
                reports.append(report)
                status = "BLOCKED" if report.blocked else ("OK" if report.ok else "FAIL")
                print(
                    f"[{status}] {engine} | {report.query} | "
                    f"score={report.score:.1f} results={report.result_count} "
                    f"domains={report.unique_domain_count} latency={report.latency_ms}ms"
                )
                if args.delay > 0:
                    time.sleep(args.delay)
        browser.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([asdict(r) for r in reports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(Path(args.markdown), reports)

    successful = [r for r in reports if r.ok]
    avg_score = sum(r.score for r in reports) / len(reports) if reports else 0
    print(f"\nWrote JSON: {output}")
    print(f"Wrote Markdown: {args.markdown}")
    print(f"Successful: {len(successful)}/{len(reports)} | average score: {avg_score:.1f}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
