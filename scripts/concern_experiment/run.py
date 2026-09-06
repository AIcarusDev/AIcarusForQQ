"""Offline Concern evaluation. Reads existing cognition; never imports the app.

python -B scripts/concern_experiment/run.py export --out tmp/concern_experiment/samples.json
python -B scripts/concern_experiment/run.py run --samples ... --out ... --batches ...
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit
from xml.sax.saxutils import escape, quoteattr

import httpx
from dotenv import dotenv_values
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = "deepseek-v4-flash"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def export_samples(out: Path, limit: int) -> None:
    """Replay complete historical extraction batches, never regroup partial ones."""
    conn = sqlite3.connect((ROOT / "data/AICQ.db").as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        groups = conn.execute(
            "SELECT origin_id, created_at, count(*) AS n FROM CognitionSources "
            "WHERE origin_type='flow' GROUP BY origin_id, created_at "
            "HAVING count(*)=5 ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        batches = []
        for i, group in enumerate(groups):
            rows = conn.execute(
                "SELECT source_uid,prompt_source_id,source_timestamp,cognition_text "
                "FROM CognitionSources WHERE origin_type='flow' AND origin_id=? "
                "AND created_at=? ORDER BY source_seq", (group["origin_id"], group["created_at"])
            ).fetchall()
            batches.append({
                "id": f"real_{i + 1:02d}", "kind": "real",
                "origin_id": group["origin_id"], "created_at": group["created_at"],
                "cognitions": [{"id": r["prompt_source_id"], "timestamp": r["source_timestamp"],
                                "text": r["cognition_text"], "source_uid": r["source_uid"]} for r in rows],
            })
    finally:
        conn.close()
    write_json(out, {"exported_at": datetime.now(timezone.utc).isoformat(),
                     "source": "read-only CognitionSources historical extraction batches", "batches": batches})
    for b in batches:
        print(b["id"], b["origin_id"], b["cognitions"][0]["timestamp"],
              sum(len(c["text"]) for c in b["cognitions"]), flush=True)


def task_xml(batch: dict) -> str:
    rows = batch["cognitions"]
    if len(rows) != 5 or len({r["id"] for r in rows}) != 5:
        raise ValueError("Each batch must contain exactly five distinct cognition IDs")
    lines = ["<task>"]
    for row in rows:
        if not row["text"].strip():
            raise ValueError("Empty cognition")
        lines.extend([f'<cognition id={quoteattr(row["id"])} timestamp={quoteattr(row["timestamp"])}>',
                      escape(row["text"]), "</cognition>"])
    return "\n".join([*lines, "</task>"])


def credentials() -> tuple[str, str]:
    config = yaml.safe_load((ROOT / "config/config_user.yaml").read_text(encoding="utf-8-sig"))
    providers = config.get("model_providers", {})
    values = providers.values() if isinstance(providers, dict) else providers
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    for provider in values:
        base = str(provider.get("base_url", "")).rstrip("/")
        parsed = urlsplit(base)
        if parsed.scheme != "https" or parsed.hostname != "api.deepseek.com":
            continue
        key = env.get(provider.get("api_key_env", "")) or provider.get("api_key")
        if key:
            return base, str(key)
    raise ValueError("No configured official DeepSeek provider with a local key")


def memory_prompt() -> str:
    # Read only the literal: importing the memory package can import app state.
    tree = ast.parse((ROOT / "src/memory/event_extraction/prompt.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EVENT_EXTRACTION_SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Memory prompt literal not found")


def validate_concerns(parsed: object, valid_ids: set[str], *, sources: dict[str, str] | None = None) -> list[str]:
    errors = []
    if not isinstance(parsed, dict) or set(parsed) != {"concerns"} or not isinstance(parsed["concerns"], list):
        return ["Expected object with only a concerns array"]
    for i, item in enumerate(parsed["concerns"]):
        if not isinstance(item, dict) or set(item) != {"question", "context", "source_ids"}:
            errors.append(f"item {i}: invalid fields")
            continue
        for field in ("question", "context"):
            if not isinstance(item[field], str) or not item[field].strip():
                errors.append(f"item {i}: empty or nonstring {field}")
        ids = item["source_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or x not in valid_ids for x in ids):
            errors.append(f"item {i}: invalid source IDs")
        elif len(ids) != len(set(ids)):
            errors.append(f"item {i}: duplicate source IDs")
        elif sources is not None and isinstance(item["context"], str) and not any(
            item["context"] in sources.get(source_id, "") for source_id in ids
        ):
            errors.append(f"item {i}: context is not a verbatim excerpt of a cited cognition")
    return errors


async def run(args: argparse.Namespace) -> None:
    data = json.loads(args.samples.read_text(encoding="utf-8"))
    selected = set(args.batches.split(",")) if args.batches else {b["id"] for b in data["batches"]}
    batches = [b for b in data["batches"] if b["id"] in selected]
    if {b["id"] for b in batches} != selected:
        raise ValueError("Unknown batch ID")
    prompts = {"concern": args.prompt.read_text(encoding="utf-8")}
    if args.with_memory:
        prompts["memory"] = memory_prompt()
    tasks = {b["id"]: task_xml(b) for b in batches}
    base, key = credentials()
    args.out.mkdir(parents=True, exist_ok=False)
    for kind, prompt in prompts.items():
        (args.out / f"{kind}.prompt.md").write_text(prompt, encoding="utf-8")
    write_json(args.out / "samples.json", {**data, "batches": batches})
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "endpoint": base + "/chat/completions", "mode": "offline_add_only_replay",
        "paired_memory": args.with_memory, "batch_size": 5, "thinking": args.concern_thinking, "reasoning_effort": args.reasoning_effort,
        "memory_thinking": args.memory_thinking,
        "response_format": args.response_format,
        "verbatim_context_check": args.verbatim_context,
        "max_tokens": 16000, "repeats": args.repeats,
        "prompt_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in prompts.items()},
        "batch_ids": [b["id"] for b in batches],
    }
    write_json(args.out / "manifest.json", manifest)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20), follow_redirects=False) as client:
        async def one(batch: dict, kind: str, repeat: int) -> dict:
            name = f'{batch["id"]}.{kind}.r{repeat}'
            payload = {"model": MODEL, "messages": [{"role": "system", "content": prompts[kind]},
                       {"role": "user", "content": tasks[batch["id"]]}], "max_tokens": 16000,
                       "thinking": {"type": args.concern_thinking if kind == "concern" else args.memory_thinking},
                       "reasoning_effort": args.reasoning_effort, "stream": False}
            if kind == "concern":
                payload["response_format"] = {"type": args.response_format}
            write_json(args.out / f"{name}.request.json", payload)
            async with semaphore:
                started = time.perf_counter()
                result = {"name": name, "batch_id": batch["id"], "kind": kind,
                          "started_at": datetime.now(timezone.utc).isoformat()}
                try:
                    response = await client.post(base + "/chat/completions", headers={"Authorization": "Bearer " + key}, json=payload)
                    result["elapsed_s"] = round(time.perf_counter() - started, 3)
                    result["http_status"] = response.status_code
                    if response.status_code != 200:
                        # Do not write arbitrary error bodies or request headers containing secrets.
                        result["errors"] = [f"HTTP {response.status_code}"]
                    else:
                        raw = response.json()
                        write_json(args.out / f"{name}.response.json", raw)
                        choice = raw["choices"][0]
                        content = choice["message"].get("content") or ""
                        result.update({"model": raw.get("model"), "usage": raw.get("usage"),
                                       "finish_reason": choice.get("finish_reason"), "errors": []})
                        if choice.get("finish_reason") != "stop":
                            result["errors"].append("Incomplete response")
                        if kind == "concern":
                            if not content.strip():
                                raise ValueError("Empty model content")
                            parsed = json.loads(content)
                            sources = {r["id"]: r["text"] for r in batch["cognitions"]}
                            result["errors"].extend(validate_concerns(parsed, set(sources), sources=sources if args.verbatim_context else None))
                            write_json(args.out / f"{name}.output.json", parsed)
                            result["item_count"] = len(parsed.get("concerns", [])) if isinstance(parsed, dict) else None
                        else:
                            (args.out / f"{name}.output.txt").write_text(content, encoding="utf-8")
                            events = [json.loads(s) for s in re.findall(r"<event>\s*(.*?)\s*</event>", content, re.DOTALL)]
                            result["item_count"] = len(events)
                            result["validation_scope"] = "event JSON parse only; no storage or structuring"
                except Exception as exc:
                    result["errors"] = [type(exc).__name__]
                    result["elapsed_s"] = round(time.perf_counter() - started, 3)
                write_json(args.out / f"{name}.result.json", result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return result

        results = await asyncio.gather(*(one(b, kind, repeat) for b in batches
                                        for repeat in range(1, args.repeats + 1) for kind in prompts))
    write_json(args.out / "results.json", results)
    if any(r["errors"] for r in results):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--limit", type=int, default=12)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--samples", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--prompt", type=Path, default=HERE / "prompt.md")
    evaluate.add_argument("--batches")
    evaluate.add_argument("--with-memory", action="store_true")
    evaluate.add_argument("--concern-thinking", choices=("enabled", "disabled"), default="enabled")
    evaluate.add_argument("--memory-thinking", choices=("enabled", "disabled"), default="disabled")
    evaluate.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="high")
    evaluate.add_argument("--response-format", choices=("json_object", "text"), default="json_object")
    evaluate.add_argument("--verbatim-context", action="store_true")
    evaluate.add_argument("--repeats", type=int, default=1)
    evaluate.add_argument("--concurrency", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.command == "export":
        export_samples(args.out, args.limit)
    else:
        if args.repeats < 1:
            parser.error("repeats must be positive")
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
