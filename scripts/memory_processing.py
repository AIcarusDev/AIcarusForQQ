"""Memory processing and maintenance utility.

Usage:
    python scripts/memory_processing.py preprocess --limit 5000
    python scripts/memory_processing.py consolidate-candidate-storylines --solidify
    python scripts/memory_processing.py synthesize-storylines
    python scripts/memory_processing.py maintain --solidify
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _open_db(path: str | None) -> sqlite3.Connection:
    if path:
        db_path = Path(path)
    else:
        import database

        db_path = Path(database.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON")
    return con


def main() -> int:
    parser = argparse.ArgumentParser(description="Run memory processing and maintenance workflows")
    parser.add_argument("--db", default="", help="SQLite DB path; defaults to data/AICQ.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    preprocess = sub.add_parser("preprocess", help="build deterministic preprocessing caches")
    preprocess.add_argument("--limit", type=int, default=5000)
    preprocess.add_argument("--raw-entities", action="store_true", help="do not canonicalize roles before relation/storyline builds")
    preprocess.add_argument("--algorithmic-storylines", action="store_true")

    candidates = sub.add_parser(
        "consolidate-candidate-storylines",
        help="solidify pending candidate storylines",
    )
    candidates.add_argument("--max-candidates", type=int, default=100)
    candidates.add_argument("--dry-run", action="store_true", help="preview without writing")
    candidates.add_argument("--solidify", action="store_true", help="write valid candidate storylines")

    summaries = sub.add_parser("synthesize-storylines", help="create or refresh recall-ready storylines")
    summaries.add_argument("--max-inputs", type=int, default=32)
    summaries.add_argument("--storyline-id", action="append", default=[])

    maintain = sub.add_parser("maintain", help="run one bounded maintenance pass")
    maintain.add_argument("--solidify", action="store_true", help="write valid candidate storylines")
    maintain.add_argument("--dry-run", action="store_true", help="force preview mode")
    maintain.add_argument("--max-candidates", type=int, default=100)
    maintain.add_argument("--algorithmic-storylines", action="store_true")

    args = parser.parse_args()

    if args.cmd == "maintain":
        from memory.maintenance.workflow import run_memory_maintenance

        stats = run_memory_maintenance(
            args.db or None,
            trigger="script.maintain",
            config={
                "memory": {
                    "processing": {
                        "dry_run": bool(args.dry_run or not args.solidify),
                        "solidify": bool(args.solidify),
                        "max_candidate_storylines_per_maintenance": int(args.max_candidates),
                        "algorithmic_storyline_enabled": bool(args.algorithmic_storylines),
                    }
                }
            },
        )
    else:
        with _open_db(args.db or None) as con:
            if args.cmd == "preprocess":
                from memory.maintenance.preprocessing import run_preprocessing

                stats = run_preprocessing(
                    con,
                    limit=args.limit,
                    trigger="script",
                    canonical_entities=not args.raw_entities,
                    algorithmic_storyline_enabled=bool(args.algorithmic_storylines),
                )
                con.commit()
            elif args.cmd == "consolidate-candidate-storylines":
                from memory.maintenance.preprocessing import run_candidate_storyline_consolidation

                stats = run_candidate_storyline_consolidation(
                    con,
                    max_candidate_storylines=args.max_candidates,
                    dry_run=bool(args.dry_run or not args.solidify),
                    solidify=bool(args.solidify),
                )
                if args.solidify and not args.dry_run:
                    con.commit()
            else:
                from memory.storyline_synthesis.workflow import run_storyline_synthesis

                stats = run_storyline_synthesis(
                    con,
                    max_inputs=args.max_inputs,
                    storyline_ids=args.storyline_id,
                )
                con.commit()
    payload = stats.to_dict() if hasattr(stats, "to_dict") else stats
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
