"""Memory V2 consolidation maintenance utility.

Usage:
    python scripts/memory_consolidation.py preprocess --limit 5000
    python scripts/memory_consolidation.py consolidate-mounts --max-mounts 100
    python scripts/memory_consolidation.py consolidate-mounts --solidify --max-mounts 100
    python scripts/memory_consolidation.py refresh-summaries
    python scripts/memory_consolidation.py sleep --solidify
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
    parser = argparse.ArgumentParser(description="Maintain Memory V2 consolidation tables")
    parser.add_argument("--db", default="", help="SQLite DB path; defaults to data/AICQ.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    preprocess = sub.add_parser("preprocess", help="build deterministic preprocessing caches")
    preprocess.add_argument("--limit", type=int, default=5000)
    preprocess.add_argument("--raw-entities", action="store_true", help="do not canonicalize roles before relation/cluster builds")

    mounts = sub.add_parser("consolidate-mounts", help="solidify pending memory mounts")
    mounts.add_argument("--max-mounts", type=int, default=100)
    mounts.add_argument("--accept-threshold", type=float, default=0.62)
    mounts.add_argument("--dry-run", action="store_true", help="preview decisions without writing")
    mounts.add_argument("--solidify", action="store_true", help="write accepted decisions to consolidation tables")

    summaries = sub.add_parser("refresh-summaries", help="bootstrap/refresh ready cluster summaries")
    summaries.add_argument("--max-inputs", type=int, default=32)
    summaries.add_argument("--max-bootstrap-clusters", type=int, default=64)

    sleep = sub.add_parser("sleep", help="run one sleep-time maintenance pass")
    sleep.add_argument("--solidify", action="store_true", help="write accepted mount decisions")
    sleep.add_argument("--dry-run", action="store_true", help="force mount-consolidation preview mode")
    sleep.add_argument("--max-mounts", type=int, default=100)
    sleep.add_argument("--accept-threshold", type=float, default=0.62)

    args = parser.parse_args()

    if args.cmd == "sleep":
        from memory.sleep_maintenance import run_sleep_memory_maintenance

        stats = run_sleep_memory_maintenance(
            args.db or None,
            trigger="script.sleep",
            config={
                "memory": {
                    "consolidation": {
                        "dry_run": bool(args.dry_run or not args.solidify),
                        "solidify": bool(args.solidify),
                        "max_mounts_per_sleep": int(args.max_mounts),
                        "accept_threshold": float(args.accept_threshold),
                    }
                }
            },
        )
    else:
        with _open_db(args.db or None) as con:
            if args.cmd == "preprocess":
                from memory.consolidation import run_preprocessing

                stats = run_preprocessing(
                    con,
                    limit=args.limit,
                    trigger="script",
                    canonical_entities=not args.raw_entities,
                )
                con.commit()
            elif args.cmd == "consolidate-mounts":
                from memory.consolidation import run_mount_consolidation

                stats = run_mount_consolidation(
                    con,
                    max_mounts=args.max_mounts,
                    dry_run=bool(args.dry_run or not args.solidify),
                    solidify=bool(args.solidify),
                    accept_threshold=float(args.accept_threshold),
                )
                if args.solidify and not args.dry_run:
                    con.commit()
            else:
                from memory.summary_worker import run_summary_refresh_worker

                stats = run_summary_refresh_worker(
                    con,
                    max_inputs=args.max_inputs,
                    max_bootstrap_clusters=args.max_bootstrap_clusters,
                )
                con.commit()
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
