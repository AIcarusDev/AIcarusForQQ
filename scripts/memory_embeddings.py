"""Memory embedding maintenance utility.

Usage:
    python scripts/memory_embeddings.py backfill --limit 100
    python scripts/memory_embeddings.py rebuild
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Maintain Memory embeddings")
    sub = parser.add_subparsers(dest="cmd", required=True)
    backfill = sub.add_parser("backfill", help="process pending/failed/stale embedding jobs")
    backfill.add_argument("--limit", type=int, default=100)
    sub.add_parser("rebuild", help="delete vectors, queue all current owners, and rebuild")
    args = parser.parse_args()

    from config_loader import load_config
    import app_state
    import database
    from memory.repo.events import rebuild_embeddings, run_embedding_backfill

    cfg, _docs = load_config()
    app_state.config = cfg
    await database.init_db()

    if args.cmd == "backfill":
        result = await run_embedding_backfill(limit=args.limit)
    else:
        result = await rebuild_embeddings()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

