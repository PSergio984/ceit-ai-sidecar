"""Offline recommender evaluation on an exported borrow-sequences file.

Leave-last-out Hit@K / MRR for three approaches — seed-based recommend,
borrow-count popularity, and title-as-query hybrid search (the old
Similar-books mechanism) — against the committed index artifacts.

    uv run python scripts/eval_recommender.py --sequences export.json \
        [--index-dir index] [--limit 10]

The real embedding model is loaded for the title-as-query baseline only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SIDECAR_TOKEN", "offline-eval")

from app.rec_eval import LIST_SIZE, evaluate, format_report, load_sequences
from app.search import HybridSearch, _version_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sequences", required=True, type=Path, help="borrow-sequences JSON export"
    )
    parser.add_argument("--index-dir", type=Path, default=ROOT / "index")
    parser.add_argument("--limit", type=int, default=10, help="Hit@K depth (default 10)")
    args = parser.parse_args()

    try:
        sequences = load_sequences(args.sequences)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    version, docs_by_id, vectors = _version_artifacts(args.index_dir)
    if version == 0 or not docs_by_id:
        print(f"error: no index artifacts under {args.index_dir}", file=sys.stderr)
        return 2

    state = json.loads((args.index_dir / "state.json").read_text(encoding="utf-8"))
    engine = HybridSearch(args.index_dir, model_name=state.get("model_name", "all-MiniLM-L6-v2"))

    def title_ranker(query: str) -> list[str]:
        return [
            r["id"]
            for r in engine.rrf_search(query, method="hybrid", limit=LIST_SIZE, corpus="catalog")
        ]

    outcome = evaluate(docs_by_id, vectors, sequences, title_ranker, hit_at=args.limit)
    engine.close()
    print(format_report(outcome, str(args.sequences)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
