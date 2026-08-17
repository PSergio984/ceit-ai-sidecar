"""Golden-set evaluation runner: precision@k / recall@k / F1 + negative pass rate.

Ported from rag-search-engine's evaluation_cli.py, matching on doc ids instead
of titles (ids are stable; titles collide). Negatives are scored separately as
negative_pass_rate — a negative case passes when ZERO relevant ids are
retrieved.

The multi-approach comparison (deliverable D2) scores every retrieval approach
("hybrid", "bm25", "semantic") on the same golden set through the same seam
(``HybridSearch.rrf_search(method=...)``) and reports which approach wins.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .config import settings
from .search import HybridSearch

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"

# Approaches compared by the multi-approach evaluation, best first.
METHODS = ("hybrid", "bm25", "semantic")


def load_golden(path: Path = GOLDEN_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(engine: HybridSearch, case: dict, limit: int, method: str = "hybrid") -> dict:
    relevant = set(case.get("relevant_docs", []))
    filters = case.get("filters") or {}
    corpus = case.get("corpus")

    retrieved = engine.rrf_search(
        case["query"],
        k=60,
        limit=limit,
        filters=filters,
        corpus=corpus,
        method=method,
    )
    retrieved_ids = {r["id"] for r in retrieved}
    top1_id = retrieved[0]["id"] if retrieved else None

    if case.get("negative"):
        passed = len(relevant & retrieved_ids) == 0
        return {
            "query": case["query"],
            "negative": True,
            "passed": passed,
            "retrieved": sorted(retrieved_ids),
        }

    hits = len(relevant & retrieved_ids)
    precision = hits / limit if limit else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "query": case["query"],
        "negative": False,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "hits": hits,
        "top1_hit": bool(top1_id and top1_id in relevant),
        "retrieved": sorted(retrieved_ids),
        "relevant": sorted(relevant),
    }


def category_of(case: dict) -> str:
    q = case["query"].lower()
    if case.get("negative"):
        return "negative"
    if q.startswith("ceit-"):
        return "catalog_code"
    if (
        "?" in case["query"]
        or "ba" in q.split()
        or "ano" in q.split()
        or "yung" in q.split()
        or "ko" in q.split()
    ):
        return "taglish"
    if "paper" in q and "by" in q:
        return "people"
    if case.get("corpus") == "policy":
        return "policy"
    return "exact_title" if len(case.get("relevant_docs", [])) == 1 else "paraphrase"


def aggregate(results: list[dict]) -> dict:
    """Per-method aggregates over `evaluate_case` results (hand-testable pure fn).

    Negatives pass when they retrieved zero relevant ids; positives contribute
    precision/recall/F1 and a top-1 hit flag. Averages are over positives only.
    """
    negatives = [r for r in results if r.get("negative")]
    positives = [r for r in results if not r.get("negative")]

    neg_pass = (
        round(sum(1 for r in negatives if r["passed"]) / len(negatives), 4) if negatives else None
    )
    top1 = (
        round(sum(1 for r in positives if r["top1_hit"]) / len(positives), 4) if positives else None
    )

    def _avg(key: str):
        return round(sum(r[key] for r in positives) / len(positives), 4) if positives else None

    return {
        "count": len(results),
        "negative_pass_rate": neg_pass,
        "top1_rate": top1,
        "avg_precision": _avg("precision"),
        "avg_recall": _avg("recall"),
        "avg_f1": _avg("f1"),
    }


def compare_methods(results_by_method: dict[str, list[dict]]) -> dict:
    """Build the multi-approach report and pick the winner.

    Winner rule (documented, stable): for a library assistant the primary
    quality gates are (1) top-1 rate — the right document surfaces first — and
    (2) negative-pass rate — no irrelevant results for "nothing here"
    queries; positive-case F1@k breaks ties. This surfaces hybrid's
    exact-match advantage (code pin + fusion) while still exposing F1@k for
    every approach in the report. Methods are reported in METHODS order
    regardless of dict insertion order.
    """
    methods = {m: aggregate(results_by_method.get(m, [])) for m in METHODS}

    def _key(m: str) -> tuple:
        a = methods[m]
        return (
            a["top1_rate"] if a["top1_rate"] is not None else -1.0,
            a["negative_pass_rate"] if a["negative_pass_rate"] is not None else -1.0,
            a["avg_f1"] if a["avg_f1"] is not None else -1.0,
        )

    return {"methods": methods, "winner": max(METHODS, key=_key)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Golden-set evaluation for the CEIT hybrid search sidecar"
    )
    parser.add_argument("--limit", type=int, default=5, help="k for precision@k/recall@k")
    parser.add_argument("--corpus", choices=["catalog", "policy", "all"], default="all")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="retrieval approaches to compare (default: all)",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    args = parser.parse_args()

    golden = load_golden()
    cases = golden["test_cases"]
    if args.corpus != "all":
        cases = [c for c in cases if (c.get("corpus") or "all") == args.corpus]

    engine = HybridSearch(Path(settings.cache_dir), engine_model())

    # Evaluate every case once per requested approach through the same seam.
    results_by_method = {
        method: [evaluate_case(engine, c, args.limit, method) for c in cases]
        for method in args.methods
    }

    comparison = compare_methods(results_by_method)

    # Detailed report from the production (hybrid) run — or the winner when
    # --methods excluded hybrid (no KeyError on partial method sets).
    report_method = "hybrid" if "hybrid" in results_by_method else comparison["winner"]
    results = results_by_method[report_method]
    agg = aggregate(results)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for case, result in zip(cases, results):
        if not result["negative"]:
            by_cat[category_of(case)].append(result)

    report = {
        "catalog_snapshot": golden.get("catalog_snapshot"),
        "limit": args.limit,
        "total_cases": len(results),
        "negative_pass_rate": agg["negative_pass_rate"],
        "top1_rate": agg["top1_rate"],
        "overall": {
            "avg_precision": agg["avg_precision"],
            "avg_recall": agg["avg_recall"],
            "avg_f1": agg["avg_f1"],
        },
        "by_category": {
            cat: {
                "count": len(items),
                "avg_precision": round(sum(r["precision"] for r in items) / len(items), 4),
                "avg_recall": round(sum(r["recall"] for r in items) / len(items), 4),
                "avg_f1": round(sum(r["f1"] for r in items) / len(items), 4),
                "top1_rate": round(sum(1 for r in items if r["top1_hit"]) / len(items), 4),
            }
            for cat, items in sorted(by_cat.items())
        },
        "cases": results,
        "approach_comparison": comparison,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(
        f"Golden set: {golden.get('catalog_snapshot')} | limit={args.limit} | {len(results)} cases"
    )
    for result in results:
        if result["negative"]:
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  [neg] {status}  {result['query'][:60]}")
        else:
            print(
                f"  p={result['precision']:.2f} r={result['recall']:.2f} f1={result['f1']:.2f}  {result['query'][:60]}"
            )
    print(f"Negative pass rate: {report['negative_pass_rate']}")
    print(f"Top-1 rate: {report['top1_rate']}")
    print(
        f"Overall avg: P@{args.limit}={report['overall']['avg_precision']} "
        f"R@{args.limit}={report['overall']['avg_recall']} F1={report['overall']['avg_f1']}"
    )
    for cat, stats in report["by_category"].items():
        print(
            f"  {cat}: n={stats['count']} P@{args.limit}={stats['avg_precision']} "
            f"top1={stats['top1_rate']} F1={stats['avg_f1']}"
        )

    # Multi-approach comparison table.
    print("\nApproach comparison (same golden set, same seam):")
    print(f"  {'approach':<10} {'P@k':>6} {'R@k':>6} {'F1':>6} {'top-1':>6} {'neg-pass':>9}")
    for method in METHODS:
        agg = comparison["methods"][method]
        marker = "  <= best" if method == comparison["winner"] else ""
        print(
            f"  {method:<10} "
            f"{agg['avg_precision'] if agg['avg_precision'] is not None else '-':>6} "
            f"{agg['avg_recall'] if agg['avg_recall'] is not None else '-':>6} "
            f"{agg['avg_f1'] if agg['avg_f1'] is not None else '-':>6} "
            f"{agg['top1_rate'] if agg['top1_rate'] is not None else '-':>6} "
            f"{agg['negative_pass_rate'] if agg['negative_pass_rate'] is not None else '-':>9}"
            f"{marker}"
        )
    print(f"Winner: {comparison['winner']}")

    return 0


def engine_model() -> str:
    from .config import settings

    return settings.model_name


if __name__ == "__main__":
    sys.exit(main())
