"""Golden-set evaluation runner: precision@k / recall@k / F1 + negative pass rate.

Ported from rag-search-engine's evaluation_cli.py, matching on doc ids instead
of titles (ids are stable; titles collide). Negatives are scored separately as
negative_pass_rate — a negative case passes when ZERO relevant ids are
retrieved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .search import HybridSearch

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"


def load_golden(path: Path = GOLDEN_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def evaluate_case(engine: HybridSearch, case: dict, limit: int) -> dict:
    relevant = set(case.get("relevant_docs", []))
    filters = case.get("filters") or {}
    corpus = case.get("corpus")

    retrieved = engine.rrf_search(
        case["query"],
        k=60,
        limit=limit,
        filters=filters,
        corpus=corpus,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Golden-set evaluation for the CEIT hybrid search sidecar"
    )
    parser.add_argument("--limit", type=int, default=5, help="k for precision@k/recall@k")
    parser.add_argument("--corpus", choices=["catalog", "policy", "all"], default="all")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    args = parser.parse_args()

    golden = load_golden()
    cases = golden["test_cases"]
    if args.corpus != "all":
        cases = [c for c in cases if (c.get("corpus") or "all") == args.corpus]

    engine = HybridSearch(Path("cache"), engine_model())

    results = [evaluate_case(engine, c, args.limit) for c in cases]

    negatives = [r for r in results if r["negative"]]
    positives = [r for r in results if not r["negative"]]

    neg_pass_rate = sum(1 for r in negatives if r["passed"]) / len(negatives) if negatives else None

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for case, result in zip(cases, results):
        if not result["negative"]:
            by_cat[category_of(case)].append(result)

    report = {
        "catalog_snapshot": golden.get("catalog_snapshot"),
        "limit": args.limit,
        "total_cases": len(results),
        "negative_pass_rate": neg_pass_rate,
        "top1_rate": round(sum(1 for r in positives if r["top1_hit"]) / len(positives), 4)
        if positives
        else None,
        "overall": {
            "avg_precision": round(sum(r["precision"] for r in positives) / len(positives), 4)
            if positives
            else None,
            "avg_recall": round(sum(r["recall"] for r in positives) / len(positives), 4)
            if positives
            else None,
            "avg_f1": round(sum(r["f1"] for r in positives) / len(positives), 4)
            if positives
            else None,
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

    return 0


def engine_model() -> str:
    from .config import settings

    return settings.model_name


if __name__ == "__main__":
    sys.exit(main())
