"""Multi-approach eval comparison: per-method aggregates + winner selection (D2).

Seam: the pure functions `aggregate()` and `compare_methods()` in `app.eval`
compute the comparison report from per-method `evaluate_case` results. Expected
values are hand-computed literals, not recomputed the way the code does.
"""

from __future__ import annotations

from app.eval import METHODS, RRF_K, aggregate, compare_methods


def _pos(precision: float, recall: float, f1: float, top1: bool) -> dict:
    return {
        "negative": False,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top1_hit": top1,
    }


def _neg(passed: bool) -> dict:
    return {"negative": True, "passed": passed}


def test_methods_constant_has_hybrid_bm25_semantic():
    assert METHODS == ("hybrid", "bm25", "semantic")


def test_aggregate_averages_hand_computed_literals():
    results = [
        _pos(0.5, 1.0, 0.6667, top1=True),
        _pos(0.25, 0.5, 0.3333, top1=False),
    ]
    agg = aggregate(results)

    assert agg["avg_precision"] == round(0.375, 4)
    assert agg["avg_recall"] == round(0.75, 4)
    assert agg["avg_f1"] == round(0.5, 4)
    assert agg["top1_rate"] == 0.5
    assert agg["count"] == 2


def test_aggregate_negative_pass_rate_and_top1_mix():
    results = [
        _neg(passed=True),
        _neg(passed=False),
        _pos(0.2, 0.8, 0.32, top1=True),
    ]
    agg = aggregate(results)

    assert agg["negative_pass_rate"] == 0.5
    assert agg["top1_rate"] == 1.0


def test_aggregate_handles_no_negatives_or_no_positives():
    all_neg = [_neg(passed=True), _neg(passed=True)]
    agg = aggregate(all_neg)
    assert agg["top1_rate"] is None
    assert agg["avg_f1"] is None
    assert agg["negative_pass_rate"] == 1.0


def test_compare_methods_winner_is_top1_then_negpass_then_f1():
    # hybrid: highest top-1 -> wins even though bm25 has higher F1.
    results = {
        "hybrid": [_pos(0.4, 0.6, 0.48, top1=True)] * 4,
        "bm25": [_pos(0.8, 1.0, 0.8889, top1=False)] * 4,
        "semantic": [_pos(0.5, 0.7, 0.5833, top1=False)] * 4,
    }
    report = compare_methods(results)

    assert report["winner"] == "hybrid"
    assert report["methods"]["bm25"]["avg_f1"] > report["methods"]["hybrid"]["avg_f1"]
    assert set(report["methods"]) == set(METHODS)


def test_compare_methods_ties_break_on_negative_pass_then_f1():
    # Same top-1: the method that never fails negatives wins; else higher F1.
    results = {
        "hybrid": [_pos(0.5, 1.0, 0.6667, top1=True)] * 2 + [_neg(False)],
        "bm25": [_pos(0.5, 1.0, 0.6667, top1=True)] * 2 + [_neg(True)],
        "semantic": [_pos(0.3, 0.6, 0.4, top1=True)] * 2 + [_neg(True)],
    }
    report = compare_methods(results)
    assert report["winner"] == "bm25"


def test_compare_methods_ties_break_on_f1_when_gates_equal():
    results = {
        "hybrid": [_pos(0.5, 1.0, 0.6667, top1=True)] * 2,
        "bm25": [_pos(0.6, 0.8, 0.6857, top1=True)] * 2,
        "semantic": [_pos(0.4, 0.5, 0.4444, top1=True)] * 2,
    }
    report = compare_methods(results)
    assert report["winner"] == "bm25"


def test_compare_methods_returns_methods_in_constant_order():
    results = {
        "semantic": [_pos(0.5, 1.0, 0.6667, top1=True)],
        "hybrid": [_pos(0.5, 1.0, 0.6667, top1=True)],
        "bm25": [],
    }
    report = compare_methods(results)
    assert list(report["methods"]) == ["hybrid", "bm25", "semantic"]


def test_compare_methods_reports_extra_methods_after_the_trio():
    """Variant methods (e.g. hybrid+rerank) are reported after the canonical trio."""
    results = {
        "hybrid": [_pos(0.5, 1.0, 0.6667, top1=True)],
        "bm25": [],
        "semantic": [],
        "hybrid+rerank": [_pos(0.8, 1.0, 0.8889, top1=True)],
    }
    report = compare_methods(results)
    assert list(report["methods"]) == ["hybrid", "bm25", "semantic", "hybrid+rerank"]
    assert report["winner"] == "hybrid+rerank"


class _FakeEngine:
    """rrf_search returns the given docs verbatim (no real index needed)."""

    def __init__(self, docs: list[dict]):
        self.docs = docs

    def rrf_search(self, query, k=RRF_K, limit=10, filters=None, corpus=None, method="hybrid"):
        return self.docs[:limit]


def _retrieval_doc(doc_id: str, bm25=None, sem=None) -> dict:
    return {
        "id": doc_id,
        "corpus": "catalog",
        "title": doc_id,
        "score": 0.5,
        "bm25_rank": bm25,
        "semantic_rank": sem,
        "metadata": {},
    }


def test_evaluate_case_rerank_moves_bm25_only_relevant_doc_to_rank_one():
    """Blend re-ranking promotes the bm25-only relevant doc above semantic-only
    noise, flipping top-1 without changing the retrieved doc set (hand-computed
    blend tiers: both-retrievers < bm25-only < semantic-only)."""
    from app.eval import evaluate_case

    engine = _FakeEngine(
        [
            _retrieval_doc("n1", bm25=None, sem=2),
            _retrieval_doc("n2", bm25=None, sem=3),
            _retrieval_doc("n3", bm25=5, sem=None),
            _retrieval_doc("paper-1", bm25=4, sem=None),
        ]
    )
    case = {"query": "x", "relevant_docs": ["paper-1"], "corpus": "catalog"}

    plain = evaluate_case(engine, case, 5, "hybrid")
    reranked = evaluate_case(engine, case, 5, "hybrid", rerank=True)

    assert plain["top1_hit"] is False
    assert reranked["top1_hit"] is True
    # Same retrieved set -> same precision/recall; only the ordering changed.
    assert reranked["precision"] == plain["precision"]
    assert reranked["recall"] == plain["recall"]
