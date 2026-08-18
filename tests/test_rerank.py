"""Reranker: blend + LLM listwise re-ranking of the top-k candidates (D6b).

Seam: `Reranker.rerank(query, results)` — the public interface the /search
flow calls. Expected orders are hand-computed from the candidate ranks (the
blend key) and from the fake provider's explicit reorder (the LLM path).
Safety contract: `none` and any failure mode leave the original order intact.
"""

from __future__ import annotations

from conftest import FakeClient

from app.rerank import RERANK_PROMPT, Reranker


def _doc(doc_id: str, bm25=None, sem=None, title: str = "") -> dict:
    return {
        "id": doc_id,
        "corpus": "catalog",
        "title": title or doc_id,
        "score": 0.5,
        "bm25_rank": bm25,
        "semantic_rank": sem,
        "metadata": {},
    }


def test_none_mode_returns_same_order_unchanged():
    docs = [_doc("a", 1, 2), _doc("b", 3, 1)]
    reranker = Reranker(mode="none")
    assert [d["id"] for d in reranker.rerank("q", docs)] == ["a", "b"]


def test_blend_ranks_both_retriever_docs_first_then_tiered():
    docs = [
        _doc("only-bm25", bm25=5),
        _doc("both-b", bm25=2, sem=3),
        _doc("only-sem", sem=2),
        _doc("both-a", bm25=1, sem=1),
    ]
    reranker = Reranker(mode="blend")
    order = [d["id"] for d in reranker.rerank("q", docs)]
    # both-a (1+1=2) < both-b (2+3=5) < only-bm25 (tier1, 5) < only-sem (tier2, 2)
    assert order == ["both-a", "both-b", "only-bm25", "only-sem"]


def test_blend_tie_breaks_by_rank_sum():
    docs = [
        _doc("both-x", bm25=4, sem=4),
        _doc("both-y", bm25=1, sem=2),
    ]
    reranker = Reranker(mode="blend")
    order = [d["id"] for d in reranker.rerank("q", docs)]
    assert order == ["both-y", "both-x"]


def test_llm_mode_reorders_by_provider_list():
    docs = [_doc("p1", 1, 1), _doc("p2", 2, 2), _doc("p3", 3, 3)]
    reranker = Reranker(mode="llm", client=FakeClient("3, 1, 2"))
    order = [d["id"] for d in reranker.rerank("q", docs)]
    assert order == ["p3", "p1", "p2"]


def test_llm_mode_failure_preserves_original_order():
    docs = [_doc("p1", 1, 1), _doc("p2", 2, 2)]
    reranker = Reranker(mode="llm", client=FakeClient("", fail=True))
    order = [d["id"] for d in reranker.rerank("q", docs)]
    assert order == ["p1", "p2"]


def test_llm_mode_garbage_output_preserves_original_order():
    docs = [_doc("p1", 1, 1), _doc("p2", 2, 2)]
    reranker = Reranker(mode="llm", client=FakeClient("no numbers here"))
    order = [d["id"] for d in reranker.rerank("q", docs)]
    assert order == ["p1", "p2"]


def test_blend_keeps_pinned_doc_first():
    """The exact-code pin (D-02) is a hard rule: re-ranking must not undo it."""
    docs = [
        _doc("both-a", bm25=1, sem=1),
        _doc("pinned", bm25=4, sem=None, title=""),
    ]
    docs[1]["pinned"] = True
    reranker = Reranker(mode="blend")
    order = [d["id"] for d in reranker.rerank("q", docs)]
    assert order[0] == "pinned"


def test_llm_mode_keeps_pinned_docs_first():
    docs = [
        _doc("p1", 1, 1),
        _doc("pinned", 3, 3),
        _doc("p2", 2, 2),
    ]
    docs[1]["pinned"] = True
    reranker = Reranker(mode="llm", client=FakeClient("1, 3, 2"))
    order = [d["id"] for d in reranker.rerank("q", docs)]
    # The LLM placed p1 first, but the pinned doc must stay ahead of everything.
    assert order[0] == "pinned"


def test_rerank_prompt_asks_for_relevance_ordered_numbers():
    assert "relevance" in RERANK_PROMPT
    assert "descending" in RERANK_PROMPT
