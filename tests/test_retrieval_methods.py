"""Single-retriever vs hybrid ranking for the multi-approach eval (deliverable D2).

Seam: `HybridSearch.rrf_search(..., method=...)` returns a ranked list whose
member/ordering reflects ONLY the selected approach:
- "bm25": only docs that matched BM25, ranked by BM25 rank.
- "semantic": only docs in the semantic index, ranked by cosine similarity.
- "hybrid": RRF fusion + code pin (the production path).
"""

from __future__ import annotations

import numpy as np
from conftest import build_test_index, embed_from

from app.search import HybridSearch


def _build_index(tmp_path, corpus_path):
    cache, docs = build_test_index(tmp_path, corpus_path)

    import app.search as search_mod

    # Pin the query embedding to doc 1 so semantic ranking is deterministic.
    target = np.asarray(embed_from(docs)([docs[0]])[0])

    class FakeQuery:
        def encode(self, texts, normalize_embeddings=True):
            return np.stack([target] * len(texts))

    search_mod.embed_query = lambda q, m: FakeQuery().encode([q])[0]
    return cache


def test_bm25_only_method_excludes_semantic_only_docs(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    # "water pump" matches paper-1 and paper-2 text exactly in FTS5.
    results = hs.rrf_search("water pump", k=60, limit=10, method="bm25")
    hs.close()

    assert results, "expected BM25 matches"
    assert results[0]["id"] == "paper-1"
    for r in results:
        assert r["bm25_rank"] is not None
        assert r["semantic_rank"] is None


def test_semantic_only_method_ranks_by_cosine(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("water pump", k=60, limit=10, method="semantic")
    hs.close()

    assert results, "expected semantic matches"
    assert results[0]["id"] == "paper-1"  # query pinned to doc 1
    for r in results:
        assert r["semantic_rank"] is not None
        assert r["bm25_rank"] is None


def test_hybrid_method_keeps_both_ranks_and_rrf_score(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("water pump", k=60, limit=10, method="hybrid")
    hs.close()

    assert results, "expected hybrid matches"
    top = results[0]
    assert top["id"] == "paper-1"
    assert top["bm25_rank"] is not None
    assert top["semantic_rank"] is not None
    # RRF both-rank-one: 1/(60+1) + 1/(60+1) = 2/61.
    assert top["score"] == round(2 / 61, 4)


def test_code_pin_is_hybrid_only(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")

    hybrid = hs.rrf_search("ceit-ee-25-01", k=60, limit=10, method="hybrid")
    semantic = hs.rrf_search("ceit-ee-25-01", k=60, limit=10, method="semantic")
    hs.close()

    # Hybrid pins the exact code to rank 1; pure semantic cannot (query
    # embedding is pinned to doc 1), so the code doc is NOT first.
    assert hybrid[0]["id"] == "paper-2"
    assert hybrid[0]["metadata"]["catalog_code"] == "CEIT-EE-25-01"
    assert hybrid[0]["pinned"] is True
    assert not any(r.get("pinned") for r in semantic)
    assert semantic[0]["id"] != "paper-2"


def test_unknown_method_falls_back_to_hybrid(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("water pump", k=60, limit=10, method="bogus")
    hs.close()

    assert results
    assert results[0]["bm25_rank"] is not None
    assert results[0]["semantic_rank"] is not None
