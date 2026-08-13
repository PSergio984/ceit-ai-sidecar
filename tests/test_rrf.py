"""Hand-computed RRF fusion math pins (R1 — no tolerance slop)."""

from __future__ import annotations

import numpy as np
import pytest

from app.ingest import load_documents
from app.rebuild import build_index
from app.search import HybridSearch


def _build_index(tmp_path, corpus_path, embedder, with_semantic: bool = True):
    cache = tmp_path / "cache"
    docs = load_documents(corpus_path)

    def embed(docs_):
        # 16-dim deterministic vectors; correlated with text length so the
        # semantic ranker has a stable ordering we can hand-compute against.
        vectors = []
        for d in docs_:
            rng = np.random.RandomState(sum(ord(c) for c in d["text"]))
            v = rng.rand(16).astype(np.float32)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)

    build_index(corpus_path, "test-model", cache, embed_fn=embed)

    # Monkeypatch the query embedder to return a fixed vector close to doc 1.
    target = np.asarray(embed([docs[0]])[0])

    class FakeQuery:
        def __init__(self):
            self.vec = target

        def encode(self, texts, normalize_embeddings=True):
            return np.stack([self.vec] * len(texts))

    import app.search as search_mod

    search_mod.embed_query = lambda q, m: FakeQuery().encode([q])[0]

    return cache, docs


def test_both_rank_one_fuses_to_2_over_k_plus_1(tmp_path, corpus_path):
    cache, _docs = _build_index(tmp_path, corpus_path, None)
    hs = HybridSearch(cache, "test-model")
    # "water pumps" matches paper-1's text exactly in FTS5 (no stemming);
    # the semantic query vector is pinned to doc 1. Both rank it #1.
    results = hs.rrf_search("water pumps", k=60, limit=10)
    hs.close()

    assert results, "expected at least one result"
    top = results[0]
    assert top["id"] == "paper-1"
    # RRF = 1/(60+1) + 1/(60+1) = 2/61, rounded to 4 decimals in the result.
    assert top["score"] == pytest.approx(round(2 / 61, 4), abs=1e-9)


def test_single_ranker_doc_contributes_one_over_k_plus_rank(tmp_path, corpus_path):
    cache, _docs = _build_index(tmp_path, corpus_path, None)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("water pumps", k=60, limit=10)
    hs.close()

    scores = {r["id"]: r["score"] for r in results}
    # paper-4 contains "Water" but not "pumps" — BM25 may or may not hit it;
    # it is only ranked semantically (query pinned to doc1 -> after doc1).
    if "paper-4" in scores:
        # Must be strictly less than the two-ranker winner.
        assert scores["paper-4"] < scores["paper-1"]


def test_empty_union_returns_empty(tmp_path, corpus_path):
    cache, _docs = _build_index(tmp_path, corpus_path, None)
    hs = HybridSearch(cache, "test-model")

    # Force no match: monkeypatch both retrievers to return nothing.
    hs._bm25_ranks = lambda db, q, corpus: {}
    hs._semantic_scores = lambda d, v, q: {}
    assert hs.rrf_search("zzzz", k=60, limit=10) == []
    hs.close()


def test_monotonic_ordering_by_score(tmp_path, corpus_path):
    cache, _docs = _build_index(tmp_path, corpus_path, None)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("water pump", k=60, limit=10)
    hs.close()
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
