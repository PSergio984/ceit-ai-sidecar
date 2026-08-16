"""Post-retrieval filters + code-exact pin behavior."""

from __future__ import annotations

import numpy as np

from app.ingest import load_documents
from app.rebuild import build_index
from app.search import CODE_PIN_RE, HybridSearch


def _build_index(tmp_path, corpus_path):
    cache = tmp_path / "cache"
    docs = load_documents(corpus_path)

    def embed(docs_):
        vectors = []
        for d in docs_:
            rng = np.random.RandomState(sum(ord(c) for c in d["text"]))
            v = rng.rand(16).astype(np.float32)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)

    build_index(corpus_path, "test-model", cache, embed_fn=embed)

    import app.search as search_mod

    target = np.asarray(embed([docs[0]])[0])

    class FakeQuery:
        def encode(self, texts, normalize_embeddings=True):
            return np.stack([target] * len(texts))

    search_mod.embed_query = lambda q, m: FakeQuery().encode([q])[0]
    return cache


def test_filter_department_excludes_other_departments(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search(
        "water pump", k=60, limit=10, filters={"department": "Civil Engineering"}
    )
    hs.close()

    assert results
    assert all(r["metadata"]["department"] == "Civil Engineering" for r in results)


def test_filter_paper_type_and_year_range(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search(
        "water",
        k=60,
        limit=10,
        filters={"paper_type": "Thesis", "year_from": 2020, "year_to": 2025},
    )
    hs.close()

    for r in results:
        assert r["metadata"]["paper_type"] == "Thesis"
        assert 2020 <= r["metadata"]["publication_year"] <= 2025


def test_filter_author_any_of_multiple(tmp_path, corpus_path):
    """author matches ANY element of metadata.authors, case-insensitive substring."""
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search(
        "water",
        k=60,
        limit=10,
        filters={"author": "juan"},
    )
    hs.close()

    assert results, "expected results for author juan"
    for r in results:
        assert any("juan" in name.lower() for name in (r["metadata"].get("authors") or []))
    # paper-2 carries "Juan Dela Cruz" as its SECOND author; paper-1 as its first.
    assert {r["id"] for r in results} == {"paper-1", "paper-2"}


def test_filter_author_combined_with_year_range(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search(
        "water",
        k=60,
        limit=10,
        filters={"author": "juan", "year_from": 2020, "year_to": 2025},
    )
    hs.close()

    # paper-1 (2015) is excluded by the year range; paper-2 (2025) remains.
    assert [r["id"] for r in results] == ["paper-2"]


def test_filter_adviser_research_or_technical(tmp_path, corpus_path):
    """adviser matches research_adviser OR technical_adviser, case-insensitive substring."""
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search(
        "water",
        k=60,
        limit=10,
        filters={"adviser": "engr. jose"},
    )
    hs.close()

    assert results, "expected results for adviser engr. jose"
    for r in results:
        meta = r["metadata"]
        assert any(
            "engr. jose" in (meta.get(k) or "").lower()
            for k in ("research_adviser", "technical_adviser")
        )
    # paper-1 matches via research_adviser only; paper-2 via technical_adviser only.
    assert {r["id"] for r in results} == {"paper-1", "paper-2"}


def test_filter_author_non_string_junk_does_not_raise(tmp_path, corpus_path):
    """Agent junk args must be str()-guarded — clean empty result, never a TypeError."""
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    list_results = hs.rrf_search("water", k=60, limit=10, filters={"author": ["x"]})
    int_results = hs.rrf_search("water", k=60, limit=10, filters={"adviser": 42})
    hs.close()

    assert list_results == []
    assert int_results == []


def test_filtered_doc_never_outranks_unfiltered_relevant_one(tmp_path, corpus_path):
    """Filtering happens BEFORE fusion: an out-of-filter doc is not in the
    candidate set at all, so it can never appear above a relevant one."""
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search(
        "water pump", k=60, limit=10, filters={"department": "Electrical Engineering"}
    )
    hs.close()

    assert all(r["metadata"]["department"] == "Electrical Engineering" for r in results)
    # paper-2 (EE) is the only EE doc; it must be the sole result.
    assert [r["id"] for r in results] == ["paper-2"]


def test_corpus_filter_separates_policy(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("General Information", k=60, limit=10, corpus="policy")
    hs.close()

    assert results
    assert all(r["corpus"] == "policy" for r in results)


def test_code_exact_pin_pins_catalog_code_first(tmp_path, corpus_path):
    cache = _build_index(tmp_path, corpus_path)
    hs = HybridSearch(cache, "test-model")
    results = hs.rrf_search("ceit-ee-25-01", k=60, limit=10)
    hs.close()

    assert results[0]["id"] == "paper-2"
    assert results[0]["metadata"]["catalog_code"] == "CEIT-EE-25-01"


def test_code_pin_regex_matches_real_codes():
    assert CODE_PIN_RE.match("CEIT-EE-25-01")
    assert CODE_PIN_RE.match("CEIT-CE-15-014")
    assert CODE_PIN_RE.match("ceit-it-23-07")
    assert not CODE_PIN_RE.match("water pump")
    assert not CODE_PIN_RE.match("CEIT-XX-1")
