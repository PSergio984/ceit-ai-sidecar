"""PgVectorSearch: SearchBackend parity over the Postgres index (#7).

Contract tests run the SAME assertions on both backends — sqlite via
HybridSearch on a test index, pgvector via PgVectorSearch on a synced schema —
so the SearchBackend seam is proven equivalent, not assumed.

DB tests run only when TEST_DATABASE_URL is set (CI provides a pgvector
container). Each uses a fresh ceit_test_<hex> schema dropped on teardown.
The golden parity test loads the committed index artifacts (real 1,338-doc
corpus + real embeddings) — exact BM25 parity is asserted only when the
lakebase_text index exists; on a plain pgvector container the keyword
channel uses the ts_rank_cd @@ fallback and only semantic parity is exact.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import numpy as np
import psycopg
import pytest
from conftest import make_corpus
from psycopg import sql

from app.config import Settings
from app.pg_index import PgIndexStore
from app.pg_search import PgVectorSearch
from app.search import HybridSearch

PG_URL = os.environ.get("TEST_DATABASE_URL")
ROOT = Path(__file__).resolve().parent.parent
COMMITTED_CACHE = ROOT / "index"


def _hash_embed_fn(docs) -> np.ndarray:
    """Deterministic 384-dim text-hash embedder.

    randn (not rand): all-positive vectors share the first orthant and
    cosine-collide at ~0.75, which would falsely clear the semantic gate.
    """
    import hashlib

    vectors = []
    for doc in docs:
        seed = int.from_bytes(hashlib.sha256(doc["text"].encode()).digest()[:4])
        v = np.random.RandomState(seed).randn(384).astype(np.float32)
        vectors.append(v / np.linalg.norm(v))
    return np.asarray(vectors, dtype=np.float32)


def _query_embed_fn(query: str) -> np.ndarray:
    """Query embedder identical to _hash_embed_fn on doc text."""
    return _hash_embed_fn([{"text": query}])[0]


def _sqlite_query_embed(query: str, _model: str) -> np.ndarray:
    """Same 384-dim hash vector as _hash_embed_fn — the sqlite index below is
    built with the same embedder, so identical text gives cosine 1.0."""
    return _query_embed_fn(query)


@pytest.fixture
def pg_schema():
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"ceit_test_{secrets.token_hex(4)}"
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    yield schema
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def pg_engine(pg_schema, tmp_path):
    corpus_path = make_corpus(tmp_path)
    store = PgIndexStore(PG_URL, schema=pg_schema)
    store.sync(corpus_path, "test-model", embed_fn=_hash_embed_fn)
    engine = PgVectorSearch(
        PG_URL, schema=pg_schema, model_name="test-model", embed_query_fn=_query_embed_fn
    )
    yield engine
    engine.close()


@pytest.fixture
def sqlite_engine(tmp_path, monkeypatch):
    from app.rebuild import build_index

    corpus_path = make_corpus(tmp_path)
    cache = tmp_path / "cache"
    # 384-dim hash embedder: 16-dim random vectors cosine-collide (~0.87
    # max over 6 docs), which would falsely clear the semantic gate.
    build_index(corpus_path, "test-model", cache, embed_fn=_hash_embed_fn)
    # embed_query(query, model_name) — identical text gives cosine 1.0 on
    # both backends; an unmatched query is ~orthogonal (gate-safe).
    monkeypatch.setattr("app.search.embed_query", _sqlite_query_embed)
    engine = HybridSearch(cache, "test-model")
    yield engine
    engine.close()


@pytest.fixture(params=["sqlite", "pgvector"])
def engine(request):
    """The same contract test runs on whichever backend the param selects.

    Only the parametrized fixture is resolved — declaring both would build
    (and require) a Postgres connection for every test.
    """
    name = {"sqlite": "sqlite_engine", "pgvector": "pg_engine"}[request.param]
    return request.getfixturevalue(name)


# ---------------------------------------------------------------------------
# Contract: the same assertions must hold on BOTH backends
# ---------------------------------------------------------------------------


def test_bm25_method_ranks_by_keyword(engine):
    results = engine.rrf_search("groundwater pumps", limit=5, method="bm25")
    ids = [r["id"] for r in results]
    assert ids and ids[0] == "paper-1"
    assert all(r["bm25_rank"] is not None and r["semantic_rank"] is None for r in results)


def test_hybrid_exact_code_pins_to_top(engine):
    results = engine.rrf_search("CEIT-IT-23-07", limit=5, method="hybrid")
    assert results[0]["id"] == "paper-3"
    assert results[0]["pinned"] is True


# "ceit" appears in every catalog doc's text (catalog_code) — the whole
# catalog is the BM25 candidate set, so filters alone decide the survivors.
def test_department_filter(engine):
    results = engine.rrf_search(
        "ceit", limit=10, method="bm25", filters={"department": "Civil Engineering"}
    )
    assert {r["id"] for r in results} == {"paper-1", "paper-4"}


def test_paper_type_filter(engine):
    results = engine.rrf_search("ceit", limit=10, method="bm25", filters={"paper_type": "Thesis"})
    assert {r["id"] for r in results} == {"paper-1", "paper-2"}


def test_publication_year_filters(engine):
    eq = engine.rrf_search("ceit", limit=10, method="bm25", filters={"publication_year": 2025})
    assert [r["id"] for r in eq] == ["paper-2"]
    rng = engine.rrf_search(
        "ceit",
        limit=10,
        method="bm25",
        filters={"year_from": 2020, "year_to": 2023},
    )
    assert {r["id"] for r in rng} == {"paper-3", "paper-4"}


def test_year_to_keeps_docs_without_a_year(engine):
    # sqlite maps a missing publication_year to 0 — year_to must KEEP those
    # docs (policy docs have no year). year_from still drops them.
    kept = engine.rrf_search(
        "ceit students school id", limit=10, method="bm25", filters={"year_to": 2000}
    )
    assert {r["id"] for r in kept} == {"policy-h1-r1"}
    dropped = engine.rrf_search(
        "ceit students school id", limit=10, method="bm25", filters={"year_from": 2000}
    )
    assert "policy-h1-r1" not in {r["id"] for r in dropped}


def test_author_and_adviser_filters(engine):
    author = engine.rrf_search("ceit", limit=10, method="bm25", filters={"author": "dela"})
    assert {r["id"] for r in author} == {"paper-1", "paper-2"}
    adviser = engine.rrf_search("ceit", limit=10, method="bm25", filters={"adviser": "maria"})
    assert {r["id"] for r in adviser} == {"paper-1", "paper-2", "paper-3"}


def test_corpus_filter(engine):
    results = engine.rrf_search("students school id", limit=10, method="bm25", corpus="policy")
    assert results and all(r["corpus"] == "policy" for r in results)


def test_semantic_gate_drops_channel(engine):
    # Deterministic vectors: an unmatched query is ~orthogonal to everything,
    # below MIN_SEMANTIC_SIMILARITY — semantic-only must return nothing.
    assert engine.rrf_search("xyzzy nonsense", limit=5, method="semantic") == []


def test_semantic_method_returns_exact_text_match(engine):
    # The deterministic embedder hashes whole text — the exact stored text
    # of policy-h1-r1 gives cosine 1.0, clearing the relevance gate.
    query = "Section: General Information\nStudents must present their school ID."
    results = engine.rrf_search(query, limit=5, method="semantic")
    assert results[0]["id"] == "policy-h1-r1"
    assert all(r["semantic_rank"] is not None for r in results)


def test_negative_query_returns_empty(engine):
    assert engine.rrf_search("!!! ???", limit=5, method="hybrid") == []
    assert engine.rrf_search("zzzqqq outofdomain", limit=5, method="bm25") == []


def test_include_text_adds_text_field(engine):
    results = engine.rrf_search("groundwater", limit=3, method="hybrid", include_text=True)
    assert results and "text" in results[0]


def test_unknown_method_falls_back_to_hybrid(engine):
    hybrid = engine.rrf_search("CEIT-CE-15-014", limit=5, method="hybrid")
    bogus = engine.rrf_search("CEIT-CE-15-014", limit=5, method="bogus")
    assert [r["id"] for r in bogus] == [r["id"] for r in hybrid]
    assert bogus[0]["pinned"] is True


# ---------------------------------------------------------------------------
# pgvector-only cases
# ---------------------------------------------------------------------------


def test_pg_unsynced_schema_returns_empty(pg_schema):
    # Schema exists but no documents/index_state — not a 500, just empty.
    engine = PgVectorSearch(
        PG_URL, schema=pg_schema, model_name="test-model", embed_query_fn=_query_embed_fn
    )
    try:
        assert engine.rrf_search("groundwater", limit=5) == []
    finally:
        engine.close()


def test_pg_recommend_not_implemented_yet(pg_engine):
    with pytest.raises(NotImplementedError):
        pg_engine.recommend(["paper-1"], [], 5)


def test_factory_builds_pgvector_engine(pg_schema, monkeypatch):
    from app import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "settings",
        Settings(
            sidecar_token="t",
            vector_backend="pgvector",
            database_url=PG_URL,
            pg_index_schema=pg_schema,
        ),
    )
    engine = backend_mod.create_backend(Path("unused"), "test-model")
    try:
        assert isinstance(engine, PgVectorSearch)
    finally:
        engine.close()


def test_factory_pgvector_requires_database_url(monkeypatch):
    from app import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "settings",
        Settings(sidecar_token="t", vector_backend="pgvector", database_url=""),
    )
    with pytest.raises(ValueError, match="DATABASE_URL"):
        backend_mod.create_backend(Path("unused"), "test-model")


# ---------------------------------------------------------------------------
# Golden parity on the committed corpus (real docs + real embeddings)
# ---------------------------------------------------------------------------


def _sync_committed(store: PgIndexStore) -> dict:
    """Push the committed index artifacts through the real sync path."""
    from app.ingest import load_documents

    docs = load_documents(ROOT / "corpus")
    vectors = np.load(COMMITTED_CACHE / "vectors-1.npy")
    cached = {d["id"]: i for i, d in enumerate(docs)}
    assert len(vectors) == len(docs)
    state = store.sync(
        ROOT / "corpus",
        "all-MiniLM-L6-v2",
        embed_fn=lambda batch: np.stack([vectors[cached[d["id"]]] for d in batch]),
    )
    return state


@pytest.fixture(scope="module")
def prod_pg_engine():
    if not PG_URL:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"ceit_test_{secrets.token_hex(4)}"
    with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        store = PgIndexStore(PG_URL, schema=schema)
        state = _sync_committed(store)
        engine = PgVectorSearch(PG_URL, schema=schema, model_name="all-MiniLM-L6-v2")
        yield engine, state.get("keyword_index")
        engine.close()
    finally:
        with psycopg.connect(PG_URL, autocommit=True, prepare_threshold=None) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _golden_run(engine, method: str) -> dict:
    from app.eval import aggregate, evaluate_case, load_golden

    cases = load_golden()["test_cases"]
    return aggregate([evaluate_case(engine, c, 5, method) for c in cases])


def _restore_real_embed_query(monkeypatch):
    """Other test files permanently rebind app.search.embed_query to a 16-dim
    fake — the golden runs need the real model on the sqlite side too."""
    from app.ingest import embed_query

    monkeypatch.setattr("app.search.embed_query", embed_query)


def test_golden_semantic_channel_matches_sqlite(prod_pg_engine, monkeypatch):
    """pgvector cosine vs numpy cosine — the math is identical in float32."""
    _restore_real_embed_query(monkeypatch)
    engine, _kw = prod_pg_engine
    sqlite = HybridSearch(COMMITTED_CACHE, "all-MiniLM-L6-v2")
    try:
        pg = _golden_run(engine, "semantic")
        sq = _golden_run(sqlite, "semantic")
    finally:
        sqlite.close()
    assert pg["negative_pass_rate"] == sq["negative_pass_rate"]
    for key in ("avg_precision", "avg_recall", "avg_f1", "top1_rate"):
        assert pg[key] == pytest.approx(sq[key], abs=0.02)


def test_golden_bm25_parity_when_lakebase_index(prod_pg_engine, monkeypatch):
    """Exact BM25 parity only where lakebase_text provides the <@> index;
    on the GIN fallback the channel is functional but not score-identical."""
    _restore_real_embed_query(monkeypatch)
    engine, keyword_index = prod_pg_engine
    if keyword_index != "lakebase_bm25":
        pytest.skip("lakebase_bm25 not available on TEST_DATABASE_URL (GIN fallback)")
    sqlite = HybridSearch(COMMITTED_CACHE, "all-MiniLM-L6-v2")
    try:
        pg = _golden_run(engine, "bm25")
        sq = _golden_run(sqlite, "bm25")
    finally:
        sqlite.close()
    assert pg["negative_pass_rate"] == sq["negative_pass_rate"]
    assert pg["avg_precision"] == pytest.approx(sq["avg_precision"], abs=0.02)
    assert pg["avg_recall"] == pytest.approx(sq["avg_recall"], abs=0.02)
    # One top-1 flip measured in the #5 spike (BM25 formula detail).
    assert pg["top1_rate"] == pytest.approx(sq["top1_rate"], abs=0.05)
