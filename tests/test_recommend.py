"""POST /recommend + recommend_rank: deterministic seed-based recommendations.

Unit tests exercise the pure fusion function over hand-built vectors; API
tests exercise the endpoint over a real versioned index (deterministic
embedder, no model download).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import build_test_index, embed_from, reset_main_singletons
from fastapi.testclient import TestClient

from app.config import Settings
from app.fusion import recommend_rank

H = {"X-Sidecar-Token": "test-token"}


def _doc(doc_id: str, corpus: str = "catalog") -> dict:
    return {
        "id": doc_id,
        "corpus": corpus,
        "title": f"Title {doc_id}",
        "metadata": {"catalog_code": doc_id.upper()},
    }


@pytest.fixture
def docs_and_vectors() -> tuple[dict, np.ndarray]:
    """5 docs; rows align with list(docs_by_id.keys()) like _semantic_scores."""
    docs = {
        "a": _doc("a"),
        "d": _doc("d"),
        "c": _doc("c"),  # identical vector to "d" — tie must break by id
        "b": _doc("b"),
        "pol": _doc("pol", corpus="policy"),
    }
    vectors = np.array(
        [
            [1.0, 0.0],  # a
            [0.6, 0.8],  # d
            [0.6, 0.8],  # c
            [0.0, 1.0],  # b
            [1.0, 0.0],  # pol (policy — never a seed or a result)
        ],
        dtype=np.float32,
    )
    return docs, vectors


def test_recommend_seeds_and_excludes_never_in_results(docs_and_vectors):
    docs, vectors = docs_and_vectors
    seeds_used, results = recommend_rank(docs, vectors, ["a"], ["c"], limit=10)
    assert seeds_used == 1
    ids = {r["id"] for r in results}
    assert "a" not in ids
    assert "c" not in ids


def test_recommend_unknown_seed_is_ignored(docs_and_vectors):
    docs, vectors = docs_and_vectors
    seeds_used, results = recommend_rank(docs, vectors, ["a", "ghost"], [], limit=10)
    assert seeds_used == 1
    assert results


def test_recommend_dedupes_seed_ids(docs_and_vectors):
    docs, vectors = docs_and_vectors
    seeds_used, _ = recommend_rank(docs, vectors, ["a", "a", "ghost"], [], limit=10)
    assert seeds_used == 1


def test_recommend_policy_seed_is_ignored_and_policy_never_returned(docs_and_vectors):
    docs, vectors = docs_and_vectors
    seeds_used, results = recommend_rank(docs, vectors, ["a", "pol"], [], limit=10)
    assert seeds_used == 1
    assert all(r["corpus"] == "catalog" for r in results)

    # A policy-only seed resolves nothing.
    seeds_used, results = recommend_rank(docs, vectors, ["pol"], [], limit=10)
    assert seeds_used == 0
    assert results == []


def test_recommend_no_resolvable_seed_returns_empty(docs_and_vectors):
    docs, vectors = docs_and_vectors
    assert recommend_rank(docs, vectors, ["ghost-1", "ghost-2"], [], limit=10) == (0, [])


def test_recommend_orders_by_cosine_to_mean_of_seeds(docs_and_vectors):
    docs, vectors = docs_and_vectors
    # mean(a, b) = [0.5, 0.5] -> c/d both at cosine ~0.99 (tie -> id order).
    seeds_used, results = recommend_rank(docs, vectors, ["a", "b"], [], limit=10)
    assert seeds_used == 2
    assert [r["id"] for r in results] == ["c", "d"]
    assert results[0]["score"] == pytest.approx(0.9899, abs=1e-4)


def test_recommend_ties_break_by_id(docs_and_vectors):
    docs, vectors = docs_and_vectors
    # seed a -> query [1, 0]: c and d tie at 0.6 despite "d" preceding "c"
    # in docs_by_id; the sort must order by id.
    _, results = recommend_rank(docs, vectors, ["a"], [], limit=10)
    assert [r["id"] for r in results] == ["c", "d", "b"]
    assert results[0]["score"] == 0.6  # rounded to 4 dp


def test_recommend_limit_truncates(docs_and_vectors):
    docs, vectors = docs_and_vectors
    _, results = recommend_rank(docs, vectors, ["a"], [], limit=2)
    assert [r["id"] for r in results] == ["c", "d"]


def test_recommend_result_shape(docs_and_vectors):
    docs, vectors = docs_and_vectors
    _, results = recommend_rank(docs, vectors, ["a"], [], limit=1)
    assert results[0] == {
        "id": "c",
        "corpus": "catalog",
        "title": "Title c",
        "score": 0.6,
        "metadata": {"catalog_code": "C"},
    }


def test_recommend_is_deterministic(docs_and_vectors):
    docs, vectors = docs_and_vectors
    first = recommend_rank(docs, vectors, ["a", "b"], ["d"], limit=5)
    second = recommend_rank(docs, vectors, ["a", "b"], ["d"], limit=5)
    assert first == second


@pytest.fixture
def client(tmp_path, corpus_path):
    cache, docs = build_test_index(tmp_path, corpus_path)

    settings = Settings(
        sidecar_token="test-token",
        corpus_path=corpus_path,
        model_name="test-model",
        host="127.0.0.1",
        port=8310,
        cache_dir=str(cache),
        rerank_mode="none",
        query_rewrite=False,
    )

    import app.main as main_mod
    import app.rebuild as rebuild_mod

    main_mod.settings = settings
    reset_main_singletons(main_mod)
    main_mod._metrics = main_mod._fresh_metrics()
    rebuild_mod._embed_override = embed_from(docs)

    return TestClient(main_mod.app)


def test_recommend_happy_path_response_keys(client):
    resp = client.post("/recommend", json={"seed_ids": ["paper-1"]}, headers=H)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"seeds_used", "total", "took_ms", "results"}
    assert body["seeds_used"] == 1
    assert body["total"] == len(body["results"])
    assert isinstance(body["took_ms"], int)
    ids = {r["id"] for r in body["results"]}
    assert "paper-1" not in ids
    for r in body["results"]:
        assert set(r) == {"id", "corpus", "title", "score", "metadata"}
        assert r["corpus"] == "catalog"


def test_recommend_requires_token(client):
    assert client.post("/recommend", json={"seed_ids": ["paper-1"]}).status_code == 401


def test_recommend_rejects_unknown_key(client):
    resp = client.post(
        "/recommend",
        json={"seed_ids": ["paper-1"], "corpus": "catalog"},
        headers=H,
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "unknown field(s)" in resp.json()["error"]["message"]


def test_recommend_rejects_bad_seed_ids(client):
    bad_payloads = [
        {},  # missing
        {"seed_ids": []},  # empty
        {"seed_ids": [f"paper-{i}" for i in range(21)]},  # >20
        {"seed_ids": "paper-1"},  # not a list
        {"seed_ids": ["paper-1", 5]},  # non-string element
        {"seed_ids": [True]},  # bool element
        {"seed_ids": ["paper-1", ""]},  # empty string
    ]
    for payload in bad_payloads:
        resp = client.post("/recommend", json=payload, headers=H)
        assert resp.status_code == 422, payload
        assert resp.json()["error"]["code"] == "invalid_request"


def test_recommend_rejects_bad_exclude_ids(client):
    for bad in (
        [f"x-{i}" for i in range(201)],  # >200
        "paper-2",  # not a list
        ["paper-2", 7],  # non-string element
    ):
        resp = client.post(
            "/recommend",
            json={"seed_ids": ["paper-1"], "exclude_ids": bad},
            headers=H,
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "invalid_request"


def test_recommend_rejects_bad_limit(client):
    for bad in (0, 51, True, "5", 1.5):
        resp = client.post(
            "/recommend",
            json={"seed_ids": ["paper-1"], "limit": bad},
            headers=H,
        )
        assert resp.status_code == 422, bad
        assert resp.json()["error"]["code"] == "invalid_request"


def test_recommend_all_unknown_seeds_returns_empty_200(client):
    resp = client.post(
        "/recommend",
        json={"seed_ids": ["nope-1", "nope-2"]},
        headers=H,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["seeds_used"] == 0
    assert body["total"] == 0
    assert body["results"] == []


def test_recommend_policy_seed_resolves_nothing(client):
    resp = client.post(
        "/recommend",
        json={"seed_ids": ["policy-h1"]},
        headers=H,
    )
    assert resp.status_code == 200
    assert resp.json()["seeds_used"] == 0


def test_recommend_repeated_calls_are_identical(client):
    payload = {"seed_ids": ["paper-1", "paper-2"], "exclude_ids": ["paper-3"], "limit": 5}
    first = client.post("/recommend", json=payload, headers=H).json()
    second = client.post("/recommend", json=payload, headers=H).json()
    assert first["seeds_used"] == second["seeds_used"] == 2
    assert first["results"] == second["results"]
    assert {r["id"] for r in first["results"]}.isdisjoint({"paper-1", "paper-2", "paper-3"})


def test_recommend_returns_catalog_docs_only(client):
    resp = client.post("/recommend", json={"seed_ids": ["paper-1"], "limit": 50}, headers=H)
    assert resp.status_code == 200
    assert all(r["corpus"] == "catalog" for r in resp.json()["results"])
    assert len(resp.json()["results"]) == 3  # 4 catalog docs minus the seed


def test_recommend_metrics(client):
    client.post("/recommend", json={"seed_ids": ["paper-1"]}, headers=H)
    text = client.get("/metrics", headers=H).text
    assert "ceit_recommend_total 1" in text
    assert "# TYPE ceit_recommend_duration_seconds histogram" in text
    assert 'ceit_recommend_duration_seconds_bucket{le="+Inf"} 1' in text
    assert "ceit_recommend_duration_seconds_count 1" in text
    assert "ceit_recommend_duration_seconds_sum" in text

    client.post("/recommend", json={"seed_ids": ["paper-2"]}, headers=H)
    text = client.get("/metrics", headers=H).text
    assert "ceit_recommend_total 2" in text
