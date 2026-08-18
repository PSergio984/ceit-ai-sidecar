"""API behavior: token auth, response shapes, rebuild atomicity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import build_test_index, embed_from, reset_main_singletons
from fastapi.testclient import TestClient

from app.config import Settings


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
        rerank_mode="blend",
    )

    import app.main as main_mod
    import app.rebuild as rebuild_mod

    main_mod.settings = settings
    reset_main_singletons(main_mod)
    main_mod._metrics = main_mod._fresh_metrics()
    rebuild_mod._embed_override = embed_from(docs)
    # Deterministic query embedder.
    target = np.asarray(embed_from(docs)([docs[0]])[0])

    class FakeQuery:
        def encode(self, texts, normalize_embeddings=True):
            return np.stack([target] * len(texts))

    import app.search as search_mod

    search_mod.embed_query = lambda q, m: FakeQuery().encode([q])[0]

    return TestClient(main_mod.app), cache, corpus_path


def test_search_without_token_is_401(client):
    app, _, _ = client
    resp = app.post("/search", json={"query": "water pump"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "auth_failed"


def test_health_without_token_is_401(client):
    app, _, _ = client
    assert app.get("/health").status_code == 401


def test_search_with_token_returns_locked_shape(client):
    app, _, _ = client
    resp = app.post(
        "/search", json={"query": "water pump"}, headers={"X-Sidecar-Token": "test-token"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"query", "total", "took_ms", "results"}
    assert body["results"], "expected results"
    first = body["results"][0]
    assert set(first) >= {
        "id",
        "corpus",
        "title",
        "score",
        "bm25_rank",
        "semantic_rank",
        "metadata",
    }


def test_search_returns_422_without_query(client):
    app, _, _ = client
    resp = app.post("/search", json={}, headers={"X-Sidecar-Token": "test-token"})
    assert resp.status_code == 422


def test_search_rejects_unknown_fields(client):
    app, _, _ = client
    resp = app.post(
        "/search",
        json={"query": "water pump", "availability": {"77": {"available": 1, "total": 2}}},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "unknown field(s)" in resp.json()["error"]["message"]


def test_search_keeps_code_pin_first_after_rerank(client):
    """End-to-end /search regression (D-02): the composed pipeline
    (rewrite -> hybrid retrieval -> blend re-rank) must keep the exact-code
    pinned document at rank 1. This is the failure site from the review that
    previously dropped hybrid top-1 0.95 -> 0.82."""
    app, _, _ = client
    resp = app.post(
        "/search",
        json={"query": "ceit-ee-25-01"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 200
    first = resp.json()["results"][0]
    assert first["id"] == "paper-2"
    assert first["metadata"]["catalog_code"] == "CEIT-EE-25-01"
    assert first["pinned"] is True


def test_search_rejects_non_integer_limit(client):
    app, _, _ = client
    resp = app.post(
        "/search",
        json={"query": "water pump", "limit": "many"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_search_rejects_non_positive_limit_and_k(client):
    app, _, _ = client
    for bad in ({"limit": 0}, {"limit": -3}, {"k": 0}, {"k": -5}):
        resp = app.post(
            "/search",
            json={"query": "water pump", **bad},
            headers={"X-Sidecar-Token": "test-token"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_request"


def test_search_rejects_non_object_filters(client):
    app, _, _ = client
    for bad in ("x", ["a"], 5, [], 0, ""):
        resp = app.post(
            "/search",
            json={"query": "water pump", "filters": bad},
            headers={"X-Sidecar-Token": "test-token"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_request"


def test_search_rejects_non_numeric_year_filters(client):
    app, _, _ = client
    for bad in (
        {"publication_year": "abc"},
        {"year_from": "later"},
        {"year_to": []},
        {"year_from": 2020.5},
    ):
        resp = app.post(
            "/search",
            json={"query": "water pump", "filters": bad},
            headers={"X-Sidecar-Token": "test-token"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "invalid_request"


def test_search_endpoint_accepts_author_adviser_filters(client):
    """author/adviser ride inside the permissive filters dict — 200 + filtered ids."""
    app, _, _ = client
    resp = app.post(
        "/search",
        json={
            "query": "water",
            "filters": {"author": "juan", "adviser": "engr. jose"},
            "corpus": "catalog",
            "limit": 10,
            "k": 60,
        },
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    ids = {r["id"] for r in body["results"]}
    assert ids == {"paper-1", "paper-2"}
    for r in body["results"]:
        meta = r["metadata"]
        assert any("juan" in name.lower() for name in meta["authors"])
        assert any(
            "engr. jose" in (meta.get(k) or "").lower()
            for k in ("research_adviser", "technical_adviser")
        )


def test_search_rejects_author_at_top_level(client):
    """Closed-schema posture unchanged: author is a filters-dict key, never top-level."""
    app, _, _ = client
    resp = app.post(
        "/search",
        json={"query": "water pump", "author": "juan"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"
    assert "unknown field(s): author" in resp.json()["error"]["message"]


def test_search_rejects_exclude_field(client):
    app, _, _ = client
    resp = app.post(
        "/search",
        json={"query": "water pump", "exclude": ["paper-77"]},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_health_reports_coverage(client):
    app, _, _ = client
    resp = app.get("/health", headers={"X-Sidecar-Token": "test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["contract_version"] == "v1"
    assert body["index"]["documents"] == 6
    assert body["index"]["embedded"] == 6
    assert body["index"]["by_corpus"] == {"catalog": 4, "policy": 2}


def test_rebuild_returns_locked_shape(client):
    app, _, _ = client
    resp = app.post("/index/rebuild", json={}, headers={"X-Sidecar-Token": "test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "rebuilt"
    assert body["documents"] == 6
    assert body["by_corpus"] == {"catalog": 4, "policy": 2}
    assert "took_ms" in body
    assert "source_generated_at" in body


def test_concurrent_rebuilds_serialize(client):
    app, cache, _ = client
    headers = {"X-Sidecar-Token": "test-token"}

    def call(_):
        return app.post("/index/rebuild", json={}, headers=headers).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(call, [None, None]))

    assert statuses == [200, 200]
    state = json.loads((cache / "state.json").read_text(encoding="utf-8"))
    assert state["documents"] == 6


def test_corpus_upload_without_token_is_401(client):
    app, _, _ = client
    resp = app.post("/corpus/upload", files={"catalog": ("catalog.json", b"{}")})
    assert resp.status_code == 401


def test_corpus_upload_with_no_files_is_422(client):
    app, _, _ = client
    resp = app.post("/corpus/upload", headers={"X-Sidecar-Token": "test-token"})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_corpus_upload_catalog_rebuilds(client):
    app, cache, corpus_path = client
    content = (corpus_path / "catalog.json").read_bytes()

    resp = app.post(
        "/corpus/upload",
        headers={"X-Sidecar-Token": "test-token"},
        files={"catalog": ("catalog.json", content, "application/json")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "uploaded_and_rebuilt"
    assert body["files"] == ["catalog.json"]
    assert body["documents"] == 6  # catalog 4 + existing policies 2
    assert body["by_corpus"] == {"catalog": 4, "policy": 2}

    state = json.loads((cache / "state.json").read_text(encoding="utf-8"))
    assert state["documents"] == 6


def test_corpus_upload_invalid_json_fails_closed(client):
    app, cache, corpus_path = client
    resp = app.post(
        "/corpus/upload",
        headers={"X-Sidecar-Token": "test-token"},
        files={"catalog": ("catalog.json", b"{not json", "application/json")},
    )
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "upload_failed"
    # Fail-closed: the bad file was removed, previous corpus + index intact.
    assert not (corpus_path / "catalog.json").exists()
    assert (corpus_path / "policies.json").exists()

    health = app.get("/health", headers={"X-Sidecar-Token": "test-token"})
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    state = json.loads((cache / "state.json").read_text(encoding="utf-8"))
    assert state["documents"] == 6
