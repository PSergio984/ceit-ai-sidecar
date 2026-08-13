"""API behavior: token auth, response shapes, rebuild atomicity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import build_test_index, embed_from
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
    )

    import app.main as main_mod
    import app.rebuild as rebuild_mod

    main_mod.settings = settings
    main_mod._search_engine = None
    main_mod._metrics = {
        "searches_total": 0,
        "rebuilds_total": 0,
        "last_rebuild_at": None,
        "search_times_ms": [],
        "index_documents": None,
    }
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
    assert resp.json()["error"]["code"] == "invalid_request"


def test_health_without_token_is_401(client):
    app, _, _ = client
    assert app.get("/health").status_code == 401


def test_search_with_token_returns_locked_shape(client):
    app, _, _ = client
    resp = app.post("/search", json={"query": "water pump"}, headers={"X-Sidecar-Token": "test-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"query", "total", "took_ms", "results"}
    assert body["results"], "expected results"
    first = body["results"][0]
    assert set(first) >= {"id", "corpus", "title", "score", "bm25_rank", "semantic_rank", "metadata"}


def test_search_returns_422_without_query(client):
    app, _, _ = client
    resp = app.post("/search", json={}, headers={"X-Sidecar-Token": "test-token"})
    assert resp.status_code == 422


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
