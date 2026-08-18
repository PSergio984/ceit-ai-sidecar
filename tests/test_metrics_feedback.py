"""Prometheus-format /metrics + POST /feedback (deliverable D4).

Seam: the HTTP endpoints. /metrics must emit Prometheus text exposition
(content-type text/plain; version=0.0.4) that a scraper can parse, and /feedback
must persist a JSONL record and move the /metrics counters.
"""

from __future__ import annotations

import json
from pathlib import Path

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
        feedback_path=str(tmp_path / "feedback.jsonl"),
        query_rewrite=False,
        rerank_mode="none",
    )

    import app.main as main_mod
    import app.rebuild as rebuild_mod

    main_mod.settings = settings
    reset_main_singletons(main_mod)
    main_mod._metrics = main_mod._fresh_metrics()
    rebuild_mod._embed_override = embed_from(docs)

    target = np.asarray(embed_from(docs)([docs[0]])[0])

    class FakeQuery:
        def encode(self, texts, normalize_embeddings=True):
            return np.stack([target] * len(texts))

    import app.search as search_mod

    search_mod.embed_query = lambda q, m: FakeQuery().encode([q])[0]

    return TestClient(main_mod.app), settings


H = {"X-Sidecar-Token": "test-token"}


def test_metrics_requires_token(client):
    app, _ = client
    assert app.get("/metrics").status_code == 401


def test_metrics_accepts_bearer_authorization_header(client):
    """Monitoring stacks send Authorization: Bearer; the app must accept it."""
    app, _ = client
    resp = app.get("/metrics", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    assert "ceit_searches_total" in resp.text


def test_metrics_rejects_wrong_bearer(client):
    app, _ = client
    resp = app.get("/metrics", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_metrics_is_prometheus_text_format(client):
    app, _ = client
    resp = app.get("/metrics", headers=H)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    text = resp.text
    assert "# TYPE ceit_searches_total counter" in text
    assert "# TYPE ceit_search_duration_seconds histogram" in text
    assert 'ceit_search_duration_seconds_bucket{le="+Inf"}' in text
    assert "ceit_searches_total 0" in text


def test_metrics_tracks_searches(client):
    app, _ = client
    app.post("/search", json={"query": "water pump"}, headers=H)
    app.post("/search", json={"query": "flood monitoring"}, headers=H)

    text = app.get("/metrics", headers=H).text
    assert "ceit_searches_total 2" in text
    assert "ceit_search_duration_seconds_count 2" in text


def test_search_latency_histogram_is_cumulative(client):
    """Prometheus histogram contract: bucket{le=X} counts EVERY sample <= X."""
    import app.main as main_mod

    app, _ = client
    app.post("/search", json={"query": "water pump"}, headers=H)

    with main_mod._metrics_lock:
        buckets = main_mod._metrics["search_buckets"]
    # Cumulative histogram: bucket{le=X} counts EVERY sample <= X, so counts
    # are non-decreasing as `le` grows, and the largest bucket saw the sample.
    vals = list(buckets.values())
    assert vals == sorted(vals)
    assert vals[-1] >= 1  # the largest bucket saw the sample


def test_search_rejects_non_integer_limit(client):
    app, _ = client
    resp = app.post("/search", json={"query": "water pump", "limit": "many"}, headers=H)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_feedback_requires_token_and_query_and_valid_rating(client):
    app, _ = client
    assert app.post("/feedback", json={}).status_code == 401
    assert app.post("/feedback", json={"rating": "up"}, headers=H).status_code == 422
    assert app.post("/feedback", json={"query": "q", "rating": "meh"}, headers=H).status_code == 422


def test_feedback_records_jsonl_and_moves_counters(client):
    app, settings = client
    resp = app.post(
        "/feedback",
        json={"query": "water pump", "rating": "up", "answer": "answer", "result_ids": ["paper-1"]},
        headers=H,
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "recorded", "rating": "up"}

    path = Path(settings.feedback_path)
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["query"] == "water pump"
    assert lines[0]["rating"] == "up"
    assert lines[0]["result_ids"] == ["paper-1"]

    app.post("/feedback", json={"query": "other", "rating": "down"}, headers=H)
    text = app.get("/metrics", headers=H).text
    assert 'ceit_feedback_total{rating="up"} 1' in text
    assert 'ceit_feedback_total{rating="down"} 1' in text
