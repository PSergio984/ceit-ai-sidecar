"""POST /chat/stream: token auth, SSE framing, search->context wiring, errors."""

from __future__ import annotations

from conftest import build_test_index, embed_from
from fastapi.testclient import TestClient

import app.main as main_mod
from app.rag import RagService

SSE_DOCS = [
    {
        "id": "policy-h1-r1",
        "corpus": "policy",
        "title": "General Information",
        "text": "Section: General Information\nStudents must present their school ID.",
        "score": 0.8,
        "metadata": {"policy_type": "regulation"},
    }
]


class FakeCompletions:
    def __init__(self, content: str, fail: bool = False):
        self.content = content
        self.fail = fail
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("provider exploded")
        if kwargs.get("stream"):
            return iter(_chunks(self.content))
        return _response(self.content)


def _chunks(content: str):
    for word in content.split():
        yield _stream_chunk(word + " ")


def _stream_chunk(delta: str):
    return type(
        "Chunk", (), {"choices": [type("C", (), {"delta": type("D", (), {"content": delta})()})()]}
    )()


def _response(content: str):
    return type(
        "Resp",
        (),
        {"choices": [type("C", (), {"message": type("M", (), {"content": content})()})()]},
    )()


class FakeCompletionsHolder:
    def __init__(self, content: str, fail: bool = False):
        self.chat = type("Chat", (), {"completions": FakeCompletions(content, fail)})()


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def rrf_search(self, query, k=60, limit=10, filters=None, corpus=None, include_text=False):
        self.calls.append({"query": query, "corpus": corpus, "limit": limit, "include_text": include_text})
        return self.results


def make_client(
    tmp_path,
    corpus_path,
    engine_results,
    *,
    content="Students must present their school ID. ",
    fail=False,
):
    from app.config import Settings

    cache, docs = build_test_index(tmp_path, corpus_path)

    settings = Settings(
        sidecar_token="test-token",
        corpus_path=corpus_path,
        model_name="test-model",
        host="127.0.0.1",
        port=8310,
        cache_dir=str(cache),
    )

    main_mod.settings = settings
    main_mod._search_engine = FakeEngine(engine_results)
    main_mod._rag = RagService(
        client=FakeCompletionsHolder(content, fail),
        model="test-model",
        max_tokens=64,
    )

    import app.rebuild as rebuild_mod

    rebuild_mod._embed_override = embed_from(docs)

    return TestClient(main_mod.app)


def test_chat_stream_requires_token(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, [])
    resp = client.post("/chat/stream", json={"query": "school ID"})
    assert resp.status_code == 401


def test_chat_stream_refuses_on_empty_retrieval_without_llm_call(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, [])
    resp = client.post(
        "/chat/stream",
        json={"query": "school ID"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 200
    assert resp.text == "data: I don't have enough information\n\n" + "data: [DONE]\n\n"
    assert main_mod._rag._client.chat.completions.calls == []


def test_chat_stream_rejects_missing_query(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, [])
    resp = client.post(
        "/chat/stream", json={"mode": "citations"}, headers={"X-Sidecar-Token": "test-token"}
    )
    assert resp.status_code == 422


def test_chat_stream_streams_chunks_and_done(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, SSE_DOCS)
    resp = client.post(
        "/chat/stream",
        json={"query": "school ID"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    body = resp.text
    assert "data: Students " in body
    assert body.endswith("data: [DONE]\n\n")


def test_chat_stream_feeds_search_results_into_prompt(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, SSE_DOCS)
    client.post(
        "/chat/stream",
        json={"query": "school ID", "corpus": "policy", "top_k": 3},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert client.app.state is not None  # keep fixture alive
    engine = main_mod._search_engine
    assert engine.calls == [
        {"query": "school ID", "corpus": "policy", "limit": 3, "include_text": True}
    ]


def test_chat_stream_rejects_non_numeric_top_k(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, SSE_DOCS)
    resp = client.post(
        "/chat/stream",
        json={"query": "school ID", "top_k": "many"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_chat_stream_rejects_unknown_corpus(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, SSE_DOCS)
    resp = client.post(
        "/chat/stream",
        json={"query": "school ID", "corpus": "bogus"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_request"


def test_rrf_search_can_include_document_text(tmp_path, corpus_path):
    import numpy as np

    import app.search as search_mod
    from app.ingest import load_documents
    from app.rebuild import build_index
    from app.search import HybridSearch

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
    target = np.asarray(embed([docs[0]])[0])

    class FakeQuery:
        def encode(self, texts, normalize_embeddings=True):
            return np.stack([target] * len(texts))

    search_mod.embed_query = lambda q, m: FakeQuery().encode([q])[0]

    engine = HybridSearch(cache, "test-model")
    results = engine.rrf_search("groundwater", limit=2, include_text=True)

    assert results, "expected results"
    assert all("text" in r and len(r["text"]) > 0 for r in results)
    assert results[0]["text"].startswith("Analysis of Groundwater Depletion")
    engine.close()


def test_chat_stream_emits_error_event_on_provider_failure(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, SSE_DOCS, fail=True)
    resp = client.post(
        "/chat/stream",
        json={"query": "school ID"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert '"code": "provider_error"' in resp.text
    assert "provider exploded" not in resp.text
    assert resp.text.endswith("data: [DONE]\n\n")
