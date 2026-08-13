"""POST /chat/stream: token auth, SSE framing, search->context wiring, errors."""

from __future__ import annotations

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

    def create(self, **kwargs):
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

    def rrf_search(self, query, k=60, limit=10, filters=None, corpus=None):
        self.calls.append({"query": query, "corpus": corpus, "limit": limit})
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
    from app.ingest import load_documents
    from app.rebuild import build_index

    cache = tmp_path / "cache"
    load_documents(corpus_path)

    def embed(docs_):
        import numpy as np

        vectors = []
        for d in docs_:
            rng = np.random.RandomState(sum(ord(c) for c in d["text"]))
            v = rng.rand(16).astype(np.float32)
            v = v / np.linalg.norm(v)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)

    build_index(corpus_path, "test-model", cache, embed_fn=embed)

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

    rebuild_mod._embed_override = embed

    return TestClient(main_mod.app)


def test_chat_stream_requires_token(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, [])
    resp = client.post("/chat/stream", json={"query": "school ID"})
    assert resp.status_code == 401


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
    assert engine.calls == [{"query": "school ID", "corpus": "policy", "limit": 3}]


def test_chat_stream_emits_error_event_on_provider_failure(tmp_path, corpus_path):
    client = make_client(tmp_path, corpus_path, SSE_DOCS, fail=True)
    resp = client.post(
        "/chat/stream",
        json={"query": "school ID"},
        headers={"X-Sidecar-Token": "test-token"},
    )
    assert resp.status_code == 200
    assert "event: error" in resp.text
    assert resp.text.endswith("data: [DONE]\n\n")
