"""FastAPI app: token-gated /health /search /index/rebuild /metrics."""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import settings
from .health import assemble_health
from .rag import RagService
from .rebuild import rebuild
from .search import HybridSearch

app = FastAPI(title="ceit-ai-sidecar", version="0.1.0")

_metrics_lock = threading.Lock()
_metrics = {
    "searches_total": 0,
    "rebuilds_total": 0,
    "last_rebuild_at": None,
    "search_times_ms": [],
    "index_documents": None,
}

_search_engine: HybridSearch | None = None
_rag: RagService | None = None


def _get_cache_dir() -> Path:
    return Path(settings.cache_dir)


def _get_engine() -> HybridSearch:
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearch(_get_cache_dir(), settings.model_name)
    return _search_engine


def _get_rag() -> RagService:
    global _rag
    if _rag is None:
        _rag = RagService()
    return _rag


def _invalid(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "message": message}},
    )


def _require_query(payload: dict) -> str | JSONResponse:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return _invalid("'query' is required")
    return query


@app.middleware("http")
async def require_token(request: Request, call_next):
    header = request.headers.get("X-Sidecar-Token", "")
    if not secrets.compare_digest(header, settings.sidecar_token):
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "missing or invalid X-Sidecar-Token",
                }
            },
        )
    return await call_next(request)


@app.get("/health")
def health():
    return assemble_health(_get_cache_dir())


@app.post("/search")
def search(payload: dict):
    query = _require_query(payload)
    if isinstance(query, JSONResponse):
        return query
    filters = payload.get("filters") or {}
    corpus = payload.get("corpus")
    limit = int(payload.get("limit", 10))
    k = int(payload.get("k", 60))

    started = time.perf_counter()
    results = _get_engine().rrf_search(query, k=k, limit=limit, filters=filters, corpus=corpus)
    took_ms = int((time.perf_counter() - started) * 1000)

    with _metrics_lock:
        _metrics["searches_total"] += 1
        _metrics["search_times_ms"].append(took_ms)
        _metrics["search_times_ms"] = _metrics["search_times_ms"][-1000:]

    return {"query": query, "total": len(results), "took_ms": took_ms, "results": results}


@app.post("/chat/stream")
def chat_stream(payload: dict):
    """SSE-streamed chat answer over hybrid-search results (ADR 0002).

    Request: {"query": str, "mode": "citations"|"question"|"rag", "corpus"?, "top_k"?}
    Response: text/event-stream — `data: <chunk>` lines, `[DONE]` terminator,
    or an `event: error` line on provider failure.
    """
    query = _require_query(payload)
    if isinstance(query, JSONResponse):
        return query
    mode = payload.get("mode", "citations")
    if not isinstance(mode, str) or mode not in ("citations", "question", "rag"):
        return _invalid("'mode' must be citations, question or rag")
    corpus = payload.get("corpus") or None
    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError):
        return _invalid("'top_k' must be an integer")
    if top_k < 1 or top_k > 50:
        return _invalid("'top_k' must be between 1 and 50")

    results = _get_engine().rrf_search(query, k=60, limit=top_k, corpus=corpus, include_text=True)
    events = _get_rag().stream_events(query, results, mode=mode)

    return StreamingResponse(events, media_type="text/event-stream")


@app.post("/index/rebuild")
def index_rebuild():
    started = time.perf_counter()
    try:
        state = rebuild(settings)
    except Exception as exc:  # noqa: BLE001 - envelope for client
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "rebuild_failed", "message": str(exc)}},
        )
    took_ms = int((time.perf_counter() - started) * 1000)

    with _metrics_lock:
        _metrics["rebuilds_total"] += 1
        _metrics["last_rebuild_at"] = state.get("built_at")
        _metrics["index_documents"] = state.get("documents")

    return {
        "status": "rebuilt",
        "contract_version": state.get("contract_version", "v1"),
        "documents": state.get("documents"),
        "by_corpus": state.get("by_corpus"),
        "took_ms": took_ms,
        "source_generated_at": state.get("source_generated_at"),
    }


@app.get("/metrics")
def metrics():
    with _metrics_lock:
        times = _metrics["search_times_ms"]
        return {
            "searches_total": _metrics["searches_total"],
            "rebuilds_total": _metrics["rebuilds_total"],
            "last_rebuild_at": _metrics["last_rebuild_at"],
            "search_avg_ms": round(sum(times) / len(times), 1) if times else None,
            "index_documents": _metrics["index_documents"],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
