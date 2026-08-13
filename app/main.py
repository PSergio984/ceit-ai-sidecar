"""FastAPI app: token-gated /health /search /index/rebuild /metrics."""

from __future__ import annotations

import secrets
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import settings
from .health import assemble_health
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


def _get_cache_dir() -> Path:
    return Path(settings.cache_dir)


def _get_engine() -> HybridSearch:
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearch(_get_cache_dir(), settings.model_name)
    return _search_engine


@app.middleware("http")
async def require_token(request: Request, call_next):
    header = request.headers.get("X-Sidecar-Token", "")
    if not secrets.compare_digest(header, settings.sidecar_token):
        return JSONResponse(
            status_code=401,
            content={"error": {"code": "invalid_request", "message": "missing or invalid X-Sidecar-Token"}},
        )
    return await call_next(request)


@app.get("/health")
def health():
    return assemble_health(_get_cache_dir())


@app.post("/search")
def search(payload: dict):
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": "'query' is required"}},
        )
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
