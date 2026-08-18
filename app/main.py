"""FastAPI app: token-gated /health /search /index/rebuild /metrics /feedback."""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .agent import AgenticLoop
from .config import settings
from .health import assemble_health
from .rag import RagService
from .rebuild import load_state, rebuild
from .rerank import Reranker
from .rewrite import QueryRewriter
from .search import RRF_K, HybridSearch

logger = logging.getLogger("ceit.sidecar")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _log_handler = logging.StreamHandler()
    _log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_log_handler)
logger.propagate = False

app = FastAPI(title="ceit-ai-sidecar", version="0.1.0")

_metrics_lock = threading.Lock()

# Prometheus histogram buckets for search latency (seconds), upper-bounded.
SEARCH_LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def _fresh_metrics() -> dict:
    return {
        "searches_total": 0,
        "chat_searches_total": 0,
        "rebuilds_total": 0,
        "feedback_up": 0,
        "feedback_down": 0,
        "last_rebuild_at": None,
        "search_sum_ms": 0.0,
        "search_buckets": {le: 0 for le in SEARCH_LATENCY_BUCKETS},
        "index_documents": None,
    }


_metrics = _fresh_metrics()

_search_engine: HybridSearch | None = None
_rag: RagService | None = None
_agent: AgenticLoop | None = None
_rewriter: QueryRewriter | None = None
_reranker: Reranker | None = None


def _get_cache_dir() -> Path:
    return Path(settings.cache_dir)


def _get_engine() -> HybridSearch:
    global _search_engine
    if _search_engine is None:
        _search_engine = HybridSearch(_get_cache_dir(), settings.model_name)
    return _search_engine


def _get_rewriter() -> QueryRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = QueryRewriter()
    return _rewriter


def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def _get_rag() -> RagService:
    global _rag
    if _rag is None:
        _rag = RagService()
    return _rag


def _count_chat_search() -> None:
    """Record one executed retrieval inside /chat/stream (tool-call hook, D-11)."""
    with _metrics_lock:
        _metrics["chat_searches_total"] += 1
    logger.info("chat retrieval executed (tool call)")


def _get_agent() -> AgenticLoop:
    global _agent
    if _agent is None:
        _agent = AgenticLoop(engine=_get_engine(), on_search=_count_chat_search)
    return _agent


def _invalid(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "message": message}},
    )


SEARCH_ALLOWED_KEYS = {"query", "filters", "corpus", "limit", "k"}
CHAT_ALLOWED_KEYS = {"query", "mode", "corpus", "top_k"}


def _reject_unknown(payload: dict, allowed: set[str]) -> JSONResponse | None:
    unknown = set(payload) - allowed
    if unknown:
        return _invalid(f"unknown field(s): {', '.join(sorted(unknown))}")
    return None


def _require_query(payload: dict) -> str | JSONResponse:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return _invalid("'query' is required")
    return query


@app.middleware("http")
async def require_token(request: Request, call_next):
    # Primary contract: X-Sidecar-Token (the Laravel integration). Also accept
    # Authorization: Bearer <token> so Prometheus/Grafana scrapers that send a
    # Bearer token work without an app change.
    header = request.headers.get("X-Sidecar-Token", "")
    if not header:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            header = auth[len("Bearer ") :]
    if not secrets.compare_digest(header, settings.sidecar_token):
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "code": "auth_failed",
                    "message": "missing or invalid X-Sidecar-Token",
                }
            },
        )
    return await call_next(request)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Runtime request log: method, path, status, duration — every request."""
    started = time.perf_counter()
    response = await call_next(request)
    took_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "req %s %s -> %s (%dms)",
        request.method,
        request.url.path,
        response.status_code,
        took_ms,
    )
    return response


def _index_doc_count() -> int:
    """Total documents in the current index state (0 = no index built)."""
    state = load_state(Path(settings.cache_dir))
    return int((state or {}).get("documents", 0))


@app.get("/health")
def health():
    return assemble_health(_get_cache_dir())


@app.post("/search")
def search(payload: dict):
    rejected = _reject_unknown(payload, SEARCH_ALLOWED_KEYS)
    if rejected:
        return rejected
    query = _require_query(payload)
    if isinstance(query, JSONResponse):
        return query
    raw_filters = payload.get("filters")
    if raw_filters is not None and not isinstance(raw_filters, dict):
        return _invalid("'filters' must be an object")
    filters = raw_filters or {}
    corpus = payload.get("corpus")

    # Validate at the request boundary so malformed values fail with 422
    # instead of crashing inside retrieval.
    for key in ("publication_year", "year_from", "year_to"):
        value = filters.get(key)
        if value is None:
            continue
        if isinstance(value, (bool, float)):
            return _invalid(f"'filters.{key}' must be an integer")
        try:
            int(value)
        except (TypeError, ValueError):
            return _invalid(f"'filters.{key}' must be an integer")

    try:
        limit = int(payload.get("limit", 10))
        k = int(payload.get("k", RRF_K))
    except (TypeError, ValueError):
        return _invalid("'limit' and 'k' must be integers")
    if limit < 1:
        return _invalid("'limit' must be positive")
    if k < 1:
        return _invalid("'k' must be positive")

    # Best-practice extras (D6): query rewriting then re-ranking around the
    # hybrid retrieval. Both degrade safely when disabled / no LLM key.
    started = time.perf_counter()
    search_query = _get_rewriter().rewrite(query) if settings.query_rewrite else query
    include_text = settings.rerank_mode == "llm"

    index_docs = _index_doc_count()
    logger.info(
        "search query=%r rewritten=%r corpus=%s filters=%s k=%d limit=%d index_docs=%d rerank=%s",
        query,
        search_query if search_query != query else "(same)",
        corpus or "both",
        filters or None,
        k,
        limit,
        index_docs,
        settings.rerank_mode,
    )

    results = _get_engine().rrf_search(
        search_query,
        k=k,
        limit=limit,
        filters=filters,
        corpus=corpus,
        include_text=include_text,
    )
    if settings.rerank_mode != "none":
        results = _get_reranker().rerank(search_query, results)
    took_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "search done query=%r results=%d/%d took_ms=%d %s",
        query,
        len(results),
        index_docs,
        took_ms,
        "(EMPTY — no index built? run /index/rebuild or /corpus/upload)" if index_docs == 0 else "",
    )

    with _metrics_lock:
        _metrics["searches_total"] += 1
        _metrics["search_sum_ms"] += took_ms
        seconds = took_ms / 1000.0
        # Cumulative histogram: every sample counts into every bucket >= it, so
        # bucket{le="0.1"} = count of samples <= 0.1s (Prometheus contract).
        for le in SEARCH_LATENCY_BUCKETS:
            if seconds <= le:
                _metrics["search_buckets"][le] += 1

    body = {"query": query, "total": len(results), "took_ms": took_ms, "results": results}
    if search_query != query:
        body["rewritten_query"] = search_query
    return body


@app.post("/chat/stream")
def chat_stream(payload: dict):
    """SSE-streamed chat answer over hybrid-search results (ADR 0002).

    Request: {"query": str, "mode": "citations"|"question"|"rag", "corpus"?, "top_k"?}
    Response: text/event-stream — `data: <chunk>` lines, `[DONE]` terminator,
    or an `event: error` line on provider failure.
    """
    rejected = _reject_unknown(payload, CHAT_ALLOWED_KEYS)
    if rejected:
        return rejected
    query = _require_query(payload)
    if isinstance(query, JSONResponse):
        return query
    mode = payload.get("mode", "citations")
    if not isinstance(mode, str) or mode not in ("citations", "question", "rag"):
        return _invalid("'mode' must be citations, question or rag")
    corpus = payload.get("corpus") or None
    if corpus is not None and corpus not in ("catalog", "policy"):
        return _invalid("'corpus' must be catalog, policy or omitted")
    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError):
        return _invalid("'top_k' must be an integer")
    if top_k < 1 or top_k > 50:
        return _invalid("'top_k' must be between 1 and 50")

    events = _get_agent().stream_agentic_events(
        query, mode=mode, corpus=corpus, default_top_k=top_k
    )

    logger.info(
        "chat query=%r mode=%s corpus=%s top_k=%d index_docs=%d",
        query,
        mode,
        corpus or "both",
        top_k,
        _index_doc_count(),
    )

    return StreamingResponse(
        events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/index/rebuild")
def index_rebuild():
    started = time.perf_counter()
    try:
        state = rebuild(settings)
    except Exception as exc:  # noqa: BLE001 - envelope for client
        logger.error("index rebuild FAILED: %r", exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "rebuild_failed", "message": str(exc)}},
        )
    took_ms = int((time.perf_counter() - started) * 1000)

    with _metrics_lock:
        _metrics["rebuilds_total"] += 1
        _metrics["last_rebuild_at"] = state.get("built_at")
        _metrics["index_documents"] = state.get("documents")

    logger.info(
        "index rebuilt: %s documents by_corpus=%s took_ms=%d source_generated_at=%s",
        state.get("documents"),
        state.get("by_corpus"),
        took_ms,
        state.get("source_generated_at"),
    )

    return {
        "status": "rebuilt",
        "contract_version": state.get("contract_version", "v1"),
        "documents": state.get("documents"),
        "by_corpus": state.get("by_corpus"),
        "took_ms": took_ms,
        "source_generated_at": state.get("source_generated_at"),
    }


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@app.post("/corpus/upload")
def corpus_upload(
    catalog: Annotated[UploadFile | None, File()] = None,
    policies: Annotated[UploadFile | None, File()] = None,
):
    """Replace corpus files in CORPUS_PATH and rebuild the index.

    Cloud-deployment hand-off (FastAPI Cloud has no Laravel beside it):
    the Laravel `ai:push-corpus` command exports and uploads the corpus
    here. At least one of `catalog`/`policies` must be present; files are
    written under settings.corpus_path and the index is rebuilt atomically.
    On validation failure the uploaded files are removed so the previous
    corpus stays intact.
    """
    if catalog is None and policies is None:
        return _invalid("provide at least one of 'catalog' or 'policies'")

    corpus_dir = Path(settings.corpus_path)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    uploaded: list[str] = []
    try:
        for field, filename in ((catalog, "catalog.json"), (policies, "policies.json")):
            if field is not None:
                content = field.file.read()
                if len(content) > MAX_UPLOAD_BYTES:
                    return _invalid(f"{filename} exceeds the {MAX_UPLOAD_BYTES}-byte upload cap")
                (corpus_dir / filename).write_bytes(content)
                uploaded.append(filename)

        started = time.perf_counter()
        try:
            state = rebuild(settings)
        except Exception as exc:  # noqa: BLE001 - invalid corpus, fail closed
            for filename in uploaded:
                (corpus_dir / filename).unlink(missing_ok=True)
            logger.error("corpus upload FAILED (files removed): %r", exc)
            return JSONResponse(
                status_code=500,
                content={"error": {"code": "upload_failed", "message": str(exc)}},
            )
        took_ms = int((time.perf_counter() - started) * 1000)

        with _metrics_lock:
            _metrics["rebuilds_total"] += 1
            _metrics["last_rebuild_at"] = state.get("built_at")
            _metrics["index_documents"] = state.get("documents")

        logger.info(
            "corpus uploaded: files=%s documents=%s by_corpus=%s took_ms=%d source_generated_at=%s",
            uploaded,
            state.get("documents"),
            state.get("by_corpus"),
            took_ms,
            state.get("source_generated_at"),
        )

        return {
            "status": "uploaded_and_rebuilt",
            "files": uploaded,
            "contract_version": state.get("contract_version", "v1"),
            "documents": state.get("documents"),
            "by_corpus": state.get("by_corpus"),
            "took_ms": took_ms,
            "source_generated_at": state.get("source_generated_at"),
        }
    finally:
        if catalog is not None:
            catalog.file.close()
        if policies is not None:
            policies.file.close()


def _last_rebuild_epoch() -> float | None:
    raw = _metrics["last_rebuild_at"]
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


@app.get("/metrics")
def metrics():
    """Prometheus text exposition (deliverable D4) — scraped by the compose stack."""
    with _metrics_lock:
        lines = [
            "# HELP ceit_searches_total Total hybrid search requests served.",
            "# TYPE ceit_searches_total counter",
            f"ceit_searches_total {_metrics['searches_total']}",
            "# HELP ceit_chat_searches_total Retrievals executed inside /chat/stream.",
            "# TYPE ceit_chat_searches_total counter",
            f"ceit_chat_searches_total {_metrics['chat_searches_total']}",
            "# HELP ceit_rebuilds_total Total index rebuilds completed.",
            "# TYPE ceit_rebuilds_total counter",
            f"ceit_rebuilds_total {_metrics['rebuilds_total']}",
            "# HELP ceit_feedback_total Thumbs-up/down feedback received.",
            "# TYPE ceit_feedback_total counter",
            f'ceit_feedback_total{{rating="up"}} {_metrics["feedback_up"]}',
            f'ceit_feedback_total{{rating="down"}} {_metrics["feedback_down"]}',
            "# HELP ceit_search_duration_seconds Search latency histogram (seconds).",
            "# TYPE ceit_search_duration_seconds histogram",
        ]
        for le in SEARCH_LATENCY_BUCKETS:
            lines.append(
                f'ceit_search_duration_seconds_bucket{{le="{le}"}} {_metrics["search_buckets"][le]}'
            )
        lines.append(
            f'ceit_search_duration_seconds_bucket{{le="+Inf"}} {_metrics["searches_total"]}'
        )
        lines.append(f"ceit_search_duration_seconds_sum {_metrics['search_sum_ms'] / 1000.0:.6f}")
        lines.append(f"ceit_search_duration_seconds_count {_metrics['searches_total']}")
        lines.append("# HELP ceit_index_documents Documents in the current index.")
        lines.append("# TYPE ceit_index_documents gauge")
        lines.append(f"ceit_index_documents {_metrics['index_documents'] or 0}")
        epoch = _last_rebuild_epoch()
        if epoch is not None:
            lines.append("# HELP ceit_last_rebuild_timestamp_seconds Last index rebuild (Unix).")
            lines.append("# TYPE ceit_last_rebuild_timestamp_seconds gauge")
            lines.append(f"ceit_last_rebuild_timestamp_seconds {epoch:.3f}")

    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4",
    )


FEEDBACK_ALLOWED_KEYS = {"query", "rating", "answer", "result_ids"}


@app.post("/feedback")
def feedback(payload: dict):
    """Record a thumbs up/down for a search/answer (deliverable D4).

    Durable log: one JSONL line per rating under `feedback_path`. Counters feed
    the Prometheus `ceit_feedback_total` series so the dashboard can chart
    satisfaction over time.
    """
    rejected = _reject_unknown(payload, FEEDBACK_ALLOWED_KEYS)
    if rejected:
        return rejected
    query = _require_query(payload)
    if isinstance(query, JSONResponse):
        return query
    rating = payload.get("rating")
    if rating not in ("up", "down"):
        return _invalid("'rating' must be 'up' or 'down'")

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "rating": rating,
        "answer": payload.get("answer"),
        "result_ids": payload.get("result_ids") or [],
    }
    # The feedback log deliberately keeps the raw query/answer so the library
    # team can inspect what got thumbs up/down. It is runtime-only data:
    # written under FEEDBACK_PATH (var/, gitignored), never exposed through an
    # endpoint, and reachable only by an authenticated client. Rotate/delete
    # var/feedback.jsonl as part of normal data hygiene.
    path = Path(settings.feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    with _metrics_lock:
        if rating == "up":
            _metrics["feedback_up"] += 1
        else:
            _metrics["feedback_down"] += 1

    return {"status": "recorded", "rating": rating}


# Startup diagnostics: log the current index state so a fresh deploy with an
# empty cache is immediately visible in the runtime logs (searches would
# return empty results until a rebuild completes).
_state = load_state(Path(settings.cache_dir))
if _state is None:
    logger.warning(
        "startup: NO INDEX BUILT — /search and /chat/stream return empty until "
        "a rebuild completes (/index/rebuild or /corpus/upload)"
    )
else:
    logger.info(
        "startup: index built at %s — %s documents by_corpus=%s model=%s source_generated_at=%s",
        _state.get("built_at"),
        _state.get("documents"),
        _state.get("by_corpus"),
        _state.get("model_name"),
        _state.get("source_generated_at"),
    )

# REBUILD_ON_STARTUP: run the full index build in a background thread so a
# cold start on a small cloud instance doesn't block/outlive an HTTP request
# (FastAPI Cloud restarts instances whose request handlers run too long).
# /health stays degraded until the build completes, then flips to ok. This is
# a no-op when a valid index already exists — a prebuilt index committed
# under `index/` is loaded as-is and never re-embedded.
if settings.rebuild_on_startup and _state is None:

    def _background_rebuild() -> None:
        try:
            state = rebuild(settings)
        except Exception as exc:  # noqa: BLE001 - never let the thread die silently
            logger.error("startup background rebuild FAILED: %r", exc)
            return
        with _metrics_lock:
            _metrics["rebuilds_total"] += 1
            _metrics["last_rebuild_at"] = state.get("built_at")
            _metrics["index_documents"] = state.get("documents")
        logger.info(
            "startup background rebuild done: %s documents by_corpus=%s source_generated_at=%s",
            state.get("documents"),
            state.get("by_corpus"),
            state.get("source_generated_at"),
        )

    import threading as _threading

    _bg = _threading.Thread(target=_background_rebuild, name="startup-rebuild", daemon=True)
    _bg.start()
    logger.info("startup: background index rebuild started (REBUILD_ON_STARTUP=true)")
elif settings.rebuild_on_startup and _state is not None:
    logger.info("startup: valid index already present — skipping background rebuild")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port)
