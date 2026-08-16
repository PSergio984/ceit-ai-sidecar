# ceit-ai-sidecar

Hybrid search sidecar for the CEIT-Library AI assistant (Milestone v2.0, Phase 8).

Combines FTS5 BM25 keyword search with multilingual semantic embeddings
(`paraphrase-multilingual-MiniLM-L12-v2`) fused via Reciprocal Rank Fusion
(k=60), with post-retrieval metadata filters. Serves the corpus exported by
the Laravel app's `ai:export-corpus` command.

## Requirements

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
cp .env.example .env   # fill SIDECAR_TOKEN (must match Laravel's SIDECAR_TOKEN)
```

First run downloads the ~470 MB embedding model (`paraphrase-multilingual-MiniLM-L12-v2`).
Pre-warm it on first ingest; subsequent builds use the local cache.

## Run

```bash
uv run uvicorn app.main:app --port 8310
```

The server binds **loopback only** (`127.0.0.1`) when run locally — it is never
exposed to the network from a dev machine. Production runs on **FastAPI Cloud**
via `fastapi run` (requires `fastapi[standard]`), bound to the platform's public
endpoint behind HTTPS; the shared `X-Sidecar-Token` header is the auth boundary
there.

## Endpoints

| Method | Path             | Description                                   |
|--------|------------------|-----------------------------------------------|
| GET    | `/health`        | Index coverage + staleness (`documents == embedded`) |
| POST   | `/search`        | Hybrid RRF search with optional filters       |
| POST   | `/chat/stream`   | SSE-streamed RAG answer over search results (Phase 9, ADR 0002) |
| POST   | `/index/rebuild` | Synchronous full rebuild (atomic swap, no downtime) |
| GET    | `/metrics`       | Minimal hand-rolled counters (Prometheus in Phase 14) |

All endpoints require `X-Sidecar-Token`; requests without it get `401`.

`POST /chat/stream` request: `{"query": "...", "mode": "citations"|"question"|"rag", "corpus"?: "catalog"|"policy", "top_k"?: 5}`.
Response is `text/event-stream`: `data: <chunk>` lines, terminated by `data: [DONE]`; provider failures surface as an `event: error` line before `[DONE]`.
The LLM provider is OpenRouter via the openai SDK (`LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`, default `meta-llama/llama-3.3-70b-instruct`).

## Corpus source

`CORPUS_PATH` points at the directory containing `catalog.json` + `policies.json`
(exported by the Laravel app's `ai:export-corpus`). Default:
`../ceit-library/storage/app/ai-corpus`.

## Tests

```bash
uv run pytest -q
uvx ruff check .
```
