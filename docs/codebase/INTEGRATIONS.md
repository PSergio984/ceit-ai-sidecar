# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System | Type (API/DB/Queue/etc) | Purpose | Auth model | Criticality | Evidence |
|--------|---------------------------|---------|------------|-------------|----------|
| CEIT-Library (Laravel) | HTTP consumer (POST /search, /chat/stream, /index/rebuild, GET /health, /metrics) | Serves the Laravel app's AI surfaces | Shared `X-Sidecar-Token` header (both `.env` files). Locally over loopback; **in production the Laravel app calls the FastAPI Cloud HTTPS URL** (`SIDECAR_URL` in Laravel `.env`) | High | `app/main.py` token middleware |
| OpenRouter | LLM API via openai SDK | Chat/generation | `LLM_API_KEY` (sidecar `.env` only, gitignored) | High | `app/config.py`, `app/rag.py` |
| HuggingFace (model hub) | Model download (first run) | Embedding model `all-MiniLM-L6-v2` | none | Medium (cold start) | `app/ingest.py` `get_embedder` |
| Corpus files | Filesystem hand-off + HTTP upload | Reads `catalog.json` + `policies.json` from `CORPUS_PATH`; cloud deployments receive them via `POST /corpus/upload` from Laravel `ai:push-corpus` | Shared token header; files only (D-17) | High | `app/ingest.py`, `app/main.py` (`corpus_upload`) |
| SonarCloud | CI quality gate | Static analysis (Python 3.13, coverage) | `SONARQUBE_TOKEN` repo secret | Med (CI only) | `.github/workflows/sonarcloud.yml`, `sonar-project.properties` |
| SonarQube CLI | CI secrets scan | Secrets detection (offline) | `SONARQUBE_TOKEN` repo secret | Med (CI only) | `.github/workflows/sonar-secrets.yml` |

### 2) Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|-------|------|--------------|----------|----------|
| Versioned index cache (`cache/`) | Search index: `index-N.db` (FTS5), `docs-N.json`, `vectors-N.npy`, `state.json` | `HybridSearch._ensure_db` / `_version_artifacts` | Stale/corrupt artifacts if a swap fails; version GC keeps 2 | `app/search.py`, `app/rebuild.py` |
| Corpus JSON (external, read-only) | Source of truth for index | `load_documents` | Schema drift vs exporter (`schema_version: 1` validation) | `app/ingest.py` |
| `data/golden_dataset.json` | Eval gold set | `app/eval.py` | Small set (27 cases) limits metric confidence | `app/eval.py`, `data/golden_dataset.json` |

### 3) Secrets and Credentials Handling

- Credential sources: `.env` only (gitignored) for local; **FastAPI Cloud dashboard env vars for production** (`SIDECAR_TOKEN`, `LLM_API_KEY`, `CORPUS_PATH`, optional `LLM_MODEL`/`LLM_MAX_TOKENS`); CI repo secrets (`SONARQUBE_TOKEN`)
- Hardcoding checks: sonar-secrets workflow passes on every push (no findings)
- Rotation notes: `SIDECAR_TOKEN` placeholder `smoke-test-token` in `.env` — rotate before production; `LLM_API_KEY` (OpenRouter) was exposed in an earlier chat session — **must be rotated**; never echo either in artifacts

### 4) Reliability and Failure Behavior

- Retry/backoff: none on the LLM path (provider errors become `event: error`); Laravel retries `/chat/stream` with `retries: 0`
- Timeout policy: Laravel `AiService` uses 120s for `/chat/stream`; sidecar has no own outbound timeouts except openai SDK defaults
- Circuit-breaker/fallback: none; refusal is the only fallback (grounding rule)
- Rebuild atomicity: versioned artifacts + `os.replace` state swap; readers never see a half-built index; old versions pruned (keep=2) after swap

### 5) Observability for Integrations

- Logging around external calls: provider errors logged (`logger.error(repr(exc))`) before the `event: error` SSE frame; `main.py` tracks `/metrics` counters (searches_total, rebuilds_total, search_avg_ms, index_documents)
- Metrics/tracing: hand-rolled counters only; no Prometheus export, no tracing, no LLM cost tracking (Phase 14: OPS-01/02) [TODO]
- Missing visibility gaps: per-query latency percentiles, token usage per request, corpus freshness alerts

### 6) Evidence

- `app/main.py` (endpoints, token gate, metrics)
- `app/config.py`, `.env.example`
- `app/ingest.py`, `app/rebuild.py`, `app/search.py`
- `.github/workflows/*.yml`, `sonar-project.properties`
