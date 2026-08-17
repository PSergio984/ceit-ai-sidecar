# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| High | OpenRouter API key (`LLM_API_KEY`) exposed in an earlier chat session; lives only in `.env` (gitignored) | session handoff docs | Unauthorized spend on the account | Rotate the key; never commit or echo it |
| High | `SIDECAR_TOKEN=smoke-test-token` placeholder | `.env`, `.env.example` | Any local process can call the sidecar | Set a real token (shared with Laravel) before production |
| Med | No rate limits or cost guards on the LLM path | `.planning/ROADMAP.md` Phase 14 (OPS-02) | Runaway token spend | Phase 14: rate limits + cost guards |
| Med | Corpus staleness if Laravel scheduler/queue isn't running (exports never land) | Laravel `routes/console.php` schedule | `/health` reports stale index; search serves old data | Health alerting on `source_generated_at` (Phase 14) |
| Med | Corpus must exist inside the FastAPI Cloud container | `CORPUS_PATH` (default `../ceit-library/storage/app/ai-corpus` — Laravel-relative, meaningless in the cloud) | Cloud index empty/degraded until a corpus is provided | **Resolved 2026-08-16**: Laravel `ai:push-corpus` (hourly :07) uploads to `POST /corpus/upload`; set `CORPUS_PATH` in the dashboard to the container path (e.g. `corpus`). Corpus still never committed to git (contains author names) |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| Whole-document embeddings, no chunking (R6) | Phase 8 decision for simplicity + scale | `app/ingest.py` `embed_documents` | Long papers lose detail; `MAX_DOC_CHARS=600` prompt truncation compounds it | Evaluate chunking when corpus grows (ADR 0013 paper shape is the trigger point) |
| BM25 retrieve-all (`num_results=1_000_000`) | Deliberate, documented ("no magic limit*500 pool") | `app/search.py` `_bm25_ranks` | O(corpus) per query; degrades at scale | Reintroduce a bounded, tuned candidate pool |
| Hand-rolled `/metrics` counters | Prometheus is Phase 14 | `app/main.py` | No standard exporter, no dashboards | Phase 14: Prometheus format + Grafana |
| Golden set is small (35 cases) | Bootstrap golden set from Phase 8 | `data/golden_dataset.json` | Metrics have wide confidence intervals | Phase 13: grow golden sets + LLM-as-judge + user feedback loop |
| `test_chat_stream_live.py` gated behind env | Needs real key + running server | `tests/test_chat_stream_live.py` | Live provider behavior untested in CI | Manual smoke per release (documented in phase research) |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| Weak shared token | A07 | `.env` (placeholder) | Timing-safe compare (`secrets.compare_digest`) in middleware | Real token rotation required — **the endpoint is now public on FastAPI Cloud behind HTTPS** (no longer loopback-only); a leaked `smoke-test-token` would expose search/chat to anyone |
| LLM prompt injection via retrieved docs | A01 | `rag.py` SYSTEM_PROMPT grounding rules | Grounding instructions + programmatic empty-retrieval refusal | No adversarial-doc eval set (Phase 13) |
| Tool-arg injection from the model | A03 | `ToolArgs`/`ToolFilterArgs` pydantic `extra="forbid"` both levels | Strict validation before execution (D-09) | None identified |
| Secret in artifacts | A05 | sonar-secrets workflow | Offline secrets scan every push | — |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| Embedding model load + lazy singleton | `app/ingest.py` `get_embedder` | First query after boot pays model load; memory still depends on runtime | Multiple instances multiply RAM | Keep single instance; use the compact English model; Phase 14 deploy as one service |
| Cosine scan over all vectors per query | `app/search.py` `_semantic_scores` (full matmul) | Fast at ~100s of docs | O(n) per query | FAISS/ANN index when corpus grows (Phase 14) |
| Rebuild blocks other rebuilds (global lock) | `app/rebuild.py` `_lock` | Sequential rebuilds | Multiple rebuild triggers queue | Keep; debounce via Laravel `ShouldBeUnique` jobs |
| In-process state (`main.py` globals) | `app/main.py` | None at scale | No multi-worker warmup; each worker duplicates model | Phase 14 deployment decision |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| SSE framing (`rag.py`/`agent.py`) | Wire contract mirrored in Laravel `AiService` | Mirror constants + additive frame types | Change frames additively; update both test suites together |
| Agentic loop (tool schema, caps, malformed handling) | LLM behavior variability | `MAX_TOOL_ROUNDS`, malformed-streak logic, parallel tool-call handling | Fake-client tests before touching provider calls |
| Versioned index artifacts | Swap/prune ordering errors corrupt serving | `keep=2` pruning, `os.replace` path | Extend `test_ingest`/`test_api` rebuild coverage on swap failure |
| Filter semantics | Filters apply pre-fusion (D-03) | `test_filters.py` | Keep filter-before-fusion invariant in any refactor |

### 6) `[ASK USER]` Questions

1. [ASK USER] Should chunking be revisited (ADR 0013 paper documents are long) — now or in Phase 13?
2. [ASK USER] Is an ANN index (FAISS) wanted at current corpus size, or only if it grows?
3. [ASK USER] Should the live chat smoke (`SIDECAR_LIVE_CHAT_TEST=1`) run as a manual release checklist item?

### 7) Evidence

- `.codebase-scan.txt` (CODE METRICS section)
- `app/search.py`, `app/ingest.py`, `app/rebuild.py`, `app/main.py`, `app/agent.py`
- `.env` (gitignored), `.env.example`, `.github/workflows/*.yml`
- `data/golden_dataset.json` (35 cases)
