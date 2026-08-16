# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `main.py` | Entry point (FastAPI app, endpoints, token middleware) | `main.py` |
| `app/` | Package: search, ingest, rebuild, rag, agent, eval, health, config | `app/` |
| `app/search.py` | `HybridSearch` — BM25 + semantic + RRF fusion, filters, code-pin | `app/search.py` |
| `app/ingest.py` | Corpus loading/validation, embedding (whole-document), versioned cache writes, model singleton | `app/ingest.py` |
| `app/rebuild.py` | Full index rebuild with atomic state swap, `_embed_override` test hook | `app/rebuild.py` |
| `app/rag.py` | `RagService` one-shot/streamed RAG, prompt modes, SSE chunk framing, citation keys | `app/rag.py` |
| `app/agent.py` | `AgenticLoop` — function-calling search loop over `/search` contract, SSE activity/citations frames | `app/agent.py` |
| `app/eval.py` | Golden-set evaluation runner (P@k / R@k / F1, top-1, negative pass rate, categories) | `app/eval.py` |
| `app/health.py` | `/health` assembly (index coverage + staleness) | `app/health.py` |
| `app/config.py` | pydantic-settings `Settings` | `app/config.py` |
| `tests/` | pytest suite (9 files incl. `conftest.py`; `test_chat_stream_live.py` env-gated) | `tests/` |
| `data/golden_dataset.json` | Golden eval set: 35 cases (30 catalog, 4 policy, 1 all; 5 negatives) | `data/` |
| `cache/` | Runtime index artifacts (versioned; gitignored) | `cache/` |
| `.github/workflows/` | `python-app.yml` (lint/test/smoke), `sonar-secrets.yml`, `sonarcloud.yml` | `.github/workflows/` |

### 2) Entry Points

- Main runtime entry: `main.py` — `app = FastAPI(...)`; run via `uv run uvicorn app.main:app`
- Secondary entry: `python -m app.eval` (evaluation CLI)
- Entry selection: uvicorn command line; eval via `__main__` guard in `app/eval.py`

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `app/search.py` | Ranking + fusion + filters (one seam for search) | LLM calls, prompt text |
| `app/rag.py` / `app/agent.py` | LLM orchestration, SSE framing, prompts | Index format/versioning details |
| `app/ingest.py` / `app/rebuild.py` | Corpus reading, embedding, cache artifact writes, atomic swaps | HTTP/API concerns |
| `app/main.py` | HTTP surface, token gate, request validation | Ranking logic |
| `app/eval.py` | Golden-set scoring | Production serving paths |

### 4) Naming and Organization Rules

- Files: snake_case modules (`search.py`, `agent.py`); classes PascalCase (`HybridSearch`, `AgenticLoop`, `RagService`)
- Type hints mandatory (`from __future__ import annotations`); pydantic models for any parsed payload
- Decision references inline in docstrings (`D-02`, `D-03`, `D-07`, `D-09`, `D-11`, `D-12`, `D-17`, `R6`, `T-04`, `T-11-11`, `WR-1`, `WR-2`) — traceable to ADR 0001-0014 + phase review items
- Contract mirrors: `CHUNK_KEY = "c"` ↔ Laravel `AiService::SSE_CHUNK_KEY`; `CITATION_KEYS` ↔ `AiService::CITATION_KEYS`

### 5) Evidence

- `.codebase-scan.txt` (directory tree)
- `main.py` (entry), `app/search.py`, `app/agent.py`, `app/eval.py`
- `README.md`
