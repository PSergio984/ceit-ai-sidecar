# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: Service modules with a thin HTTP shell; search is a rank-fusion pipeline over versioned, file-based index artifacts.
- Why this classification: `app/main.py` is a thin FastAPI layer; all logic lives in `search`/`ingest`/`rebuild`/`rag`/`agent` modules; the index is a set of versioned files swapped atomically (`os.replace`), not a running DB service.
- Primary constraints:
  1. **Files only** — the sidecar never touches the Laravel database (D-17); everything derives from the exported corpus JSON.
  2. **Grounding** — answers come only from retrieved documents; empty retrieval → canonical refusal with zero LLM calls (ADR 0006).
  3. **Closed contracts** — every wire payload is validated (pydantic `extra="forbid"`), SSE framing is additive, and citation keys mirror Laravel's `AiService` constants.

### 2) System Flow

```text
export:   Laravel ai:export-corpus -> storage/app/ai-corpus/{catalog,policies}.json
          (cloud: Laravel ai:push-corpus hourly -> POST /corpus/upload -> files land in CORPUS_PATH)
rebuild:  /index/rebuild -> load_documents (validate) -> embed (SentenceTransformer)
          -> build FTS5 index -> write versioned artifacts (docs-N.json, vectors-N.npy, index-N.db)
          -> atomic state.json swap -> prune old versions (keep 2)
search:   /search -> HybridSearch.rrf_search:
          BM25 ranks (FTS5, retrieve-all) ∪ semantic scores (normalized cosine matmul)
          -> post-retrieval metadata filters -> RRF k=60 fusion
          -> code-exact pin (CEIT-XX-NN[-N] -> rank 1) -> top-limit
chat:     /chat/stream -> AgenticLoop:
          LLM round 1 (tools=[search], tool_choice=auto)
          - no tool call -> direct answer (no frames; Laravel falls back to companionCitations)
          - tool call(s) -> validate args -> activity frame -> rrf_search -> tool result
          -> loop (MAX_TOOL_ROUNDS=3 executed searches)
          -> final streamed answer grounded in merged deduped docs
          -> event: citations (numbered 1..N) -> data: [DONE]
          empty docs -> refusal; provider error -> event: error -> [DONE]
```

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `main.py` | HTTP endpoints, `X-Sidecar-Token` gate (timing-safe), payload key allow-lists, `/metrics` counters | Ranking, prompts | `app/main.py` |
| `search.py` | BM25 + semantic retrieval, RRF fusion, filters, code pin | LLM calls | `app/search.py` |
| `ingest.py` | Corpus schema validation (fail-closed, T-04), embeddings, cache writes | Serving queries | `app/ingest.py` |
| `rebuild.py` | Full rebuild pipeline, atomic swap, global lock, injectable embedder (tests) | Query paths | `app/rebuild.py` |
| `rag.py` | Prompt modes (rag/citations/question), context truncation (600 chars), SSE chunk framing, refusal | Tool-loop orchestration | `app/rag.py` |
| `agent.py` | Tool schema + arg validation, loop cap, activity/citations frames, dedupe merge | HTTP surface | `app/agent.py` |
| `eval.py` | Golden-set scoring | Production endpoints | `app/eval.py` |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| Versioned cache artifacts + atomic swap | `rebuild.py` (`os.replace` on `state.json`), `ingest.py` `write_cache` | Readers never observe a half-built index; rollback = keep previous version |
| Global lock around rebuild | `rebuild.py` `_lock` | Idempotent per-flight rebuilds |
| Lazy singleton embedder (thread-safe) | `ingest.py` `get_embedder` | Compact English model loaded once |
| Injectable collaborators for tests | `rebuild._embed_override`, `AgenticLoop`/`RagService` constructor injection | Deterministic fast tests without HuggingFace/OpenRouter |
| Injectable client pattern (Laravel-style) | `AgenticLoop`, `RagService` (`client`, `base_url`, `api_key` args) | Testability; same shape as Laravel DI |
| Mirror constants across repos | `rag.CHUNK_KEY`/`CITATION_KEYS` ↔ `AiService::SSE_CHUNK_KEY`/`CITATION_KEYS` | Contract drift prevention |
| Pydantic closed schemas | `ToolArgs`/`ToolFilterArgs` (`extra="forbid"` both levels) | LLM-produced args validated before execution (D-09) |

### 5) Known Architectural Risks

- Whole-document embeddings with `MAX_DOC_CHARS=600` prompt truncation lose long-document detail (R6 decision: no sentence chunking).
- BM25 retrieve-all (`num_results=1_000_000`) is fine at corpus scale but has no candidate-pool bound (documented trade-off).
- Single-instance, in-process state (`_search_engine`, `_rag`, `_agent` globals) — no horizontal scaling; rebuild lock serializes rebuilds.
- `AgenticLoop` no-opens history over the wire; multi-turn memory is Laravel's job (Conversation/Message schema).
- No rate limits or cost guards on the LLM path (Phase 14).

### 6) Evidence

- `app/search.py`, `app/agent.py`, `app/rebuild.py`, `app/rag.py`, `app/eval.py`, `app/main.py`
- `README.md`, `docs/adr/` mirror contracts in the Laravel repo

## Extended Sections (Optional)

### Search internals — direct answers (BM25 vs TF-IDF vs vector)

- **BM25 is used; TF-IDF is not used anywhere.** Keyword retrieval is SQLite FTS5 via `sqlitesearch.TextSearchIndex` (`text_fields=["text"]`, keyword fields `corpus`/`department`/`paper_type`); FTS5's default ranking is BM25. Source: `app/search.py` `_bm25_ranks` → `db.search(query, ...)`.
- **Vector search IS used**: whole-document embeddings from `sentence-transformers` (`all-MiniLM-L6-v2`), normalized, cosine similarity via matmul (`vectors @ q`). Source: `app/search.py` `_semantic_scores`, `app/ingest.py` `embed_documents`.
- **Fusion is RRF (Reciprocal Rank Fusion) with k=60**: each document scores `1/(60+rank)` per list (semantic rank derived from score ordering). Post-retrieval metadata filters (paper_type, department, publication_year, year_from/to, author, adviser) are applied to the merged candidate set BEFORE fusion, so filtered docs cannot outrank relevant ones (D-03). Exact `CEIT-XX-NN[-N]` catalog codes pin the matching doc to rank 1 (D-02).
- **Evals**: `app/eval.py` scores the golden set (`data/golden_dataset.json`, 27 cases) with precision@k/recall@k/F1 (k default 5), top-1 rate, negative pass rate, and per-category breakdowns (catalog_code, exact_title, paraphrase, people). Run via `uv run python -m app.eval [--json] [--corpus catalog|policy|all] [--limit N]`.
