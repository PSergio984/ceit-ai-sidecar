# ADR 0015: POST /recommend — seed-based catalog recommendations

Status: accepted

## Context

Laravel needs a "more like this" call so a catalog detail page can list related
papers before pgvector is live (issue #10). The index already stores per-document
embeddings under the SQLite backend (`vectors-<version>.npy`), so similarity
recommendations need no model call, no BM25, and no LLM.

## Decision

Add `POST /recommend`, token-gated like `/search`, backed by
`SearchBackend.recommend` on the SQLite backend (`HybridSearch`).

### Contract

- Closed key set — unknown keys get the standard 422 envelope:
  - `seed_ids` (required): list of 1–20 non-empty strings.
  - `exclude_ids` (optional, default `[]`): list of 0–200 strings.
  - `limit` (optional, default 10): integer 1–50.
- Response: `{"seeds_used": int, "total": int, "took_ms": int, "results": [...]}`;
  each result is `{id, corpus, title, score, metadata}` with `score` rounded to
  4 decimal places.

### Ranking

- A seed **resolves** only if its id exists in the index AND its corpus is
  `catalog`. Unknown ids and `policy` docs are ignored; `seeds_used` counts
  distinct resolved seeds. Zero resolved seeds → `200` with `seeds_used: 0`
  and an empty list — not a 404, because a miss on one seed must not fail the
  request and callers can't know which ids are indexed.
- The query vector is the L2-normalized **mean of the resolved seed
  embeddings**; a zero mean yields an empty result.
- Candidates: `catalog` docs only, minus every requested seed id and every
  `exclude_ids` entry. Score is cosine similarity (stored vectors are
  normalized). Ordering is `(-score, id)`, so results are fully deterministic
  and repeated identical calls return identical lists.

### Why semantic-only (no BM25, no LLM)

Recommendations are "similar to these papers", not "match this text" — there is
no query string to feed FTS5, and RRF fusion has no second channel to fuse.
Rewriter/reranker/LLM stages are skipped for the same reason `/search` uses
them: there is no natural-language query to rewrite or re-rank against, and the
response must stay deterministic and cheap.

### Backend seam

`recommend` joins `rrf_search` on the `SearchBackend` Protocol. The SQLite
implementation reads stored vectors directly (no embedding call); a future
pgvector backend implements the same signature over its own vector store, so
the endpoint contract survives the backend swap.

### Metrics

`ceit_recommend_total` (counter) and `ceit_recommend_duration_seconds`
(histogram, same buckets as search) count only requests that passed validation.
