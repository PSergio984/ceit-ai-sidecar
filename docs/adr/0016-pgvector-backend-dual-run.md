# ADR 0016: pgvector backend behind `VECTOR_BACKEND`, dual-run with SQLite

Status: accepted

## Context

The SQLite backend stores the search index as versioned files committed to
git (`index-<version>.db`, `vectors-<version>.npy`, `docs-<version>.json`),
and every corpus push rebuilds the whole index locally — the 500 MB cloud
host cannot re-embed the corpus itself (issue #2). The spec moves index
storage and retrieval to Postgres on Neon (`vector` for embeddings,
`lakebase_text` for BM25) under two hard requirements: SQLite stays
permanently supported for local dev and tests, and the switch must be a
single reversible env change so a bad flip rolls back instantly (issues #2,
#9).

`lakebase_text` is Neon-specific — Neon retired `pg_search` on 2026-09-21 —
so Postgres parity alone does not remove the need for the SQLite backend;
it is also the portability path for any non-Neon host (ADR 0017).

ADR 0015 anticipated this seam: "a future pgvector backend implements the
same signature over its own vector store, so the endpoint contract survives
the backend swap."

## Decision

Two backend-selecting factories, both keyed on `VECTOR_BACKEND`
(`sqlite` default | `pgvector`; `app/config.py`):

- `create_index_store(settings)` (`app/index_store.py`) — storage, sync and
  health state. Returns `SqliteIndexStore` (delegates to the existing
  versioned-file `build_index`) or `PgIndexStore` (`app/pg_index.py`).
- `create_backend(cache_dir, model_name)` (`app/backend.py`) — the query
  seam. Returns the unchanged `HybridSearch` or `PgVectorSearch`
  (`app/pg_search.py`) behind the `SearchBackend` Protocol (`rrf_search`,
  `recommend`, `close`).

Every construction site — the API, agent loop, eval and judge — goes
through the factory, so the reranker, query rewriter and agent loop are
untouched (issue #7 acceptance). `sqlite` remains the default until the
eval gate in issue #9 passes; both backends are kept permanently.

Selecting `pgvector` without `DATABASE_URL` raises at factory time;
`/health` answers `degraded` with the `backend` field set rather than
500ing.

### Settings

`DATABASE_URL` (Neon pooled endpoint), `PG_INDEX_SCHEMA` (default
`public`; schema-qualified `psycopg.sql.Identifier` names, never the
`search_path` startup parameter poolers reject), `PG_CONNECT_TIMEOUT`
(15 s default; bounds `/health` and sync against an unreachable or
sleeping database), `PG_POOL_SIZE` (4 default; search-path pool, ADR 0019).

### Shared fusion, single semantics

`app/fusion.py` owns RRF k=60 (`fuse`), the exact-code pin
(`pin_catalog_code`), result shaping (`shape_results`) and
`recommend_rank`. Both backends call it, so fused scores, pinning and the
result shape cannot drift between backends.

`PgVectorSearch.rrf_search` preserves the SQLite semantics exactly:

- BM25 over **all** matching documents with an OR of query tokens; a query
  with no alphanumeric token produces no keyword channel at all.
- The relevance gate spans the whole **unfiltered** index: max cosine over
  every row < `MIN_SEMANTIC_SIMILARITY` (0.5) drops the semantic channel,
  so off-corpus queries return nothing instead of nearest neighbours.
- Filters apply per channel **before** ranking. SQLite filters
  post-retrieval in Python and re-indexes ranks contiguously; Postgres
  pushes the same predicates into the channel's `WHERE` so ranks are
  computed over the filtered set in one pass. `_filter_sql` mirrors
  `passes_filters`: `paper_type`/`department` exact match on `metadata`
  jsonb; `publication_year` via a CASE-guarded `::int` cast (digits regex
  only — a non-numeric year yields NULL, `COALESCE 0`, matching sqlite's
  `or 0`: `year_to` keeps year-less docs, `year_from` drops them, equality
  excludes them); `author`/`adviser` case-insensitive substring via `ILIKE`
  with escaped wildcard characters; `corpus` equality.
- The exact-code pin is hybrid-only and applies to surviving candidates;
  unknown `method` falls back to `hybrid`.
- `recommend` (issue #11) mirrors `fusion.recommend_rank` in SQL: catalog-
  only seed resolution, float32 mean, L2 normalize, cosine order with `id`
  tie-break, seeds and `exclude_ids` excluded.

### Schema

`documents(id, corpus, title, text, metadata jsonb, content_hash,
embedding vector(384), tsv generated stored)` + a `corpus` index + a
single-row `index_state` table (version, built_at, source_generated_at,
documents, embedded, by_corpus, model_name, contract_version,
keyword_index). `CREATE EXTENSION IF NOT EXISTS vector`; `lakebase_text`
only when available. `/health` reads `load_state()` from whichever store
is active and reports the store's `name` as `backend`.

### Tests

Postgres tests run in CI against a `pgvector/pgvector:pg17` service
container via `TEST_DATABASE_URL` and skip locally when it is unset; the
same search contract fixtures run parametrized over both backends so the
suite stays runnable without a database.

### Alternatives rejected

- **Replace SQLite wholesale** — loses the portable dev/test path and the
  instant-rollback story the spec requires; `lakebase_text` being
  Neon-only makes a Postgres-only world non-portable.
- **A flag inside `HybridSearch`** — two storage engines behind one class
  instead of one seam; the sqlite implementation stays untouched this
  way, and the same contract tests prove both.
- **Per-backend endpoints** — the point of the seam is that `/search`,
  `/recommend`, chat retrieval and the eval harness see one contract;
  callers must not know which store answers.
