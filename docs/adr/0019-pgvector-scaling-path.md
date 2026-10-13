# ADR 0019: pgvector scaling path — pooled connections, latency budget, known cliffs

Status: accepted

## Context

The pgvector backend deliberately ships with exact, unindexed scans: at
~1,338 documents a sequential scan is fast and returns the same ranks as
the NumPy backend, which keeps the eval gate comparison meaningful
(issue #2: "No ANN index in v1"). The same spec asks for the scaling path
to be written down so a growing catalog triggers a planned change, not a
surprise: beyond roughly 20–50k documents, move to truncated candidate
pools plus an HNSW index and re-run the gate, because truncation
legitimately changes fused ranks.

## Decision

### Pooled connections on the query path only

`PgVectorSearch` holds a `psycopg_pool.ConnectionPool` (`min_size=1`,
`max_size=PG_POOL_SIZE`, default 4) so search never pays a connect
handshake per request. The pool opens lazily (`open=False`) —
constructing the engine must not need the database, so a sleeping Neon
compute or a dead URL fails on first use, not at startup. Waiting for a
connection is bounded by `PG_CONNECT_TIMEOUT` (15 s); the pool's own
30 s default would outlive the request.

`PgIndexStore` (sync, health) connects per call instead — writes are
infrequent, and per-call connections keep `/health` honest about reach
without holding pool state.

All connections set `prepare_threshold=None`: pooled endpoints run
transaction pooling, which cannot serve prepared statements across
pooled transactions. Object names are schema-qualified via
`psycopg.sql.Identifier` — never `search_path`, a startup parameter
poolers reject.

### Neon autosuspend

The first request after idle pays a compute wake. `load_state` issues a
trivial `SELECT 1` and `PG_CONNECT_TIMEOUT` covers the wake, so the
health check warms the database before real traffic hits (issue #2).

### Latency and memory budget

The issue #9 flip gate budgets warm p95 search latency at **baseline +
50 ms** and sidecar RSS at **< 400 MB measured on the real host**.
Baseline bench (SQLite, in-process, `docs/baseline/latency-memory.json`):
p50/p95 hybrid 14.1/28.3 ms, working set ~200 MB. The pgvector search
path issues several SQL round-trips per query (state read, gated
semantic scan, BM25 transaction, document fetch), so network RTT to the
database is the dominant new term; `PG_POOL_SIZE` keeps it from also
paying connection setup.

pgvector does not meaningfully reduce runtime RAM at this scale — the
vectors are ~2 MB. The memory win from the migration is incremental
ingest and offline re-embeds (ADR 0018), not serving memory; the 400 MB
check is a real-host measurement, not a dev-bench number.

### Known cliffs and the planned response

| Limit | Where | Response |
|---|---|---|
| Semantic channel scans every row with exact cosine (`embedding <=>`) per query | `PgVectorSearch._semantic_scores` | At ~20–50k docs, switch to truncated candidate pools + HNSW and re-run the eval gate — truncation legitimately changes fused ranks |
| `lakebase_bm25.default_limit` hard cap of 65,535 matching rows per query | `SET LOCAL` in `_bm25_ranks` (extension rejects -1 and ≥65536) | Same candidate-pool change; a broader match truncates silently today |
| Connection count is per process | `PG_POOL_SIZE` | Horizontal replicas multiply connections against Neon's limit — budget `PG_POOL_SIZE` × replicas |
| `lakebase_text` is Neon-only | `index_state.keyword_index` | Off Neon the channel drops to the `gin` fallback — functional but not score-identical (ADR 0017); re-run the gate on the new host |

Watch `ceit_search_duration_seconds` p95 against the +50 ms budget and
`ceit_index_documents` against the ~20–50k trigger; either one arriving
first schedules the candidate-pool work.

### Alternatives rejected

- **HNSW + candidate pools now** — approximate retrieval and truncated
  pools change fused ranks at 1.3k docs for no measurable latency win,
  and would make the parity gate compare two different algorithms
  instead of two stores.
- **Prepared statements / `search_path`** — incompatible with
  transaction-mode pooling; rejected by the deployment target, not on
  taste.
- **Pool on the write path too** — sync and health are rare; pooling
  them would hold idle connections and complicate the advisory-lock
  story (session vs transaction scoping, ADR 0018) for no benefit.
