# Rollout: pgvector backend on Neon (issue #9)

Production rollout and rollback runbook for switching the sidecar's
retrieval backend from SQLite to Postgres/pgvector. Related decisions:
ADR 0016 (dual-run seam), ADR 0017 (BM25 parity), ADR 0018 (incremental
ingest + push), ADR 0019 (scaling path).

## Gate the flip is conditioned on

Do not flip until the gate report against the frozen baseline
(`docs/baseline/`) passes every criterion:

- top-1 rate and negative-pass rate **at least** baseline;
- average recall and F1 at k=5 within **0.02 below** baseline, overall
  and per category;
- judge relevant rate within the measured noise band (two baseline runs
  differed by 0.025; a change under ~0.05 is not a regression);
- warm p95 search latency within **50 ms** of baseline;
- sidecar memory under **400 MB on the real host** — a real-host
  measurement, not the dev bench.

If any criterion fails, the report says which and the default backend
stays `sqlite`. Reproduce the golden side with
`PYTHONHASHSEED=0 MIN_SEMANTIC_SIMILARITY=0.5 python -m app.eval
--with-rerank --json` (see `docs/baseline/README.md`).

## 1. Create the Neon project and branch

Human prerequisite: the maintainer creates the Neon project and supplies
the connection strings as environment secrets — never committed.

```sh
neon link --project-id <id> --branch production
neon connection-string production --pooled    # -> DATABASE_URL for the app
neon connection-string production             # -> direct endpoint, for the push
```

Use a Postgres version with `vector` and `lakebase_text` available — the
spike ran on Postgres 18.6, `lakebase_text` 0.1.3, region ap-southeast-1.

## 2. Full initial load

Run the offline embed-and-push from any machine that can reach Neon:

```sh
DATABASE_URL=<direct endpoint> uv run --group offline \
    python scripts/push_embeddings.py <corpus>
```

`<corpus>` is the export directory holding `catalog.json` /
`policies.json`. `PG_INDEX_SCHEMA`, `MODEL_NAME`, `PG_CONNECT_TIMEOUT`
are read from the environment like the app. `--force-full` re-embeds
everything; an unchanged corpus is a no-op.

**Pooled vs direct:** use the **direct** endpoint here, not the
`-pooler` one — the push takes a *session-scoped* advisory lock around
the schema ensure, data transaction and post-commit `CREATE INDEX` /
`VACUUM`, and session state does not survive transaction pooling. The
in-app sync (`POST /index/rebuild`) uses a transaction-scoped lock and
is the alternative path if only the pooled endpoint is reachable.

Expected output:
`pushed: version=1 documents=1338 embedded=1338 (added=1338 changed=0 removed=0)`.

## 3. Deploy the sidecar — dual-run posture

Deploy with the pgvector settings present but the backend still SQLite:

```
DATABASE_URL=<pooled endpoint>     # pooled: search path uses a conn pool
VECTOR_BACKEND=sqlite              # unchanged — traffic keeps using sqlite
# PG_INDEX_SCHEMA=public  PG_CONNECT_TIMEOUT=15  PG_POOL_SIZE=4
```

Traffic is unchanged; the Postgres index sits loaded underneath.

## 4. Smoke test (pre-flip)

`GET /health` (with `X-Sidecar-Token`):

- `status: ok`
- `backend: sqlite` — storage/search still served by SQLite
- `index.documents` / `index.embedded`: 1338 and equal (healthy rule)
- `index.by_corpus`: `{"catalog": 959, "policy": 379}`

Verify the Postgres side directly — `index_state.keyword_index` is **not**
exposed on `/health`; check it in the database:

```sh
psql "$DATABASE_URL" -c "SELECT keyword_index, version, documents, embedded FROM index_state WHERE id = 1"
-- expect keyword_index = lakebase_bm25 on Neon (gin on plain pgvector)
```

## 5. Real-host gate measurements — before any traffic switches

The dev-machine gate report (`docs/baseline/pgvector-gate.md`) leaves two
criteria unanswered by design: warm p95 latency and RSS are real-host
numbers. Measure them on the cloud host **without switching production
traffic** — e.g. a second sidecar instance (or container replica) deployed
with `VECTOR_BACKEND=pgvector`, `DATABASE_URL`=<pooled endpoint> and the
same `PG_INDEX_SCHEMA`, not attached to the Laravel app:

```sh
# on that pgvector instance, from the host:
PYTHONHASHSEED=0 MIN_SEMANTIC_SIMILARITY=0.5 \
  python -m app.eval --json          # sanity: quality matches the gate report
python docs/baseline/bench.py        # warm p95 per method + RSS
```

Requirements to proceed (from the gate AC):

- warm p95 within **50 ms** of the SQLite baseline on the same host —
  re-bench the sqlite instance for a like-for-like number;
- RSS under **400 MB**;
- the maintainer's decision on the top-1 label ambiguity (issue #26) —
  either accept the corrected 0.7727 reference or regenerate the golden
  set first.

If any of these fail, stop here: production stays on `sqlite` and the
report records which criterion failed.

## 6. Flip — human step

The maintainer sets `VECTOR_BACKEND=pgvector` on the cloud host and
restarts/redeploys. This is deliberately manual: the gate report, the
smoke test and the real-host measurements are reviewed first.

## 7. Smoke test (post-flip)

- `GET /health` → `backend: pgvector`, `status: ok`, same
  `documents`/`embedded`/`by_corpus` as above.
- `POST /search` — a catalog-code query (e.g. `CEIT-EE-…`) returns the
  paper first (code pin); an off-corpus query returns an empty list
  (relevance gate); a filtered query honours the filters.
- `POST /chat/stream` — grounded answer with numbered citations.
- `POST /recommend` — `{"seed_ids": ["<catalog id>"]}` returns ranked
  catalog papers, `seeds_used: 1`.

## 8. Rollback

Set `VECTOR_BACKEND=sqlite` and restart. **Rollback is a single env
change** — that is an acceptance criterion, not a fallback plan. The
committed SQLite index artifacts are untouched by the migration, so the
old backend serves immediately; the Postgres index can stay in place for
a later re-flip (re-run `push_embeddings.py` first if the corpus moved).

## 9. Watch post-flip

- `ceit_search_duration_seconds` p95 — hold it inside baseline +50 ms;
  the first request after Neon idle pays the compute wake (health check
  warms it).
- `ceit_searches_total` / `ceit_chat_searches_total` — request mix
  unchanged; watch for drops (errors) not just latency.
- HTTP 5xx rate and empty-result rate on `/search`.
- `ceit_recommend_total`, `ceit_recommend_duration_seconds` once Laravel
  calls `/recommend`.
- `ceit_index_documents`, `ceit_rebuilds_total`,
  `ceit_last_rebuild_timestamp_seconds` — corpus pushes stay incremental;
  `took_ms` on `/index/rebuild` and `/corpus/upload` should be near zero
  for unchanged corpora.
- Neon connection usage vs `PG_POOL_SIZE` (4 default, per process).
