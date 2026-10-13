# ADR 0018: Incremental ingest (`PgIndexStore.sync`) and the offline corpus push

Status: accepted

## Context

The SQLite path rebuilds the whole index per push: re-embed every
document, rewrite every artifact. On the 500 MB cloud host a full
re-embed runs out of memory, which is why index files ended up committed
to git. Issue #2 requires the Postgres path to embed only new or changed
documents, delete removed ones, no-op on an unchanged corpus (zero
embedder calls), commit atomically so readers never see a partial index,
and leave the previous index intact on any failure. It also requires an
offline script for the full load so a re-embed never touches the small
host — with LangChain confined to that script so the runtime image stays
small.

## Decision

Two writers, one index, one shared core. `PgIndexStore.sync`
(`app/pg_index.py`) is the in-app incremental path — `/index/rebuild` and
`/corpus/upload` route through `create_index_store(...).sync(...)` with
their request/response contracts unchanged. `scripts/push_embeddings.py`
is the offline path. Both share `app/pg_common.py` (content hash, diff,
advisory-lock key) and `PgIndexStore`'s own schema/diff/maintenance
helpers, so the writers cannot drift on what counts as "changed".

### Incremental sync

- `load_documents()` validates the corpus **before any connection** — a
  bad export never touches the database, and a failed push leaves the
  previous index intact.
- `content_hash` is sha256 over canonical JSON of
  `{model_name, id, corpus, title, text, metadata}` — `model_name` is in
  the hash, so a model change re-embeds every document.
  `diff_documents` splits the corpus into added / changed / removed;
  only added+changed are embedded (`embed_documents`, bounded batches of
  `EMBED_BATCH_SIZE`=64) and upserted in batches of 200; removed ids are
  deleted. An unchanged corpus short-circuits to a metadata refresh —
  zero embedder calls.
- Everything else is **one transaction**: advisory lock → schema ensure →
  upserts + deletes + the `index_state` write → commit. Readers see the
  old index until commit — a concurrent reader sees old or new, never
  partial.
- The lock is `pg_advisory_xact_lock(727_272_006)`: transaction-scoped,
  released at commit, so it serializes concurrent syncs/pushes across
  processes *and* stays safe under Neon's transaction pooling (a
  session-scoped lock would not).
- `index_state` is a single row (`id = 1` CHECK). `version`/`built_at`
  advance only when the data changed; `documents`/`embedded`/`by_corpus`
  are counted live from `documents`, never trusted from bookkeeping, so
  `/health` notices a table that disagrees with the state row.
- `load_state` reads the state row and the live counts inside one
  `REPEATABLE READ` snapshot — a sync committing between the two reads
  must never pair version N with counts from version N+1. A leading
  `SELECT 1` wakes a sleeping Neon compute; missing tables map to `None`
  (never synced → `/health` degraded, same as an unbuilt SQLite index).

### Post-commit maintenance

After the data commits, on an autocommit connection: the keyword index is
ensured (`lakebase_bm25` when the extension is available, else GIN —
recorded in `index_state.keyword_index`), then `VACUUM (ANALYZE)` when
the sync was dirty. This runs **after** commit because `VACUUM` cannot
run inside a transaction, and the index ensure runs on *every* sync so a
failed `CREATE INDEX` cannot strand `index_state.keyword_index`.
Maintenance failures are logged, never raised — the data already
committed.

### Offline embed-and-push

`uv run --group offline python scripts/push_embeddings.py <corpus>
[--force-full]` mirrors `sync` end to end: validate before connecting,
the same hash diff (`--force-full` re-embeds all), embed only
added/changed, one transaction for every write, then the same
post-commit maintenance. An unchanged corpus is a complete no-op.

Rows go through langchain_postgres' `AsyncPGVectorStore` into the same
`<schema>.documents` table — `metadata_columns=[corpus, title,
content_hash]` so stored rows match a sync exactly. Embeddings come from
`app.ingest.embed_documents`, not from LangChain: the store's required
`Embeddings` object is never invoked. The LangChain store commits per
call on its own SQLAlchemy connection, so its engine is fed a shared
psycopg `AsyncConnection` (`_HeldConnection`) whose commit/rollback/close
are neutralised — upserts, deletes and the state write land in the one
explicit transaction the script owns.

The push runs under the same `SYNC_LOCK_KEY` as sync, but session-scoped
(`pg_advisory_lock`) so the lock also covers the schema ensure and
post-commit maintenance outside the data transaction — run it against
Neon's direct endpoint, not the transaction pooler (see
`docs/rollout-pgvector.md`).

LangChain stays offline-only: it is lazily imported inside the script,
lives in the optional `offline` uv group (`langchain-postgres`), and the
Dockerfile's `uv sync --frozen --no-dev` never installs it — the runtime
image and default install carry no LangChain (issue #8 acceptance).

### Alternatives rejected

- **Full rebuild per push** (the SQLite model) — re-embeds 1,338 docs for
  a one-line corpus change; OOMs the 500 MB host.
- **Per-statement autocommit writes** — a mid-sync failure or concurrent
  reader would expose a half-updated index; the single transaction is
  what makes "old or new, never partial" provable.
- **Trusting stored counts in `index_state`** — live counts keep
  `/health` honest when the table and bookkeeping disagree.
- **A bespoke SQL writer in the script** — the shared `pg_common`
  helpers guarantee hash/diff/lock parity; LangChain's store is used
  only for its tested upsert/delete shape, and only offline.
