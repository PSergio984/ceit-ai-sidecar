# ADR 0017: BM25 parity — `lakebase_text` with a normalized tsvector

Status: accepted

## Context

The SQLite backend's keyword channel is FTS5 BM25. The Postgres backend
needs equal keyword ranking or every fused result shifts. Neon retired
`pg_search` on 2026-09-21; `lakebase_text` (the `lakebase_bm25` access
method) is its BM25 replacement and is Neon-specific, so a plain-pgvector
fallback is also needed for CI and non-Neon hosts.

The issue #5 spike (`docs/spikes/neon-bm25.md`) loaded the committed
1,338-document corpus on a Neon branch (Postgres 18.6, `lakebase_text`
0.1.3) and scored BM25-only golden metrics against the live SQLite FTS5
run. Verdict: **GO** — but only with a normalized tsvector. The measured
contingency, built-in `ts_rank_cd`, came out clearly worse.

## Decision

The keyword channel runs `tsv <@> to_bm25query(...)` on the
`lakebase_bm25` index when `index_state.keyword_index = 'lakebase_bm25'`;
otherwise it runs the `ts_rank_cd`/`@@` GIN path (below). The choice is
read from `index_state` per query, so it always tracks what is actually
installed.

### Tokenization parity

The tsvector is a STORED generated column built from normalized text:

```sql
tsv tsvector GENERATED ALWAYS AS
  (to_tsvector('simple', regexp_replace(text, '[^[:alnum:]]+', ' ', 'g'))) STORED
```

Postgres' default parser turns `CEIT-EE-03-07` into `'ceit-ee' 'ceit' 'ee'
'03' '-07'` — the last segment becomes the signed-number token `-07`, so
the query token `07` that FTS5 (unicode61) and the sidecar tokenizer
produce can never match. With a raw tsvector the four catalog-code
queries lose ~0.14 average recall and two lose top-1. Replacing
non-alphanumerics with spaces before `to_tsvector` mirrors FTS5
tokenization.

On the query side the app tokenizes with the same tokenizer as the
SQLite backend — `Tokenizer(pattern=r"[\s\W]+", stop_words="english")` —
keeps digits (catalog codes, years), drops English stop words (the
`simple` config has none), and passes the joined tokens to
`to_bm25query(to_tsvector('simple', ...))`. A query with no alphanumeric
token issues no keyword query at all, matching `HybridSearch`.

Spike result, normalized: negative pass rate, precision, recall and F1
equal SQLite exactly (delta 0.0000). Top-1 is +0.0455 (0.7727 vs 0.7273)
— one query of 22, in the favourable direction.

### Per-transaction index settings

Inside the BM25 query transaction:

```sql
SET LOCAL lakebase_bm25.default_limit = 65535;
SET LOCAL lakebase_bm25.prefilter = on;
```

`SET LOCAL` scopes the GUCs to the transaction, so they are safe through
Neon's transaction-mode pooler — the spike measured identical golden
metrics through the pooled endpoint. `default_limit` defaults to 1000 and
silently truncates a match-everything query; 65535 is the maximum the
setting accepts (65536+ rejected at `SET`, -1 errors on index scan).
`prefilter=on` applies corpus/metadata filters before the candidate limit
— with it off and a small limit, filtered queries returned rows from a
truncated candidate list.

### Negative scores and ordering

`<@>` scores are negative; ascending order is most-relevant-first. A
sequential scan (the planner's default on a small table) returns *every*
row with score 0, so the query wraps the index scan and filters
`score < 0`, then orders `ORDER BY score, id` for deterministic output.

### Index

```sql
CREATE INDEX documents_bm25 ON documents
  USING lakebase_bm25 (tsv) WITH (k1 = 1.2, b = 0.75);
```

Created after the initial load, followed by `VACUUM` (ADR 0018); k1/b are
the FTS5 defaults, keeping the scoring parameters aligned.

### Fallback on plain pgvector

When `lakebase_text` is unavailable `index_state.keyword_index = 'gin'`
and the channel runs
`-ts_rank_cd(tsv, to_tsquery('simple', 't1 | t2 | …'))` over a
`GIN (tsv)` index with the same `WHERE` filters and `ORDER BY score, id`.
It is **functional but not score-identical**: the spike measured top-1
0.5909 / F1 0.4032 versus 0.7727 / 0.4321 for normalized BM25. It exists
so CI (`pgvector/pgvector:pg17` container) and non-Neon hosts keep a
working keyword channel; it is not a production substitute and the eval
gate must not be run against it. At query time, if the extension or index
vanished after state was written (`UndefinedObject`), the channel
degrades to the GIN path instead of 500ing.

### Alternatives rejected

- **Raw `to_tsvector` (no normalization)** — measured worse: catalog-code
  queries unmatchable, avg recall 0.6153 vs 0.7517.
- **`ts_rank_cd`/`@@` as the primary channel** — the spec's contingency;
  measured worse across top-1, recall and F1. Kept only as the
  portability fallback.
- **Stemmed configs (`english`)** — FTS5 `unicode61` does not stem; the
  `simple` config preserves that (stop words are removed app-side only,
  same as SQLite).
