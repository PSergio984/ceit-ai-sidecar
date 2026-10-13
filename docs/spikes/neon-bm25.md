# Spike: lakebase_text BM25 parity on Neon (issue #5)

**Recommendation: GO** on `lakebase_text` BM25, provided the tsvector is built from normalized text (see "Tokenization"). The built-in `ts_rank_cd` fallback measured clearly worse and is not needed.

Measured 2026-10-13 on a Neon branch of project `ceit-library-ai` (Postgres 18.6, `lakebase_text` 0.1.3, region ap-southeast-1), loading the committed 1,338-document corpus (959 catalog + 379 policy). Reproduce with `scripts/spike_neon_bm25.py` (throwaway; reads `SPIKE_DATABASE_URL`, contains no credentials). The spike branch is set to auto-expire.

## Golden metrics, BM25-only, k=5 (27 cases: 22 positive, 5 negative)

Baseline is the live SQLite FTS5 BM25-only run through the same `app.eval` code (identical to the frozen `docs/baseline/golden-run1.json` bm25 row).

| Metric | SQLite FTS5 | PG BM25, raw tsvector | PG BM25, normalized tsvector | PG `ts_rank_cd` (normalized) |
|---|---|---|---|---|
| negative pass rate | 1.0 | 1.0 | 1.0 | 1.0 |
| avg precision | 0.5455 | 0.5182 | **0.5455** | 0.4909 |
| avg recall | 0.7517 | 0.6153 | **0.7517** | 0.7320 |
| avg F1 | 0.4321 | 0.3867 | **0.4321** | 0.4032 |
| top-1 rate | 0.7273 | 0.6364 | **0.7727** | 0.5909 |

With the normalized tsvector, negative pass rate, precision, recall and F1 equal SQLite exactly (delta 0.0000). Top-1 is +0.0455 (one query out of 22): `CEIT-CE-07-10` ranks the correct paper-400 first on Postgres and second on SQLite (paper-11 first). That is outside the literal 0.02 band but in the favourable direction, and I did not investigate the BM25 formula difference behind it. Golden metrics for the normalized variant were identical when run through Neon's pooled endpoint and with `prefilter` on; the other probes (candidate limit, prefilter correctness, ordering) were run over the direct endpoint only.

Caveat: 22 positive cases means one query moves a rate by 0.045. The corpus is synthetic.

## Tokenization of catalog codes

Postgres' default parser turns `CEIT-EE-03-07` into `'ceit-ee' 'ceit' 'ee' '03' '-07'`: the last segment becomes the signed-number token `-07`, so a query token `07` (what the sidecar's tokenizer and FTS5 produce) can never match. With the raw tsvector the four catalog-code queries lose avg recall 0.75 and two lose top-1.

Fix: build the tsvector from text with non-alphanumerics replaced by spaces, which mirrors FTS5 `unicode61`:

```sql
tsv tsvector GENERATED ALWAYS AS
  (to_tsvector('simple', regexp_replace(body, '[^[:alnum:]]+', ' ', 'g'))) STORED
```

Query side: tokenize in the app with the existing `Tokenizer(pattern=r"[\s\W]+", stop_words="english")` (the `simple` config has no stop words) and pass the joined tokens to `to_bm25query(to_tsvector('simple', ...))`. With this, all four catalog-code queries match SQLite.

## Behaviour findings

- **Candidate limit.** `lakebase_bm25.default_limit` defaults to 1000. On the index path a query matching all 1,338 rows returned only 1,000 at the default and all 1,338 at 65535, with scores equal to a sequential-scan ground truth. The setting accepts -1..65535 at `SET` time, but 65536 and above are rejected and `-1` errors when the index is scanned ("number of needed rows is set to -1"). So "retrieve everything" works only up to 65,535 rows; fine at 1,338, a hard ceiling to remember.
- **Prefilter and filters.** Corpus and metadata filters were honoured in every configuration (all returned rows passed). With `prefilter` off and `default_limit` 20, a filtered query returned only 9 of the 20 expected rows (the filter was applied to a truncated candidate list); with `prefilter` on, or `default_limit` 200 or more, it returned the correct 20.
- **Ordering and zero scores.** `<@>` scores are negative; ascending order is most relevant first. Repeat runs gave identical order and no matching rows tied on score in the test queries. The result depends on the plan: through the index only matching rows come back (7), but a sequential scan, which the planner chooses by default on a table this small, returns every row with score 0. Any query must therefore guard with `score < 0` (or `tsv @@ ...`) and add `id` as a tie-break in the final `ORDER BY`.
- **Pooling.** `SET LOCAL` inside a transaction works through the pooled (`-pooler`) endpoint; golden metrics were identical.

## Recommended settings for #6/#7

`SET LOCAL lakebase_bm25.default_limit = 65535` and `lakebase_bm25.prefilter = on` per query inside a transaction; filter `score < 0`; final `ORDER BY score, id`; create the BM25 index after the initial load, then `VACUUM`; `k1 = 1.2`, `b = 0.75`.

## Not measured

Query latency, index build time at scale, incremental-update behaviour of the BM25 index, and the semantic channel (pgvector, #6/#7).
