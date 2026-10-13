# pgvector gate report — issue #9

Run 2026-10-13 against the frozen baseline (`golden-run1.json`,
`judge-run1/2`, `latency-memory.json`).

Environment: Neon branch `test-pgstore` (project `bitter-bar-65388505`,
Postgres 18, `lakebase_text` — `keyword_index = lakebase_bm25`), pooled
endpoint, schema `gate_run`, 1,338 documents synced via `PgIndexStore.sync`
with the committed `index/` artifacts (bit-identical vectors to the
baseline). Measurement host: Windows dev machine, ~35 ms network RTT to the
Neon region (ap-southeast-1).

Command: `VECTOR_BACKEND=pgvector PG_INDEX_SCHEMA=gate_run
MODEL_NAME=all-MiniLM-L6-v2 python -m app.eval --json`

## Retrieval quality — PASS

Hybrid method vs baseline:

| metric | baseline | pgvector | criterion | verdict |
|---|---|---|---|---|
| negative_pass_rate | 1.0000 | 1.0000 | ≥ baseline | pass |
| top1_rate | 0.8182 | 0.7727 | ≥ baseline | **fail** vs frozen file; **exact** vs corrected reference* |
| avg_recall @5 | 0.7239 | 0.7235 | within −0.02 | pass (−0.0004) |
| avg_f1 @5 | 0.3913 | 0.3908 | within −0.02 | pass (−0.0005) |

\* Read literally, "top-1 rate at least baseline" fails: 0.7727 < the
frozen file's 0.8182. The qualifier is that `docs/baseline/README.md`
documents 0.8182 as pre-deterministic-tie-break hash luck on a
duplicate-title case and names **0.7727** as "the reference for later
gates" — the measured value equals it exactly. So the criterion as
written is not met; satisfying it requires either revising the frozen
baseline (a data decision — issue #26) or accepting the documented
corrected reference. That decision belongs to the maintainer, not this
report. Every one of the 3 case-level differences vs the frozen
file is a duplicate-title ambiguity ("Energy-Efficient Street Lighting
Control Using Microcontrollers" papers 914/943, "Secure Role-Based Access
Control for Student Portals" papers 849/893/942, "Intelligent Traffic
Monitoring for Campus Roadways" papers 847/872/908): the returned doc shares
the title and a statistically identical fused score; which twin lands first
is arbitrary. Not a retrieval regression — but worth noting the golden set
still labels only one twin per query, so this class of flip will keep
happening until the labels are disambiguated.

Per-category (hybrid) — all within the 0.02 band:

| category | recall Δ | f1 Δ | top1 Δ |
|---|---|---|---|
| catalog_code | 1.000→1.000 | 0.333→0.333 | 1.000→1.000 |
| exact_title | 1.000→1.000 | 0.333→0.333 | 0.625→0.500 (dup titles) |
| paraphrase | 0.197→0.196 | 0.291→0.290 | 0.833→0.833 |
| people | 0.685→0.685 | 0.715→0.715 | 1.000→1.000 |

Per-method: bm25 top-1 improved 0.7273→0.7727 (the documented favourable
lakebase flip on one duplicate-title case); semantic identical; both
channels' negative pass rate 1.0.

## Judge — PASS (within noise)

`python -m app.judge`, model `qwen/qwen3.8-27b` (the baseline model;
`.env`'s `llama-3.3-70b-versatile` is retired on Groq and fails 404 —
`.env` needs updating):

- pgvector run: RELEVANT 12 / 40 → relevant_rate **0.30**
- baseline runs: 0.25 and 0.225; observed noise band ~0.05, 7/40 questions
  changed label between runs.

0.30 is inside the band (and in the favourable direction). Note the judge
numbers are Groq-run numbers, not comparable to production OpenRouter runs,
same caveat as the baseline.

## Latency — FAILS as measured here; needs the real host

Warm p95, same 22 positive golden queries × 3 passes, in-process calls:

| method | baseline p95 | pgvector p95 | Δ |
|---|---|---|---|
| bm25 | 17.2 ms | 375.7 ms | +358 ms |
| semantic | 16.6 ms | 250.9 ms | +234 ms |
| hybrid | 28.3 ms | 462.5 ms | +434 ms |

The +50 ms criterion fails by a wide margin **on this measurement**, but the
dominant term is network, not query cost: a bare `SELECT 1` round trip to
the pooled endpoint measures ~35 ms (p50), and `rrf_search` issues ~5
sequential round trips per query (state read, BM25 tx, gate scan, scores,
doc fetch) — a ~175 ms floor before any work. The remaining delta is query
execution plus TLS/RTT variance.

This number does not settle the criterion: the gate intends the *real host*
measurement (same AC as the 400 MB memory check). If the cloud host sits in
or near the Neon region with a few-ms RTT, ~5 round trips ≈ 15–25 ms of
network and the criterion is plausibly met. If not, the fix is structural —
fewer round trips per search (pipelined/batched queries or a single fused
SQL statement), not a bigger machine.

## Memory — NOT MEASURED (real-host step)

Per the AC, RSS < 400 MB must be measured on the real host; a dev-machine
number is not meaningful for the decision.

## Verdict

- Retrieval quality: **pass on negative-pass, recall, F1 and every
  per-category band**; top-1 **fails the criterion as written** (0.7727 <
  frozen 0.8182) — the entire delta is the itemized duplicate-title label
  ambiguity, and the measured value equals the documented post-tie-break
  reference. Whether to accept the corrected reference or regenerate the
  golden set is the maintainer's call (issue #26).
- Judge: **pass**, within the measured noise band.
- Latency: **incomplete** — fails vs +50 ms as measured dev→Neon-SG where
  RTT dominates; the criterion is only answerable on the real host. If the
  host is remote from the database, expect this to need the round-trip
  reduction work noted in ADR 0019.
- Memory: **not measured** — real-host step.

Per the spec, "if any criterion fails, the report says which and the
default backend stays SQLite": top-1 fails the frozen-baseline criterion
for the documented duplicate-title reason, and latency/memory are
incomplete pending the real host — so the default backend stays
`sqlite` and the flip waits on step 5 of `docs/rollout-pgvector.md`
plus the maintainer's call on the golden-set labels.
