# Search baseline (SQLite hybrid backend) - ticket #3

Captured 2026-10-12 on the current SQLite FTS5 + MiniLM + RRF backend, `MIN_SEMANTIC_SIMILARITY=0.5`.

## Reproduce
Golden: `PYTHONHASHSEED=0 MIN_SEMANTIC_SIMILARITY=0.5 python -m app.eval --with-rerank --json`
(`golden-run1.json` and `golden-run2.json` are byte-identical.)

## Known nondeterminism (must be fixed in the shared fusion module, ticket #4)
Hybrid fusion builds candidates from a `set`, so score ties are ordered by the string hash
seed. Without `PYTHONHASHSEED` pinned, hybrid top-1 varies between 0.8182 and 0.8636 across
runs (seeds 1-3 observed); precision/recall/F1 do not vary. The baseline is therefore
only reproducible with `PYTHONHASHSEED=0`. Any id tie-break change must be evaluated against
this baseline, not assumed neutral.

## Golden results (27 cases, k=5; PYTHONHASHSEED=0)
See `golden-run1.json` (`approach_comparison`, `by_category`).

## Judge
Model (answerer and judge): `qwen/qwen3.8-27b` on Groq, harness defaults (judge max_tokens=32).
`llama-3.3-70b-versatile` is not available to this key and the `gpt-oss` models return empty
labels at 32 tokens, so absolute judge numbers are not comparable to production OpenRouter runs.
- Run 1: 40 questions, RELEVANT 10 / PARTLY 0 / NON_RELEVANT 30, relevant rate 0.25 (`judge-run1.*`)
- Run 2: NOT captured. First attempt hit a Groq 503, retry hit the 200k tokens-per-day limit.
  The noise band is therefore still unmeasured; rerun after the daily limit resets.

## Latency and memory (`latency-memory.json`, `bench.py`)
Warm, 27 golden queries x 5 passes, Windows dev machine, in-process (no HTTP). p50/p95 ms:
bm25 6.5/17.2, semantic 12.1/16.6, hybrid 14.1/28.3, hybrid+rerank 16.2/32.7.
Working set after warm-up 200 MB, peak 235 MB (loads vectors into RAM).
