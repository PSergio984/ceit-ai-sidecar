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
- Run 2: 40 questions, RELEVANT 9 / PARTLY 0 / NON_RELEVANT 31, relevant rate 0.225 (`judge-run2.*`).
  Captured after adding multi-key failover (the first attempt had hit a Groq 503, then the daily token limit).
- Noise band (same code, same model): relevant rate differs by 0.025 between runs; 7 of 40 questions
  (17.5%) changed label (q01, q10, q11, q14, q21, q22, q23). Judge-score comparisons in the pgvector
  gate must be read against this band, so a relevant-rate change under about 0.05 is not a regression.

## Latency and memory (`latency-memory.json`, `bench.py`)
Warm, 27 golden queries x 5 passes, Windows dev machine, in-process (no HTTP). p50/p95 ms:
bm25 6.5/17.2, semantic 12.1/16.6, hybrid 14.1/28.3, hybrid+rerank 16.2/32.7.
Working set after warm-up 200 MB, peak 235 MB (loads vectors into RAM).

## Post-refactor check (ticket #4, deterministic tie-break)
After moving fusion to `app/fusion.py` with a `(-score, id)` tie-break, golden results are identical
across `PYTHONHASHSEED` 0-3. Versus the frozen baseline: bm25 and semantic are unchanged; hybrid and
hybrid+rerank precision/recall/F1 are unchanged, and top-1 is 0.7727 (17/22) instead of 0.8182
(seed 0) / 0.8636. The single changed case is the exact-title query "Energy-Efficient Street Lighting
Control Using Microcontrollers": paper-914 and paper-943 share that title, the golden set marks only
paper-943 relevant, and the old hash order happened to put 943 first. The id tie-break picks 914.
This is a golden-set ambiguity (duplicate titles), not a retrieval regression; the deterministic
number (0.7727) is the reference for later gates.
