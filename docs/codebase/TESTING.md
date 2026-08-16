# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary test framework: pytest (>=8.0, dev group)
- Assertion/mocking tools: pytest fixtures, injected fake LLM clients, deterministic embedder, `httpx` (TestClient + live smoke)
- Commands:

```bash
uv run pytest                # full suite: 74 passed / 1 skipped (live test env-gated)
uv run pytest tests/test_api.py
uv run ruff check .
uv run ruff format --check .
```

### 2) Test Layout

- Test file placement: `tests/` flat, one file per module: `test_api.py` (endpoints), `test_search.py`/`test_rrf.py` (ranking), `test_ingest.py`, `test_rag.py`, `test_chat_stream.py`, `test_agentic_loop.py`, `test_filters.py`, `test_chat_stream_live.py` (env-gated)
- Naming convention: `test_<area>.py`, functions `test_*`
- Setup files: `tests/conftest.py` — sets `SIDECAR_TOKEN`/`CORPUS_PATH` **before app imports** (app modules construct `Settings()` at import time), inserts repo root on `sys.path`, provides `DeterministicEmbedder` (hash-based 8-dim vectors), temp-corpus fixtures, `build_test_index` helper

### 3) Test Scope Matrix

| Scope | Covered? | Typical target | Notes |
|-------|----------|----------------|-------|
| Unit | yes | RRF math, filters, citation payload shape, tool-arg validation, ingest validation | `test_rrf.py`, `test_filters.py`, `test_ingest.py` |
| Integration | yes | API endpoints through `TestClient`, agentic loop with injected fake client, streamed chat framing | `test_api.py`, `test_agentic_loop.py`, `test_chat_stream.py` |
| E2E / live | gated | Real provider round-trip (`/chat/stream` against running server) | `test_chat_stream_live.py` — skipped unless `SIDECAR_LIVE_CHAT_TEST=1`; never in CI |
| Golden-set eval | separate | Retrieval quality vs `data/golden_dataset.json` | `app/eval.py` (not pytest) |

### 4) Mocking and Isolation Strategy

- Main mocking approach: `rebuild._embed_override` (deterministic embedder) so no HuggingFace model loads in tests; `AgenticLoop`/`RagService` accept injected `client` objects (fake OpenAI-compatible stream responses)
- Isolation guarantees: temp corpus per test (`tmp_path`); versioned cache under tmp dirs; in-memory FTS5 files in tmp
- Common failure mode: importing `app.config` before env is set → conftest `os.environ.setdefault` ordering (documented in conftest)

### 5) Coverage and Quality Signals

- Coverage tool: pytest-cov, only in the SonarCloud workflow (`pytest --cov=app --cov-report=xml`) — no local threshold
- Current reported coverage: not gated locally; SonarCloud consumes `coverage.xml` [TODO — no reported number on file]
- Known gaps: live LLM behavior (env-gated), provider edge cases (simulated via fake clients), large-corpus performance

### 6) Evidence

- `tests/conftest.py`, `tests/test_api.py`, `tests/test_agentic_loop.py`, `tests/test_chat_stream_live.py`
- `pyproject.toml` (`[tool.pytest.ini_options]` testpaths, dev group)
- `.github/workflows/python-app.yml` (lint/test/smoke jobs), `.github/workflows/sonarcloud.yml`

## Extended Sections (Optional)

### CI pipeline (GitHub Actions)

- `python-app.yml`: lint (ruff check + format) → test (pytest) → smoke (boot uvicorn on the runner, verify `/health` 200 with token and 401 without) — the smoke job mirrors the docker-build job in the Laravel repo (no Dockerfile here)
- `sonar-secrets.yml`: offline SonarQube secrets scan (passes independently of org status)
- `sonarcloud.yml`: coverage run + SonarCloud scan (project key `PSergio984_ceit-ai-sidecar`); requires `SONARQUBE_TOKEN` repo secret + active org
- Local parity: 74 passed / 1 skipped; ruff clean; format clean
