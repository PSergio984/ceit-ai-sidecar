# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item | Rule | Example | Evidence |
|------|------|---------|----------|
| Files/modules | snake_case | `search.py`, `rebuild.py`, `rag.py` | `app/` |
| Classes | PascalCase | `HybridSearch`, `AgenticLoop`, `RagService`, `ToolArgs` | `app/` |
| Functions/methods | snake_case, descriptive | `rrf_search`, `stream_agentic_events`, `merge_dedup` | `app/` |
| Constants | UPPER_SNAKE | `MAX_TOOL_ROUNDS`, `MAX_DOC_CHARS`, `CITATION_KEYS`, `CONTRACT_VERSION` | `app/agent.py`, `app/rag.py` |
| Env vars (settings fields) | snake_case fields ← UPPER_SNAKE env | `sidecar_token` ← `SIDECAR_TOKEN`, `corpus_path` ← `CORPUS_PATH` | `app/config.py`, `.env.example` |

### 2) Formatting and Linting

- Formatter + linter: ruff (line-length 100)
- Enforced: `ruff check` (lint) + `ruff format --check` (format) — both CI gates in `.github/workflows/python-app.yml`
- Run commands:
```bash
uv run ruff check .
uv run ruff format --check .
```

### 3) Import and Module Conventions

- Standard style: stdlib → third-party → local (`from __future__ import annotations` first in typed modules)
- Relative imports within the package (`from .config import settings`)
- Type hints on public signatures; pydantic `BaseModel` for any validated payload

### 4) Error and Logging Conventions

- Provider/LLM errors: caught at the SSE boundary, logged (`logger.error(repr(exc))`), surfaced as `event: error` with a user-safe message (`provider_error`) — never raw exception text to clients
- Corpus/validation errors: `ValueError` with explicit messages (fail-closed, T-04); `load_documents` refuses partial/malformed corpora
- Tool-arg errors: `ValidationError` → corrective tool message back to the LLM; 2-malformed streak aborts (fail-closed to refusal or grounded answer)
- Envelope errors: never leak internals; `/index/rebuild` wraps exceptions in `{"error": {"code": "rebuild_failed", ...}}`

### 5) Testing Conventions

- Test file naming: `tests/test_<module>.py` (`test_search`→`test_rrf.py`, `test_api.py`, `test_agentic_loop.py`, ...)
- Isolation: `conftest.py` sets `SIDECAR_TOKEN`/`CORPUS_PATH` before app imports; deterministic hash-based embedder + temp corpora; `_embed_override` for API/rebuild tests; injected fake LLM clients for chat tests
- Live smoke (`test_chat_stream_live.py`) skipped unless `SIDECAR_LIVE_CHAT_TEST=1` — never in CI
- Coverage: `pytest --cov=app --cov-report=xml` only in the SonarCloud workflow (no local threshold)
- Weekday/env-dependent tests are gated by explicit env checks

### 6) Evidence

- `pyproject.toml` (`[tool.ruff]`, `[tool.pytest.ini_options]`)
- `tests/conftest.py`, `tests/test_api.py`
- `app/agent.py` (docstring decision references D-xx/T-xx/WR-x)
- `.github/workflows/python-app.yml`

## Extended Sections (Optional)

### Decision-reference convention

Docstrings cite phase decisions (`D-02` code pin, `D-03` filter-before-fusion, `D-07` direct answer, `D-09` closed schemas, `D-11` 3-round cap, `D-12` no args/results JSON in activity lines, `D-17` files-only) and review items (`R6` no chunking, `T-04` fail-closed ingest, `WR-1` content/tool_calls mutual exclusion, `WR-2` tool-result truncation). New code touching these seams should keep the convention so the trace stays readable.
