# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area | Value | Evidence |
|------|-------|----------|
| Primary language | Python | `pyproject.toml` |
| Runtime + version | Python >= 3.13 | `pyproject.toml` (`requires-python`), `.python-version` |
| Package manager | uv (lockfile `uv.lock`) | `uv.lock`, `pyproject.toml` |
| Module/build system | uv (non-package project, `[tool.uv] package = false`); no build step | `pyproject.toml` |

### 2) Production Frameworks and Dependencies

| Dependency | Version | Role in system | Evidence |
|------------|---------|----------------|----------|
| fastapi | >=0.141,<0.142 | HTTP API (search, chat, health, metrics) | `pyproject.toml` |
| uvicorn[standard] | >=0.52,<0.53 | ASGI server | `pyproject.toml` |
| sentence-transformers | >=5.7,<5.8 | Multilingual embedding model `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) | `pyproject.toml`, `app/ingest.py` |
| torch | CPU index (`pytorch-cpu`) | Embedding backend | `pyproject.toml` `[tool.uv.index]` / `[tool.uv.sources]` |
| sqlitesearch | >=0.3,<0.4 | SQLite FTS5 text index — **BM25 keyword retrieval** | `pyproject.toml`, `app/search.py` |
| pydantic | >=2.13,<2.14 | Request/response models, strict tool-args validation (`extra="forbid"`) | `pyproject.toml`, `app/agent.py` |
| pydantic-settings | >=2.15,<2.16 | `Settings` from env/.env | `pyproject.toml`, `app/config.py` |
| numpy | >=2.0 | Vector math (cosine via normalized matmul), `.npy` persistence | `pyproject.toml`, `app/search.py` |
| openai | >=2.45,<3 | LLM client (OpenRouter via `base_url`) | `pyproject.toml`, `app/rag.py`, `app/agent.py` |

### 3) Development Toolchain

| Tool | Purpose | Evidence |
|------|---------|----------|
| ruff | Lint + format (line-length 100); CI gate | `pyproject.toml` `[tool.ruff]`, `.github/workflows/python-app.yml` |
| pytest + pytest-cov | Tests; coverage for SonarCloud | `pyproject.toml` dev group, `.github/workflows/sonarcloud.yml` |
| httpx | Test client / live smoke | `pyproject.toml` dev group |

### 4) Key Commands

```bash
uv sync --frozen            # install from lockfile
uv run uvicorn app.main:app --port 8310   # run server (binds 127.0.0.1)
uv run pytest               # test suite (74 passed / 1 skipped)
uv run ruff check .         # lint
uv run ruff format --check .  # format gate
uv run python -m app.eval --json          # golden-set retrieval evaluation
uv run python -m app.eval --corpus policy # eval a single corpus
```

### 5) Environment and Config

- Config sources: `.env` (pydantic-settings, `SettingsConfigDict(env_file=".env")`)
- Required env vars: `SIDECAR_TOKEN` (shared with Laravel), `CORPUS_PATH` (default `../ceit-library/storage/app/ai-corpus`), `MODEL_NAME` (default `paraphrase-multilingual-MiniLM-L12-v2`); LLM: `LLM_BASE_URL` (OpenRouter), `LLM_API_KEY`, `LLM_MODEL` (default `meta-llama/llama-3.3-70b-instruct`), `LLM_MAX_TOKENS` (512)
- Deployment/runtime constraints: local dev binds loopback only; **production deploys to FastAPI Cloud** (`fastapi run` — requires the `fastapi[standard]` extra, added 2026-08-16); first run downloads the ~470 MB model; index cache in `cache/` (versioned artifacts); `CORPUS_PATH` must point at a corpus available in the cloud container (see INTEGRATIONS)

### 6) Evidence

- `pyproject.toml`, `uv.lock`
- `.env.example`, `app/config.py`
- `README.md`
- `.github/workflows/python-app.yml`, `.github/workflows/sonarcloud.yml`
