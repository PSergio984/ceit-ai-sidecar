# CEIT AI Sidecar — containerized hybrid-search + RAG service.
#
# Build:  docker build -t ceit-ai-sidecar .
# Run:    docker compose up --build  (sidecar + Prometheus + Grafana)
#
# The corpus is bundled in the image (corpus/catalog.json + corpus/policies.json)
# so the index builds standalone. The embedding model downloads on first
# rebuild — the cache volume keeps it warm across restarts.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.13-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY --from=builder /app/.venv /app/.venv
COPY . .
EXPOSE 8310
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8310"]
