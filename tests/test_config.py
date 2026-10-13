"""Settings must load a filled-in .env, including keys it does not declare."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
UNDECLARED_KEYS = ("GRAFANA_ADMIN_PASSWORD", "DATABASE_URL_UNPOOLED", "NEON_BRANCH")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("SIDECAR_TOKEN", *UNDECLARED_KEYS):
        monkeypatch.delenv(key, raising=False)


def test_settings_loads_env_example_with_filled_in_extra_keys(tmp_path):
    env_file = tmp_path / ".env"
    lines = [
        line
        for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
        if not line.startswith(tuple(f"{key}=" for key in ("SIDECAR_TOKEN", *UNDECLARED_KEYS)))
    ]
    lines += [
        "SIDECAR_TOKEN=test-token",
        "GRAFANA_ADMIN_PASSWORD=not-a-real-password",
        "DATABASE_URL=postgresql://example.invalid/pooled",
        "DATABASE_URL_UNPOOLED=postgresql://example.invalid/direct",
        "NEON_BRANCH=test",
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    settings = Settings(_env_file=str(env_file))

    assert settings.sidecar_token == "test-token"
    assert settings.vector_backend == "sqlite"
