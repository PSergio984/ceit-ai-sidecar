"""create_backend factory: settings.vector_backend selects the implementation."""

from __future__ import annotations

import pytest

from app.backend import create_backend
from app.config import settings
from app.search import HybridSearch


def test_sqlite_default_returns_hybrid_search(tmp_path):
    backend = create_backend(tmp_path, "test-model")
    assert isinstance(backend, HybridSearch)
    backend.close()


def test_pgvector_requires_database_url(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_backend", "pgvector")
    monkeypatch.setattr(settings, "database_url", "")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        create_backend(tmp_path, "test-model")


def test_unknown_backend_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "vector_backend", "bogus")
    with pytest.raises(ValueError):
        create_backend(tmp_path, "test-model")
