"""Search backend seam: Protocol + factory selected by settings.vector_backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import settings
from .fusion import RRF_K


class SearchBackend(Protocol):
    def rrf_search(
        self,
        query: str,
        k: int = RRF_K,
        limit: int = 10,
        filters: dict | None = None,
        corpus: str | None = None,
        include_text: bool = False,
        method: str = "hybrid",
    ) -> list[dict]: ...

    def close(self) -> None: ...


def create_backend(cache_dir: Path, model_name: str) -> SearchBackend:
    if settings.vector_backend == "sqlite":
        from .search import HybridSearch

        return HybridSearch(cache_dir, model_name)
    if settings.vector_backend == "pgvector":
        raise NotImplementedError("pgvector backend is not implemented yet")
    raise ValueError(f"unknown vector_backend: {settings.vector_backend}")
