"""Index-storage seam: which backend owns the document index + its state.

SQLite keeps the versioned file artifacts (docs-N.json / vectors-N.npy /
index-N.db + state.json); pgvector keeps documents and index_state in
Postgres. Search/recommend on Postgres land in later tickets — this seam
covers storage, sync and health state only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import Settings


class IndexStore(Protocol):
    name: str

    def load_state(self) -> dict | None:
        """Current index state, or None when no index has been built yet."""
        ...

    def sync(self, corpus_path: Path, model_name: str, embed_fn=None) -> dict:
        """Sync the index to the corpus; return the new index state."""
        ...


class SqliteIndexStore:
    """Delegates to the existing versioned-file rebuild — no behaviour change."""

    name = "sqlite"

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    def load_state(self) -> dict | None:
        from .rebuild import load_state

        return load_state(self.cache_dir)

    def sync(self, corpus_path: Path, model_name: str, embed_fn=None) -> dict:
        from .rebuild import build_index

        return build_index(corpus_path, model_name, self.cache_dir, embed_fn=embed_fn)


def create_index_store(settings: Settings) -> IndexStore:
    if settings.vector_backend == "sqlite":
        return SqliteIndexStore(Path(settings.cache_dir))
    if settings.vector_backend == "pgvector":
        if not settings.database_url:
            raise ValueError("vector_backend=pgvector requires DATABASE_URL")
        from .pg_index import PgIndexStore

        return PgIndexStore(
            settings.database_url,
            schema=settings.pg_index_schema,
            connect_timeout=settings.pg_connect_timeout,
        )
    raise ValueError(f"unknown vector_backend: {settings.vector_backend}")
