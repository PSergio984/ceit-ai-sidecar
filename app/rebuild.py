"""Full index rebuild: load -> embed -> build FTS5 index -> atomic swap.

Always a FULL rebuild from exported JSON (D-12). The old index keeps
serving while the new one is built; state.json is swapped last so readers
never observe a half-built index.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sqlitesearch import TextSearchIndex

from .ingest import (
    embed_documents,
    load_documents,
    prune_old_versions,
    source_generated_at,
    write_cache,
)

CONTRACT_VERSION = "v1"

_lock = threading.Lock()

# Test hook: tests inject a deterministic embedder here so API tests never
# hit HuggingFace. None = use the real ONNX embedder.
_embed_override = None


def _build_version(cache_dir: Path) -> int:
    state_path = cache_dir / "state.json"
    if state_path.exists():
        try:
            return int(json.loads(state_path.read_text(encoding="utf-8")).get("version", 0)) + 1
        except (json.JSONDecodeError, ValueError):
            return 1
    return 1


def build_index(
    corpus_path: Path,
    model_name: str,
    cache_dir: Path,
    embed_fn=None,
) -> dict:
    """Build a fresh index into versioned cache artifacts; return index_state.

    embed_fn is injectable for tests (fast deterministic embedder); the
    default embeds with the real ONNX embedder.
    """
    docs = load_documents(corpus_path)

    if embed_fn is None:
        vectors = embed_documents(docs, model_name)
    else:
        vectors = embed_fn(docs)

    version = _build_version(cache_dir)

    db_path = cache_dir / f".index-{version}.db"
    index = TextSearchIndex(
        text_fields=["text"],
        keyword_fields=["corpus", "department", "paper_type"],
        id_field="doc_id",
        db_path=str(db_path),
    )
    index_docs = [{**d, "doc_id": d["id"]} for d in docs]
    index.fit(index_docs)
    index.close()

    write_cache(cache_dir, version, docs, vectors, db_path)
    db_path.unlink(missing_ok=True)

    generated_at = source_generated_at(corpus_path)

    state = {
        "version": version,
        "built_at": datetime.now(UTC).isoformat(),
        "source_generated_at": generated_at,
        "documents": len(docs),
        "embedded": len(vectors),
        "by_corpus": dict(Counter(d["corpus"] for d in docs)),
        "model_name": model_name,
        "contract_version": CONTRACT_VERSION,
    }

    state_path = cache_dir / "state.json"
    state_tmp = cache_dir / ".state.json.tmp"
    state_tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(state_tmp, state_path)  # atomic: readers see old state until here

    prune_old_versions(cache_dir, keep=2)

    return state


def rebuild(settings) -> dict:
    """Synchronous index sync under a global lock (idempotent per flight).

    Delegates to the configured IndexStore: sqlite does a full versioned
    rebuild, pgvector an incremental content-hash sync.
    """
    from .index_store import create_index_store

    with _lock:
        return create_index_store(settings).sync(
            corpus_path=settings.corpus_path,
            model_name=settings.model_name,
            embed_fn=_embed_override,
        )


def load_state(cache_dir: Path) -> dict | None:
    """Current index state, or None when no index has been built yet."""
    state_path = cache_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
