"""Corpus ingest: load exported JSON documents and embed them (whole-document).

Files only — the sidecar never touches the Laravel database (D-17).
Whole-document embeddings only — no sentence chunking (R6).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

INDEX_FILES = ("catalog.json", "policies.json")

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def load_documents(corpus_path: Path) -> list[dict]:
    """Load + validate catalog.json and policies.json from the corpus dir.

    Raises ValueError on malformed envelopes, missing required fields, or
    duplicate ids — never a silent partial index (T-04).
    """
    docs: list[dict] = []
    seen_ids: set[str] = set()
    found = 0

    for filename in INDEX_FILES:
        path = corpus_path / filename
        if not path.exists():
            continue
        found += 1

        payload = json.loads(path.read_text(encoding="utf-8"))

        if payload.get("schema_version") != 1:
            raise ValueError(
                f"{filename}: schema_version must be 1, got {payload.get('schema_version')!r}"
            )
        if not payload.get("generated_at"):
            raise ValueError(f"{filename}: missing required 'generated_at'")
        try:
            from datetime import datetime

            datetime.fromisoformat(payload["generated_at"])
        except ValueError as exc:
            raise ValueError(
                f"{filename}: 'generated_at' is not a parseable ISO timestamp"
            ) from exc

        for doc in payload.get("documents", []):
            for field in ("id", "corpus", "title", "text", "metadata"):
                if field not in doc:
                    raise ValueError(
                        f"{filename}: document missing required field '{field}': {doc.get('id')!r}"
                    )
            if doc["id"] in seen_ids:
                raise ValueError(f"duplicate document id: {doc['id']!r}")
            seen_ids.add(doc["id"])
            docs.append(doc)

    if found == 0:
        raise ValueError(
            f"no corpus files found in {corpus_path} (expected catalog.json and/or policies.json)"
        )

    return docs


def get_embedder(model_name: str):
    """Lazy singleton SentenceTransformer embedder (thread-safe)."""
    with _model_lock:
        if model_name not in _model_cache:
            from sentence_transformers import SentenceTransformer

            _model_cache[model_name] = SentenceTransformer(model_name)
        return _model_cache[model_name]


def embed_documents(docs: list[dict], model_name: str) -> np.ndarray:
    """One normalized vector per document (whole-document, no chunking)."""
    embedder = get_embedder(model_name)
    vectors = embedder.encode([d["text"] for d in docs], normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Normalized query embedding for cosine similarity."""
    embedder = get_embedder(model_name)
    vector = embedder.encode([query], normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)[0]


def write_cache(
    cache_dir: Path, version: int, docs: list[dict], vectors: np.ndarray, db_path: Path
) -> None:
    """Persist versioned index artifacts (atomic swap via rename)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_docs = cache_dir / f".docs-{version}.json.tmp"
    tmp_vec = cache_dir / f".vectors-{version}.tmp.npy"  # np.save appends .npy if missing

    tmp_docs.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    np.save(tmp_vec, vectors)

    docs_path = cache_dir / f"docs-{version}.json"
    vec_path = cache_dir / f"vectors-{version}.npy"
    final_db = cache_dir / f"index-{version}.db"

    import shutil

    shutil.copy2(tmp_docs, docs_path)
    shutil.copy2(tmp_vec, vec_path)
    shutil.copy2(db_path, final_db)
    tmp_docs.unlink(missing_ok=True)
    tmp_vec.unlink(missing_ok=True)


def prune_old_versions(cache_dir: Path, keep: int) -> None:
    """Best-effort cleanup of older versioned artifacts."""
    versions: set[int] = set()
    for p in cache_dir.glob("docs-*.json"):
        try:
            versions.add(int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            continue
    for old in sorted(versions)[:-keep]:
        for suffix in (".json", ".npy", ".db"):
            (cache_dir / f"docs-{old}{suffix}").unlink(missing_ok=True)
            (cache_dir / f"vectors-{old}{suffix}").unlink(missing_ok=True)
            (cache_dir / f"index-{old}{suffix}").unlink(missing_ok=True)
