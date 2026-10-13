"""Dependency-free Postgres helpers shared by PgIndexStore (#6) and the
offline embed-and-push script (#8).

Both writers must agree on the content hash, the added/changed/removed diff
and the advisory-lock key — a push that disagreed with the in-app sync on any
of them would re-embed needlessly or fail to serialise against it.
"""

from __future__ import annotations

import hashlib
import json

# Serialises concurrent syncs/pushes across processes; transaction-scoped so
# it is safe under transaction pooling.
SYNC_LOCK_KEY = 727_272_006


def content_hash(model_name: str, doc: dict) -> str:
    """sha256 over canonical JSON — includes model_name so a model change
    re-embeds every document."""
    payload = {
        "model_name": model_name,
        "id": doc["id"],
        "corpus": doc["corpus"],
        "title": doc["title"],
        "text": doc["text"],
        "metadata": doc["metadata"],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def diff_documents(stored: dict[str, str], docs: list[dict], model_name: str) -> tuple:
    """(added, changed, removed, wanted) given stored id -> content_hash."""
    wanted = {d["id"]: content_hash(model_name, d) for d in docs}
    added = [d for d in docs if d["id"] not in stored]
    changed = [d for d in docs if d["id"] in stored and stored[d["id"]] != wanted[d["id"]]]
    removed = [doc_id for doc_id in stored if doc_id not in wanted]
    return added, changed, removed, wanted
