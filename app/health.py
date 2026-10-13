"""Health assembly: index coverage + staleness per the v1 contract."""

from __future__ import annotations

import logging

from .index_store import IndexStore

logger = logging.getLogger("ceit.sidecar")


def assemble_health(store: IndexStore) -> dict:
    """GET /health shape: healthy = index loaded AND documents == embedded.

    An unreachable store degrades (index null) rather than failing — a
    sleeping or down database must never turn /health into a 500.
    """
    try:
        state = store.load_state()
    except Exception:  # unreachable backend reports degraded
        logger.warning("health: index store load_state failed", exc_info=True)
        state = None

    if state is None:
        return {
            "status": "degraded",
            "contract_version": "v1",
            "model": None,
            "index": None,
            "backend": store.name,
        }

    healthy = state["documents"] == state["embedded"]
    return {
        "status": "ok" if healthy else "degraded",
        "contract_version": state.get("contract_version", "v1"),
        "model": state.get("model_name"),
        "index": {
            "built_at": state.get("built_at"),
            "source_generated_at": state.get("source_generated_at"),
            "documents": state.get("documents"),
            "embedded": state.get("embedded"),
            "by_corpus": state.get("by_corpus"),
        },
        "backend": store.name,
    }
