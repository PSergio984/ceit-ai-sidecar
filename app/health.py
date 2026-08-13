"""Health assembly: index coverage + staleness per the v1 contract."""

from __future__ import annotations

from .rebuild import load_state


def assemble_health(cache_dir) -> dict:
    """GET /health shape: healthy = index loaded AND documents == embedded."""
    state = load_state(cache_dir)

    if state is None:
        return {
            "status": "degraded",
            "contract_version": "v1",
            "model": None,
            "index": None,
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
    }
