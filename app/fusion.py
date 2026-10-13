"""Backend-independent fusion: filters, RRF ranking, code pin, result shaping.

Pure functions over plain dicts — no I/O, no sqlite. Every ordering is
deterministic: hybrid/semantic/recommend sort by (-score, id), bm25 by rank —
never by set or dict iteration order.
"""

from __future__ import annotations

import re

import numpy as np

CODE_PIN_RE = re.compile(r"^CEIT-[A-Z]{2}-\d{2}(-\d+)?$", re.IGNORECASE)

AUTHOR_KEYS = ("research_adviser", "technical_adviser")

# Reciprocal Rank Fusion constant (shared across the API, agent, eval, judge).
RRF_K = 60


def _rrf_score(k: int, rank: int) -> float:
    return 1.0 / (k + rank)


def passes_filters(doc: dict, filters: dict | None, corpus: str | None) -> bool:
    meta = doc.get("metadata") or {}
    f = filters or {}
    if f.get("paper_type") and meta.get("paper_type") != f["paper_type"]:
        return False
    if f.get("department") and meta.get("department") != f["department"]:
        return False
    if f.get("publication_year") and meta.get("publication_year") != int(f["publication_year"]):
        return False
    if f.get("year_from") and int(meta.get("publication_year") or 0) < int(f["year_from"]):
        return False
    if f.get("year_to") and int(meta.get("publication_year") or 0) > int(f["year_to"]):
        return False
    if f.get("author"):
        needle = str(f["author"]).lower()
        if not any(needle in str(name).lower() for name in (meta.get("authors") or [])):
            return False
    if f.get("adviser"):
        needle = str(f["adviser"]).lower()
        if not any(needle in str(meta.get(k) or "").lower() for k in AUTHOR_KEYS):
            return False
    return not (corpus and doc.get("corpus") != corpus)


def _semantic_rank_map(semantic_scores: dict[str, float]) -> dict[str, int]:
    """1-based rank per doc id by descending cosine; ties break by id."""
    order = sorted(semantic_scores, key=lambda did: (-semantic_scores[did], did))
    return {did: i + 1 for i, did in enumerate(order)}


def fuse(
    method: str,
    k: int,
    bm25_ranks: dict[str, int],
    semantic_scores: dict[str, float],
) -> tuple[list[str], dict[str, float], dict[str, int]]:
    """Fuse per-channel candidates into (ranked_ids, scores, sem_rank_map).

    `bm25_ranks` holds the surviving hits re-indexed to contiguous 1-based
    ranks (a filtered-out doc must not consume a rank position or distort the
    fused score of survivors).
    """
    sem_rank_map = _semantic_rank_map(semantic_scores)

    if method == "bm25":
        ranked = sorted(bm25_ranks, key=bm25_ranks.get)
        scores = {did: _rrf_score(k, bm25_ranks[did]) for did in ranked}
    elif method == "semantic":
        ranked = sorted(semantic_scores, key=lambda did: (-semantic_scores[did], did))
        scores = dict(semantic_scores)
    else:  # hybrid — RRF fusion.
        scores = {did: _rrf_score(k, rank) for did, rank in bm25_ranks.items()}
        for did, rank in sem_rank_map.items():
            scores[did] = scores.get(did, 0.0) + _rrf_score(k, rank)
        ranked = sorted(scores, key=lambda did: (-scores[did], did))

    return ranked, scores, sem_rank_map


def recommend_rank(
    docs_by_id: dict,
    vectors: np.ndarray,
    seed_ids: list[str],
    exclude_ids: list[str],
    limit: int,
) -> tuple[int, list[dict]]:
    """Rank catalog docs by cosine to the mean of the resolved seed vectors.

    `vectors` rows align with `list(docs_by_id.keys())` (the _semantic_scores
    convention). A seed resolves only when it exists in docs_by_id AND its
    corpus is "catalog"; unknown and policy seeds are ignored. seeds_used is
    the count of distinct resolved seeds — 0 means no query vector, so the
    result is (0, []). Candidates are catalog docs only, minus every requested
    seed id (resolved or not) and exclude_ids. Ordering is (-score, id) so the
    same request always returns the same list.
    """
    ids = list(docs_by_id.keys())
    requested = set(seed_ids)
    resolved = [
        i
        for i, doc_id in enumerate(ids)
        if doc_id in requested and docs_by_id[doc_id].get("corpus") == "catalog"
    ]
    seeds_used = len(resolved)
    if seeds_used == 0:
        return 0, []

    q = vectors[resolved].mean(axis=0)
    norm = np.linalg.norm(q)
    if norm == 0:
        return seeds_used, []
    q = q / norm

    blocked = requested | set(exclude_ids)
    scores = vectors @ q  # normalized vectors -> cosine similarity
    candidates = sorted(
        (
            (float(scores[i]), doc_id)
            for i, doc_id in enumerate(ids)
            if doc_id not in blocked and docs_by_id[doc_id].get("corpus") == "catalog"
        ),
        key=lambda item: (-item[0], item[1]),
    )
    results = []
    for score, doc_id in candidates[:limit]:
        doc = docs_by_id[doc_id]
        results.append(
            {
                "id": doc["id"],
                "corpus": doc.get("corpus"),
                "title": doc.get("title"),
                "score": round(score, 4),
                "metadata": doc.get("metadata") or {},
            }
        )
    return seeds_used, results


def pin_catalog_code(query: str, ranked: list[str], docs_by_id: dict) -> str | None:
    """Code-exact pin (D-02): matching catalog_code moves to rank 1 in place."""
    upper_query = query.upper()
    if not CODE_PIN_RE.match(upper_query):
        return None
    for did in ranked:
        meta = docs_by_id[did].get("metadata") or {}
        if (meta.get("catalog_code") or "").upper() == upper_query:
            ranked.remove(did)
            ranked.insert(0, did)
            return did
    return None


def shape_results(
    ranked: list[str],
    limit: int,
    docs_by_id: dict,
    scores: dict[str, float],
    bm25_ranks: dict[str, int],
    sem_rank_map: dict[str, int],
    method: str,
    pinned_id: str | None,
    include_text: bool,
) -> list[dict]:
    results = []
    for did in ranked[:limit]:
        doc = docs_by_id[did]
        meta = doc.get("metadata") or {}
        result = {
            "id": doc["id"],
            "corpus": doc.get("corpus"),
            "title": doc.get("title"),
            "score": round(scores[did], 4),
            "bm25_rank": bm25_ranks.get(did) if method in ("bm25", "hybrid") else None,
            "semantic_rank": (sem_rank_map.get(did) if method in ("semantic", "hybrid") else None),
            "pinned": pinned_id == did,
            "metadata": meta,
        }
        if include_text:
            result["text"] = doc.get("text", "")
        results.append(result)
    return results
