"""HybridSearch: FTS5 BM25 + semantic cosine, fused with RRF k=60.

Post-retrieval filters applied on the expanded candidate lists BEFORE
fusion — a filtered doc can never outrank an unfiltered relevant one.
Code-exact catalog-code queries pin the matching doc to rank 1.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sqlitesearch import TextSearchIndex, Tokenizer

from . import fusion
from .config import settings
from .fusion import (
    RRF_K,
    fuse,
    passes_filters,
    pin_catalog_code,
    recommend_rank,
    shape_results,
)
from .ingest import embed_query

# Re-exports kept for existing importers (`from app.search import ...`).
CODE_PIN_RE = fusion.CODE_PIN_RE
_rrf_score = fusion._rrf_score


def _version_artifacts(cache_dir: Path) -> tuple[int, dict, np.ndarray]:
    """Return (version, docs_by_id, vectors) for the current index state."""
    state_path = cache_dir / "state.json"
    if not state_path.exists():
        return 0, {}, np.empty((0, 0), dtype=np.float32)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    version = int(state.get("version", 0))

    docs_path = cache_dir / f"docs-{version}.json"
    vec_path = cache_dir / f"vectors-{version}.npy"
    if not docs_path.exists() or not vec_path.exists():
        return 0, {}, np.empty((0, 0), dtype=np.float32)

    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    docs_by_id = {d["id"]: d for d in docs}
    vectors = np.load(vec_path)
    return version, docs_by_id, vectors


class HybridSearch:
    """Loads the versioned index artifacts and runs RRF-fused hybrid search."""

    def __init__(self, cache_dir: Path, model_name: str, db_dir: Path | None = None):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self._db = None
        self._db_key: int | None = None

    def _ensure_db(self, version: int) -> TextSearchIndex | None:
        db_path = self.cache_dir / f"index-{version}.db"
        if not db_path.exists():
            return None
        if self._db is None or self._db_key != version:
            if self._db is not None:
                self._db.close()
            self._db = TextSearchIndex(
                text_fields=["text"],
                keyword_fields=["corpus", "department", "paper_type"],
                id_field="doc_id",
                db_path=str(db_path),
                # Query tokenizer mirrors the FTS5 unicode61 index (split on
                # non-alphanumerics, KEEP digits — years, catalog codes) but
                # drops English stop words on the query side only: function
                # words match real titles (e.g. "for", "a") and make
                # out-of-domain queries false-positive. sqlitesearch's default
                # ALSO splits on `\d`, which silently drops year/code terms.
                tokenizer=Tokenizer(pattern=r"[\s\W]+", stop_words="english"),
            )
            self._db_key = version
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _bm25_ranks(self, db: TextSearchIndex, query: str, corpus: str | None) -> dict[str, int]:
        # A query with no alphanumeric tokens (e.g. an LLM rewrite degraded to
        # punctuation) cannot form an FTS5 MATCH — FTS5 raises a syntax error.
        # Treat it as "no lexical match" so search degrades to the semantic
        # channel instead of 500ing.
        if not re.search(r"[A-Za-z0-9]", query):
            return {}
        filter_dict = {"corpus": corpus} if corpus else None
        # Retrieve-all at corpus scale (no magic limit*500 pool, documented).
        try:
            results = db.search(query, filter_dict=filter_dict, num_results=1_000_000)
        except Exception:  # noqa: BLE001 — any FTS5 edge falls back cleanly
            return {}
        ranks: dict[str, int] = {}
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id") or doc.get("doc_id") or doc.get("rowid")
            if doc_id is not None:
                ranks[str(doc_id)] = rank
        return ranks

    def _semantic_scores(
        self, docs_by_id: dict, vectors: np.ndarray, query: str
    ) -> dict[str, float]:
        if vectors.shape[0] == 0:
            return {}
        q = embed_query(query, self.model_name)
        scores = vectors @ q  # normalized vectors -> cosine similarity
        ids = list(docs_by_id.keys())
        return {doc_id: float(score) for doc_id, score in zip(ids, scores)}

    def recommend(
        self, seed_ids: list[str], exclude_ids: list[str], limit: int
    ) -> tuple[int, list[dict]]:
        """Catalog recommendations from stored vectors — no embedding call.

        Ranks catalog docs by cosine to the mean of the resolved seed
        embeddings; seeds and exclude_ids never appear in results. Returns
        (seeds_used, results); (0, []) when no index is built or no seed
        resolves.
        """
        version, docs_by_id, vectors = _version_artifacts(self.cache_dir)
        if version == 0 or not docs_by_id:
            return 0, []
        return recommend_rank(docs_by_id, vectors, seed_ids, exclude_ids, limit)

    def rrf_search(
        self,
        query: str,
        k: int = RRF_K,
        limit: int = 10,
        filters: dict | None = None,
        corpus: str | None = None,
        include_text: bool = False,
        method: str = "hybrid",
    ) -> list[dict]:
        """Rank the corpus by one of three retrieval approaches (D2 eval).

        `method` selects the ranking source so the same seam scores every
        approach in the multi-approach comparison:
        - "bm25": FTS5 BM25 ranks only (semantic never consulted).
        - "semantic": cosine similarity only (BM25 never consulted).
        - "hybrid": RRF k=60 fusion + exact-code pin (production default).

        Unknown methods fall back to hybrid. Filters apply BEFORE ranking in
        every method (D-03); the code pin is a hybrid-only enhancement.
        """
        method = method if method in ("bm25", "semantic") else "hybrid"
        version, docs_by_id, vectors = _version_artifacts(self.cache_dir)
        if version == 0 or not docs_by_id:
            return []

        db = self._ensure_db(version)

        # Only the selected backend is invoked (D-11): bm25-only never embeds,
        # semantic-only never runs the BM25 query.
        bm25_ranks = (
            self._bm25_ranks(db, query, corpus) if method in ("bm25", "hybrid") and db else {}
        )
        semantic_scores = (
            self._semantic_scores(docs_by_id, vectors, query)
            if method in ("semantic", "hybrid")
            else {}
        )

        # Relevance gate (MIN_SEMANTIC_SIMILARITY): when even the nearest
        # document is below the cosine threshold, the query has no relevant
        # semantic match — drop the channel so off-corpus queries return
        # nothing instead of nearest-neighbour noise. Exact codes and titles
        # keep working through BM25 + the code pin.
        if semantic_scores and max(semantic_scores.values()) < settings.min_semantic_similarity:
            semantic_scores = {}

        # Post-retrieval filtering on expanded candidate lists (D-03): an
        # excluded document must not consume a rank position or distort the RRF
        # score of surviving documents. BM25 ranks are re-indexed to contiguous
        # 1-based ranks.
        surviving_bm25 = sorted(bm25_ranks, key=bm25_ranks.get)
        bm25_ranks = {
            did: rank
            for rank, did in enumerate(
                (d for d in surviving_bm25 if passes_filters(docs_by_id[d], filters, corpus)),
                start=1,
            )
        }
        semantic_scores = {
            did: s
            for did, s in semantic_scores.items()
            if passes_filters(docs_by_id[did], filters, corpus)
        }

        ranked, scores, sem_rank_map = fuse(method, k, bm25_ranks, semantic_scores)

        if not ranked:
            return []

        # Code-exact pin (D-02 exact-match): matching catalog_code -> rank 1.
        # Hybrid-only: the enhancement that justifies hybrid over pure rankers.
        # The winning doc is marked `pinned` so later stages (re-ranking) can
        # preserve the hard exact-match rule instead of undoing it.
        pinned_id = pin_catalog_code(query, ranked, docs_by_id) if method == "hybrid" else None

        return shape_results(
            ranked,
            limit,
            docs_by_id,
            scores,
            bm25_ranks,
            sem_rank_map,
            method,
            pinned_id,
            include_text,
        )
