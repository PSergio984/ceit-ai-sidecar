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
from sqlitesearch import TextSearchIndex

from .ingest import embed_query

CODE_PIN_RE = re.compile(r"^CEIT-[A-Z]{2}-\d{2}(-\d+)?$", re.IGNORECASE)

AUTHOR_KEYS = ("research_adviser", "technical_adviser")


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
            )
            self._db_key = version
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def _bm25_ranks(self, db: TextSearchIndex, query: str, corpus: str | None) -> dict[str, int]:
        filter_dict = {"corpus": corpus} if corpus else None
        # Retrieve-all at corpus scale (no magic limit*500 pool, documented).
        results = db.search(query, filter_dict=filter_dict, num_results=1_000_000)
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

    def rrf_search(
        self,
        query: str,
        k: int = 60,
        limit: int = 10,
        filters: dict | None = None,
        corpus: str | None = None,
        include_text: bool = False,
    ) -> list[dict]:
        version, docs_by_id, vectors = _version_artifacts(self.cache_dir)
        if version == 0 or not docs_by_id:
            return []

        db = self._ensure_db(version)

        bm25_ranks = self._bm25_ranks(db, query, corpus) if db else {}
        semantic_scores = self._semantic_scores(docs_by_id, vectors, query)

        # Post-retrieval filtering on expanded candidate lists (D-03).
        def passes(doc: dict) -> bool:
            meta = doc.get("metadata") or {}
            f = filters or {}
            if f.get("paper_type") and meta.get("paper_type") != f["paper_type"]:
                return False
            if f.get("department") and meta.get("department") != f["department"]:
                return False
            if f.get("publication_year") and meta.get("publication_year") != int(
                f["publication_year"]
            ):
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

        candidate_ids = set(bm25_ranks) | set(semantic_scores)
        candidate_ids = {did for did in candidate_ids if passes(docs_by_id[did])}

        if not candidate_ids:
            return []

        # RRF k=60 fusion.
        rrf_scores: dict[str, float] = {}
        for did in candidate_ids:
            score = 0.0
            if did in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[did])
            if did in semantic_scores:
                sem_rank = (
                    sum(
                        1
                        for other in semantic_scores
                        if semantic_scores[other] > semantic_scores[did]
                    )
                    + 1
                )
                score += 1.0 / (k + sem_rank)
            rrf_scores[did] = score

        ranked = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)

        # Code-exact pin (D-02 exact-match): matching catalog_code -> rank 1.
        upper_query = query.upper()
        if CODE_PIN_RE.match(upper_query):
            for did in ranked:
                meta = docs_by_id[did].get("metadata") or {}
                if (meta.get("catalog_code") or "").upper() == upper_query:
                    ranked.remove(did)
                    ranked.insert(0, did)
                    break

        results = []
        for did in ranked[:limit]:
            doc = docs_by_id[did]
            meta = doc.get("metadata") or {}
            result = {
                "id": doc["id"],
                "corpus": doc.get("corpus"),
                "title": doc.get("title"),
                "score": round(rrf_scores[did], 4),
                "bm25_rank": bm25_ranks.get(did),
                "semantic_rank": (
                    sum(
                        1
                        for other in semantic_scores
                        if semantic_scores[other] > semantic_scores[did]
                    )
                    + 1
                    if did in semantic_scores
                    else None
                ),
                "metadata": meta,
            }
            if include_text:
                result["text"] = doc.get("text", "")
            results.append(result)
        return results
