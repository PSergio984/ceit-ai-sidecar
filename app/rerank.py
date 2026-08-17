"""Re-ranking of the top-k candidates (best-practice extra, deliverable D6b).

Two re-rankers, both running AFTER retrieval on the small top-k set:

- ``blend`` (default): deterministic second pass that re-orders candidates by
  a consensus+rank key — documents retrieved by BOTH the BM25 and semantic
  retrievers rank above single-retriever hits, ties broken by the sum of the
  two ranks. Zero extra latency, zero model, fits the 500 MB cloud budget.
- ``llm``: RankGPT-style listwise re-ranking — the configured LLM re-orders
  the numbered candidates by relevance (cross-encoder behaviour without a
  second model). Uses the same OpenRouter client shape as ``RagService``.

Safety contract: mode ``none`` or any provider failure returns the input order
unchanged — re-ranking never drops or fabricates documents.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_RERANK_DOCS = 20
RERANK_MAX_DOC_CHARS = 400

RERANK_PROMPT = (
    "Rank the numbered documents below by relevance to the user's query, most "
    "relevant first. Respond with ONLY a comma-separated list of numbers in "
    "descending relevance order, e.g. 3, 1, 2.\n\n"
    "Query: {query}\n\n"
    "{documents}\n\n"
    "Relevant order:"
)


def _blend_key(doc: dict) -> tuple:
    bm25 = doc.get("bm25_rank")
    sem = doc.get("semantic_rank")
    if bm25 is not None and sem is not None:
        return (0, bm25 + sem)
    if bm25 is not None:
        return (1, bm25)
    if sem is not None:
        return (2, sem)
    return (3, 0)


def _parse_reorder(payload: str) -> list[int]:
    """Extract the first run of integers from the LLM reorder payload."""
    match = re.search(r"[\d,\s]+", payload or "")
    if not match:
        return []
    return [int(n) for n in re.findall(r"\d+", match.group(0)) if int(n) >= 1]


class Reranker:
    """Re-orders retrieved candidates; always returns the same docs back."""

    def __init__(
        self,
        *,
        mode: str | None = None,
        client: OpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        mode = mode if mode is not None else settings.rerank_mode
        self._mode = mode if mode in ("none", "blend", "llm") else "none"
        self._client = client
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or 128

    def rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Return a re-ordered copy of `results` (never drops or invents docs)."""
        if self._mode == "none" or not results:
            return list(results)
        if self._mode == "llm":
            return self._llm_rerank(query, list(results))
        return sorted(results, key=_blend_key)

    def _llm_rerank(self, query: str, results: list[dict]) -> list[dict]:
        if self._client is None and not self._api_key:
            return results
        documents = []
        for i, doc in enumerate(results[:MAX_RERANK_DOCS], start=1):
            text = (doc.get("text") or doc.get("title") or "").strip().replace("\n", " ")
            if len(text) > RERANK_MAX_DOC_CHARS:
                text = text[:RERANK_MAX_DOC_CHARS].rstrip() + "…"
            documents.append(f"{i}. {doc.get('title', '')} - {text}")
        try:
            client = self._ensure_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a retrieval re-ranker."},
                    {
                        "role": "user",
                        "content": RERANK_PROMPT.format(
                            query=query, documents="\n".join(documents)
                        ),
                    },
                ],
                max_tokens=self._max_tokens,
                temperature=0,
                stream=False,
            )
            payload = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001 - fall back to original order
            logger.warning("LLM re-rank failed, keeping original order: %r", exc)
            return results

        order = _parse_reorder(payload)
        if not order:
            return results
        by_number = {i + 1: doc for i, doc in enumerate(results[:MAX_RERANK_DOCS])}
        reordered: list[dict] = []
        seen: set[int] = set()
        for n in order:
            if n in by_number and n not in seen:
                reordered.append(by_number[n])
                seen.add(n)
        # Docs the model did not place keep their original relative order.
        reordered.extend(doc for i, doc in enumerate(results, start=1) if i not in seen)
        return reordered

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client
