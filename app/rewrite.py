"""Query rewriting: LLM-based retrieval-query optimization with safe fallback.

Best-practice extra (deliverable D6a): conversational, Taglish, or verbose
user queries are rewritten into a concise keyword-style search query before
retrieval. Uses the same OpenRouter/openai client shape as ``RagService``.

Safety contract: rewriting is strictly optional — when disabled, when no API
key is configured, or when the provider fails, ``rewrite()`` returns the
original query unchanged. The retrieval flow never hard-fails on rewrite.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "Rewrite the following user query into a concise keyword-style search query "
    "optimized for an academic-paper catalog. Keep catalog codes (e.g. CEIT-IT-23-01), "
    "names, and department names exact. Output ONLY the rewritten query with no "
    "explanation or quotation marks.\n\n"
    "User query: {query}\n\n"
    "Rewritten query:"
)


class QueryRewriter:
    """Rewrites a user query for retrieval; degrades to the original query."""

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        enabled: bool | None = None,
    ):
        self._client = client
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or 64
        self._enabled = enabled if enabled is not None else settings.query_rewrite

    def rewrite(self, query: str) -> str:
        if not self._enabled or not query.strip():
            return query
        if self._client is None and not self._api_key:
            # No provider configured — original query is the best rewrite.
            return query
        try:
            client = self._ensure_client()
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a search-query optimizer."},
                    {"role": "user", "content": REWRITE_PROMPT.format(query=query)},
                ],
                max_tokens=self._max_tokens,
                temperature=0,
                stream=False,
            )
            rewritten = (response.choices[0].message.content or "").strip()
            return rewritten if rewritten else query
        except Exception as exc:  # noqa: BLE001 - fall back, never break retrieval
            logger.warning("query rewrite failed, using original query: %r", exc)
            return query

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client
