"""RagService: one-shot + streamed RAG answers over hybrid-search results.

Ported from `rag-search-engine/cli/augmented_generation_cli.py` (modes rag /
citations / question) and `D:\\ai-eng\\llm-zc\\rag_helper.py` (RAGBase),
domain-parameterized for the CEIT Library (catalog + policy corpora).
See ADR 0003 for mode selection; ADR 0001 for the provider (OpenRouter via
the openai SDK); ADR 0002 for the SSE streaming contract.

Refusal is programmatic on empty retrieval (no LLM call, canonical string)
and prompt-only otherwise (source-faithful instructions per ADR 0006).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

PROMPTS: dict[str, str] = {
    "rag": (
        "You are the CEIT Library assistant.\n"
        "Your task is to provide a natural-language answer to the user's query based on "
        "documents retrieved during search.\n"
        "Provide a comprehensive answer that addresses the user's query.\n"
        "Answer only from the provided documents; if the documents do not contain the "
        'answer, say "I don\'t have enough information".\n\n'
        "Query: {query}\n\n"
        "Documents:\n{docs}\n\n"
        "Answer:"
    ),
    "citations": (
        "You are the CEIT Library assistant. Answer the query below using ONLY the "
        "provided documents.\n\n"
        "Query: {query}\n\n"
        "Documents:\n{docs}\n\n"
        "Instructions:\n"
        "- Provide a comprehensive answer that addresses the query\n"
        "- Cite sources in the format [1], [2], etc. when referencing information\n"
        "- If sources disagree, mention the different viewpoints\n"
        "- If the answer isn't in the provided documents, say \"I don't have enough information\"\n"
        "- Be direct and informative\n\n"
        "Answer:"
    ),
    "question": (
        "Answer the user's question about the CEIT Library based on the provided documents.\n\n"
        "Question: {question}\n\n"
        "Documents:\n{context}\n\n"
        "Instructions:\n"
        "- Answer questions directly and concisely\n"
        "- Be casual and conversational\n"
        "- Don't be cringe or hype-y\n"
        "- Talk like a normal person would in a chat conversation\n\n"
        "Answer:"
    ),
}

MAX_DOC_CHARS = 600

# Wire key for the JSON-encoded SSE chunk envelope — the Laravel parser's
# counterpart constant lives in AiService::SSE_CHUNK_KEY.
CHUNK_KEY = "c"

SYSTEM_PROMPT = (
    "You are the CEIT Library assistant. Answer only from the provided documents; "
    'if the documents do not contain the answer, say "I don\'t have enough information".'
)


def build_context(results: list[dict]) -> str:
    """Number the retrieved docs for the prompt: `{i}. {title} - {text}`."""
    blocks = []
    for i, doc in enumerate(results, start=1):
        text = (doc.get("text") or doc.get("title") or "").strip().replace("\n", " ")
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS].rstrip() + "…"
        blocks.append(f"{i}. {doc.get('title', '')} - {text}")
    return "\n".join(blocks)


def build_prompt(mode: str, query: str, docs: str) -> str:
    """Build the mode prompt; defaults to the citations prompt on unknown modes."""
    template = PROMPTS.get(mode, PROMPTS["citations"])
    return template.format(query=query, question=query, docs=docs, context=docs)


class RagService:
    """Runs the LLM over hybrid-search results (one-shot or streamed)."""

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ):
        self._client = client
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or settings.llm_max_tokens

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def _messages(self, query: str, results: list[dict], mode: str) -> list[dict]:
        docs = build_context(results)
        prompt = build_prompt(mode, query, docs)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def answer(self, query: str, results: list[dict], mode: str = "citations") -> str:
        """One-shot non-streamed answer (RAGBase `rag()` shape)."""
        if not results:
            return "I don't have enough information"
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=self._messages(query, results, mode),
            max_tokens=self._max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def stream_answer(
        self, query: str, results: list[dict], mode: str = "citations"
    ) -> Iterator[str]:
        """Yield answer text chunks from the provider's stream."""
        client = self._ensure_client()
        stream = client.chat.completions.create(
            model=self._model,
            messages=self._messages(query, results, mode),
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def stream_events(
        self, query: str, results: list[dict], mode: str = "citations"
    ) -> Iterator[str]:
        """SSE-framed events: `data: <chunk>` lines, `[DONE]` terminator, or
        an `event: error` line on provider failure (ADR 0002 framing).

        Chunk payloads are JSON-encoded (`{"c": "<delta>"}`) so provider
        deltas containing newlines cannot break the SSE line framing —
        text framing would silently lose paragraph breaks (review W-1).
        """
        if not results:
            yield "data: I don't have enough information\n\n"
            yield "data: [DONE]\n\n"
            return
        try:
            for delta in self.stream_answer(query, results, mode):
                payload = json.dumps({CHUNK_KEY: delta}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        except Exception as exc:  # noqa: BLE001 - provider errors become SSE error events
            logger.error(repr(exc))
            error_payload = {
                "code": "provider_error",
                "message": "The AI provider is temporarily unavailable. Please try again.",
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
        yield "data: [DONE]\n\n"
