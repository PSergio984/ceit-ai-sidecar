"""AgenticLoop: bounded function-calling search loop over the closed /search contract.

CHAT-05 (ADR 0014): the first LLM call of every /chat/stream request is a single
non-streamed tool-eligible call (`tools=[SEARCH_TOOL]`, `tool_choice="auto"`).
Tool-use in the response IS the auto-detect — no classifier. Each executed tool
round validates args against the closed schema (pydantic `extra="forbid"`) before
calling `rrf_search` on the same seam the one-shot path used; results merge into a
deduped doc set renumbered 1..N for the citations frame. The loop is capped at
MAX_TOOL_ROUNDS executed searches; on cap or malformed-args failure it fails
closed to the canonical ADR 0006 refusal (zero LLM calls) or an answer grounded in
the accumulated docs. SSE framing stays additive within ADR 0002: `event: activity`
/ `event: citations` frames, unchanged chunk envelope and `event: error` taxonomy.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import settings
from .rag import (
    CITATION_KEYS,
    MAX_DOC_CHARS,
    PROMPTS,
    SYSTEM_PROMPT,
    build_context,
    chunk_frame,
)
from .search import HybridSearch

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Counts EXECUTED searches (D-11: initial retrieval + 2 refinements).
MAX_TOOL_ROUNDS = 3

SEARCH_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the CEIT Library catalog (academic papers) or policy rulebook. "
            "Use when the user's question needs retrieved documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "corpus": {"type": "string", "enum": ["catalog", "policy"]},
                "filters": {
                    "type": "object",
                    "properties": {
                        "paper_type": {"type": "string"},
                        "department": {"type": "string"},
                        "publication_year": {"type": "integer"},
                        "year_from": {"type": "integer"},
                        "year_to": {"type": "integer"},
                        "author": {"type": "string"},
                        "adviser": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class ToolFilterArgs(BaseModel):
    """Closed filter schema — unknown keys are rejected before execution (D-09)."""

    model_config = ConfigDict(extra="forbid")

    paper_type: str | None = None
    department: str | None = None
    publication_year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    author: str | None = None
    adviser: str | None = None


class ToolArgs(BaseModel):
    """Server-side mirror of the search tool spec — `extra="forbid"` both levels."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    corpus: Literal["catalog", "policy"] | None = None
    filters: ToolFilterArgs | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


def merge_dedup(docs: list[dict], new_docs: list[dict]) -> list[dict]:
    """Append results whose id is not already present (first-seen order)."""
    seen = {doc["id"] for doc in docs}
    for doc in new_docs:
        if doc["id"] not in seen:
            docs.append(doc)
            seen.add(doc["id"])
    return docs


def citation_payload(docs: list[dict]) -> list[dict]:
    """ADR 0006 payload: the numbered set the final prompt worked from (1..N).

    Key set comes from the shared rag.CITATION_KEYS literal so the payload
    shape cannot drift from the Laravel-side checker (AiService::CITATION_KEYS).
    """
    return [
        {key: value for key, value in zip(CITATION_KEYS, _citation_values(doc, i + 1))}
        for i, doc in enumerate(docs)
    ]


def _citation_values(doc: dict, n: int) -> tuple:
    return (
        n,
        doc["id"],
        doc["corpus"],
        doc["title"],
        (doc.get("metadata") or {}).get("url"),
        (doc.get("metadata") or {}).get("catalog_code"),
    )


def activity_line(args: ToolArgs, executed_rounds: int, corpus: str | None = None) -> str:
    """Static copy per UI-SPEC — never args/results JSON (D-12, T-11-11).

    Precedence: per-filter copy, then corpus copy, then refinement/generic
    fallbacks — a filtered refinement still names its filter (UI-SPEC rows
    are not round-scoped). `corpus` is the effective corpus (request corpus
    when the tool call omitted it, Spec review S-5).
    """
    filters = args.filters
    if filters and filters.author:
        return "Searching papers by author…"
    if filters and filters.adviser:
        return "Searching papers by adviser…"
    if filters and (filters.year_from is not None or filters.year_to is not None):
        return f"Searching papers from {filters.year_from}–{filters.year_to}…"
    effective_corpus = args.corpus or corpus
    if effective_corpus == "policy":
        return "Searching policy documents…"
    if effective_corpus == "catalog":
        return "Searching the catalog…"
    if executed_rounds > 0:
        return "Narrowing results…"
    return "Searching papers…"


def _activity_frame(line: str) -> str:
    return f"event: activity\ndata: {json.dumps({'text': line}, ensure_ascii=False)}\n\n"


def _chunk_frames(text: str) -> Iterator[str]:
    for word in text.split():
        yield chunk_frame(word + " ")


def _assistant_tool_message(tool_calls) -> dict:
    # WR-1: content and tool_calls are mutually exclusive per the OpenAI
    # contract — null content alongside tool_calls avoids provider 400s.
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in tool_calls
        ],
    }


def _truncate_docs_for_tool(results: list[dict]) -> list[dict]:
    """Tool-result copy with doc text capped at MAX_DOC_CHARS (WR-2).

    The final prompt still reads the full-text docs through build_context;
    only what ships in the tool-result wire message is truncated.
    """
    truncated = []
    for doc in results:
        copy = dict(doc)
        text = copy.get("text") or ""
        if len(text) > MAX_DOC_CHARS:
            copy["text"] = text[:MAX_DOC_CHARS].rstrip() + "…"
        truncated.append(copy)
    return truncated


class AgenticLoop:
    """Bounded search loop in the RagService injectable-client shape.

    Constructor takes client/engine/model/max_tokens/prompts with settings
    fallbacks; `client` is lazily materialized (`_ensure_client()`), `engine`
    defaults to the production HybridSearch wiring (same as main._get_engine).
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        engine: HybridSearch | None = None,
        prompts: dict[str, str] | None = None,
    ):
        self._client = client
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._model = model or settings.llm_model
        self._max_tokens = max_tokens or settings.llm_max_tokens
        self._engine = engine
        self._prompts = prompts or PROMPTS

    def _ensure_client(self) -> OpenAI:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def _engine_or_default(self) -> HybridSearch:
        if self._engine is None:
            self._engine = HybridSearch(Path(settings.cache_dir), settings.model_name)
        return self._engine

    def _final_prompt(self, mode: str, query: str, docs_context: str) -> str:
        template = self._prompts.get(mode, self._prompts["citations"])
        return template.format(query=query, question=query, docs=docs_context, context=docs_context)

    def _stream_final_answer(self, query: str, docs: list[dict], mode: str) -> Iterator[str]:
        client = self._ensure_client()
        prompt = self._final_prompt(mode, query, build_context(docs))
        stream = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield chunk_frame(delta)

    def stream_agentic_events(
        self, query: str, mode: str = "citations", corpus: str | None = None, default_top_k: int = 5
    ) -> Iterator[str]:
        """Yield raw SSE event strings for one agentic turn (ADR 0002 framing).

        `corpus` is the request-scoped corpus from the /chat/stream payload
        (ADR 0004 — absent = both). It becomes the DEFAULT corpus for tool
        calls that omit `corpus`; an explicit tool-call corpus still wins, so
        the request scope is honored unless the model deliberately widens it.
        `default_top_k` is the endpoint's top_k contract value used when a tool
        call omits `top_k`; the loop itself never adds history over the wire.
        """
        client = self._ensure_client()
        engine = self._engine_or_default()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        docs: list[dict] = []
        rounds = 0
        malformed_streak = 0
        try:
            while True:
                resp = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=[SEARCH_TOOL],
                    tool_choice="auto",
                    max_tokens=self._max_tokens,
                    stream=False,
                )
                msg = resp.choices[0].message
                tool_calls = msg.tool_calls or []
                if not tool_calls:
                    if rounds == 0:
                        # Direct answer (D-07): no search happened, no frames —
                        # Laravel falls back to companionCitations (ADR 0014).
                        yield from _chunk_frames(msg.content or "")
                        yield "data: [DONE]\n\n"
                        return
                    break
                if rounds >= MAX_TOOL_ROUNDS:
                    break
                # WR-1: execute EVERY call of a parallel response (capped at
                # the remaining round budget); calls beyond the cap never
                # enter messages, so no unmatched tool_call_id can 400 the
                # next provider call.
                calls = tool_calls[: MAX_TOOL_ROUNDS - rounds]
                messages.append(_assistant_tool_message(calls))
                malformed_abort = False
                for call in calls:
                    try:
                        args = ToolArgs.model_validate_json(call.function.arguments)
                    except ValidationError as exc:
                        malformed_streak += 1
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": (
                                    "Error: invalid arguments for the search tool "
                                    f"({exc}). Correct the arguments and call search once."
                                ),
                            }
                        )
                        if malformed_streak >= 2:
                            malformed_abort = True
                            break
                        continue
                    malformed_streak = 0
                    effective_corpus = args.corpus or corpus
                    yield _activity_frame(activity_line(args, rounds, effective_corpus))
                    results = engine.rrf_search(
                        query=args.query,
                        k=60,
                        limit=args.top_k or default_top_k,
                        filters=args.filters.model_dump(exclude_none=True) if args.filters else {},
                        corpus=effective_corpus,
                        include_text=True,
                    )
                    docs = merge_dedup(docs, results)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(
                                _truncate_docs_for_tool(results), ensure_ascii=False
                            ),
                        }
                    )
                    rounds += 1
                if malformed_abort:
                    break

            # Final answer after tool rounds: grounded in accumulated docs,
            # or the zero-token canonical refusal when nothing was retrieved.
            if not docs:
                yield "data: I don't have enough information\n\n"
                yield "data: [DONE]\n\n"
                return
            yield from self._stream_final_answer(query, docs, mode)
            yield f"event: citations\ndata: {json.dumps(citation_payload(docs), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001 - provider errors become SSE error events
            logger.error(repr(exc))
            error_payload = {
                "code": "provider_error",
                "message": "The AI provider is temporarily unavailable. Please try again.",
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            yield "data: [DONE]\n\n"
