"""AgenticLoop: tool-loop cap, closed-schema arg validation, fail-closed, frames.

Uses the same fake stack as test_chat_stream.py (FakeCompletionsHolder) plus a
tool_calls variant of the completion message — the only new fake surface — and a
FakeEngine double that records rrf_search calls.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.agent import (
    MAX_TOOL_ROUNDS,
    SEARCH_TOOL,
    AgenticLoop,
    ToolArgs,
    citation_payload,
    merge_dedup,
)

DOC1 = {
    "id": "paper-1",
    "corpus": "catalog",
    "title": "Analysis of Groundwater Depletion",
    "text": "Analysis of Groundwater Depletion Caused By Excessive Use of Water Pumps.",
    "metadata": {
        "url": "/academic-papers/1",
        "catalog_code": "CEIT-CE-15-014",
        "authors": ["Juan Dela Cruz"],
        "publication_year": 2015,
    },
}

DOC2 = {
    "id": "paper-2",
    "corpus": "catalog",
    "title": "Design of a Smart Flood Monitoring System",
    "text": "Design of a Smart Flood Monitoring System.",
    "metadata": {
        "url": "/academic-papers/2",
        "catalog_code": "CEIT-EE-25-01",
        "authors": ["Maria Santos"],
        "publication_year": 2025,
    },
}


def _stream_chunk(delta: str):
    return type(
        "Chunk", (), {"choices": [type("C", (), {"delta": type("D", (), {"content": delta})()})()]}
    )()


def _chunks(content: str):
    for word in content.split():
        yield _stream_chunk(word + " ")


def _tool_call(arguments: str):
    return type(
        "TC",
        (),
        {"id": "call_1", "function": type("F", (), {"name": "search", "arguments": arguments})()},
    )()


def _response(content: str, tool_calls=None):
    return type(
        "Resp",
        (),
        {
            "choices": [
                type("C", (), {"message": type("M", (), {"content": content, "tool_calls": tool_calls})()})()
            ]
        },
    )()


class FakeCompletions:
    def __init__(self, content: str = "", tool_sequence: list | None = None, fail: bool = False):
        self.content = content
        self.tool_sequence = list(tool_sequence or [])
        self.fail = fail
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.tool_sequence:
            item = self.tool_sequence.pop(0)
            if item == "FAIL":
                raise RuntimeError("provider exploded")
            return _response(self.content, tool_calls=[_tool_call(item)])
        if self.fail:
            raise RuntimeError("provider exploded")
        if kwargs.get("stream"):
            return iter(_chunks(self.content))
        return _response(self.content)


class FakeCompletionsHolder:
    def __init__(self, content: str = "", tool_sequence: list | None = None, fail: bool = False):
        self.chat = type(
            "Chat", (), {"completions": FakeCompletions(content, tool_sequence, fail)}()
        )()


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.calls: list[dict] = []

    def rrf_search(self, query, k=60, limit=10, filters=None, corpus=None, include_text=False):
        self.calls.append(
            {
                "query": query,
                "k": k,
                "limit": limit,
                "filters": filters,
                "corpus": corpus,
                "include_text": include_text,
            }
        )
        return self.results


def make_loop(content="", tool_sequence=None, results=(), fail=False):
    client = FakeCompletionsHolder(content=content, tool_sequence=tool_sequence, fail=fail)
    engine = FakeEngine(list(results))
    loop = AgenticLoop(client=client, engine=engine, model="test-model", max_tokens=64)
    return client, engine, loop


def _activity_lines(events: list[str]) -> list[str]:
    lines = []
    for event in events:
        if event.startswith("event: activity\n"):
            lines.append(json.loads(event.split("\ndata: ", 1)[1].strip())["text"])
    return lines


def test_direct_answer_streams_without_search():
    client, engine, loop = make_loop(content="Hello there. ")
    events = list(loop.stream_agentic_events("hello"))
    assert any('"c": "Hello ' in event for event in events)
    assert any(event == "data: [DONE]\n\n" for event in events)
    assert engine.calls == []
    assert all("event: activity" not in event for event in events)
    assert all("event: citations" not in event for event in events)


def test_tool_call_triggers_search_then_answer():
    args = json.dumps(
        {
            "query": "papers by juan dela cruz",
            "filters": {"author": "juan dela cruz"},
            "corpus": "catalog",
            "top_k": 7,
        }
    )
    client, engine, loop = make_loop(
        content="The papers by Juan Dela Cruz cover groundwater. ",
        tool_sequence=[args],
        results=[DOC1, DOC2],
    )
    events = list(loop.stream_agentic_events("papers by juan dela cruz"))

    decision = client.chat.completions.calls[0]
    assert decision["tools"] == [SEARCH_TOOL]
    assert decision["tool_choice"] == "auto"
    assert decision["stream"] is False

    assert engine.calls == [
        {
            "query": "papers by juan dela cruz",
            "k": 60,
            "limit": 7,
            "filters": {"author": "juan dela cruz"},
            "corpus": "catalog",
            "include_text": True,
        }
    ]

    citations_event = next(e for e in events if e.startswith("event: citations\n"))
    payload = json.loads(citations_event.split("\ndata: ", 1)[1].strip())
    assert payload == citation_payload([DOC1, DOC2])

    assert any('"c": "The papers by ' in event for event in events)
    assert events[-1] == "data: [DONE]\n\n"


def test_loop_caps_at_three_rounds_and_fails_closed():
    args = json.dumps({"query": "multi-hop question"})
    client, engine, loop = make_loop(
        content="Final grounded answer. ",
        tool_sequence=[args, args, args, args],
        results=[DOC1],
    )
    events = list(loop.stream_agentic_events("multi-hop question"))

    assert len(engine.calls) == MAX_TOOL_ROUNDS
    assert _activity_lines(events) == ["Searching papers…", "Narrowing results…", "Narrowing results…"]
    assert any('"c": "Final grounded answer. "' in event for event in events)
    citations_event = next(e for e in events if e.startswith("event: citations\n"))
    payload = json.loads(citations_event.split("\ndata: ", 1)[1].strip())
    assert payload == citation_payload([DOC1])
    assert events[-1] == "data: [DONE]\n\n"


def test_loop_caps_at_three_rounds_with_zero_docs_fails_closed_refusal():
    args = json.dumps({"query": "multi-hop question"})
    client, engine, loop = make_loop(
        content="should never stream",
        tool_sequence=[args, args, args, args],
        results=[],
    )
    events = list(loop.stream_agentic_events("multi-hop question"))

    assert len(engine.calls) == MAX_TOOL_ROUNDS
    assert events[-2] == "data: I don't have enough information\n\n"
    assert events[-1] == "data: [DONE]\n\n"
    assert not any('"c": ' in event for event in events)
    assert not any(c.get("stream") for c in client.chat.completions.calls)


def test_malformed_tool_args_correct_once_then_fail_closed():
    bad = json.dumps({"query": "papers", "bogus_filter": "x"})
    client, engine, loop = make_loop(content="never streams", tool_sequence=[bad, bad], results=[DOC1])
    events = list(loop.stream_agentic_events("papers"))

    assert engine.calls == []
    assert events[-2] == "data: I don't have enough information\n\n"
    assert events[-1] == "data: [DONE]\n\n"
    assert not any('"c": ' in event for event in events)
    assert client.chat.completions.calls[1]["messages"][-1]["role"] == "tool"
    assert "invalid arguments" in client.chat.completions.calls[1]["messages"][-1]["content"]


def test_activity_and_citations_frame_ordering():
    args = json.dumps({"query": "papers by juan dela cruz", "filters": {"author": "juan dela cruz"}})
    client, engine, loop = make_loop(
        content="Answer text. ",
        tool_sequence=[args, args],
        results=[DOC1, DOC2],
    )
    events = list(loop.stream_agentic_events("papers by juan dela cruz"))

    def index_of(predicate):
        return next(i for i, e in enumerate(events) if predicate(e))

    activity_i = index_of(lambda e: e.startswith("event: activity"))
    chunk_i = index_of(lambda e: e.startswith('data: {"c"'))
    citations_i = index_of(lambda e: e.startswith("event: citations"))
    done_i = index_of(lambda e: e.startswith("data: [DONE]"))
    assert activity_i < chunk_i < citations_i < done_i

    assert _activity_lines(events) == ["Searching papers by author…", "Narrowing results…"]

    citations_event = next(e for e in events if e.startswith("event: citations\n"))
    payload = json.loads(citations_event.split("\ndata: ", 1)[1].strip())
    assert [c["n"] for c in payload] == [1, 2]


def test_activity_copy_lines_for_corpus_and_year():
    policy_args = json.dumps({"query": "school id", "corpus": "policy"})
    catalog_args = json.dumps({"query": "flood", "corpus": "catalog"})
    year_args = json.dumps({"query": "papers", "filters": {"year_from": 2015, "year_to": 2020}})
    client, engine, loop = make_loop(content="x ", tool_sequence=[policy_args, catalog_args, year_args], results=[DOC1])
    events = list(loop.stream_agentic_events("q"))
    assert _activity_lines(events) == [
        "Searching policy documents…",
        "Searching the catalog…",
        "Searching papers from 2015–2020…",
    ]


def test_merge_dedup_keeps_first_seen_order():
    merged = merge_dedup([], [DOC1, DOC2])
    merged = merge_dedup(merged, [DOC2, DOC1, DOC1])
    assert [d["id"] for d in merged] == ["paper-1", "paper-2"]


def test_tool_args_rejects_unknown_keys():
    with pytest.raises(ValidationError):
        ToolArgs.model_validate_json(json.dumps({"query": "q", "mystery": 1}))
    with pytest.raises(ValidationError):
        ToolArgs.model_validate_json(json.dumps({"query": "q", "filters": {"sneaky": "x"}}))
