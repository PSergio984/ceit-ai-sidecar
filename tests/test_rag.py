"""RagService: ported prompt modes (rag/citations/question), context building,
one-shot answers and streaming chunks. Provider is injected/mocked — no network."""

from __future__ import annotations

import json

import pytest

from app.rag import RagService, build_context, build_prompt

RESULTS = [
    {
        "id": "paper-1",
        "corpus": "catalog",
        "title": "Analysis of Groundwater Depletion",
        "score": 0.9,
        "metadata": {"catalog_code": "CEIT-CE-15-014", "publication_year": 2015},
    },
    {
        "id": "policy-h1-r1",
        "corpus": "policy",
        "title": "General Information",
        "score": 0.7,
        "metadata": {"policy_type": "regulation"},
    },
]


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(FakeChunk(word + " ") for word in self.content.split())
        return FakeResponse(content=self.content)


class FakeDelta:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, delta: str, stream: bool):
        if stream:
            self.delta = FakeDelta(delta)
        else:
            self.message = FakeMessage(delta)


class FakeChunk:
    def __init__(self, delta: str):
        self.choices = [FakeChoice(delta, stream=True)]


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content, stream=False)]


class FakeClient:
    def __init__(self, completions: FakeCompletions):
        self.chat = type("Chat", (), {"completions": completions})()


@pytest.fixture
def fake_client():
    return FakeClient(FakeCompletions("CEIT policy states students must present their school ID. "))


def test_build_context_numbers_docs_in_order():
    ctx = build_context(RESULTS)
    lines = ctx.split("\n")
    assert lines[0].startswith("1. Analysis of Groundwater Depletion")
    assert lines[1].startswith("2. General Information")


def test_build_context_skips_empty_docs():
    assert build_context([]) == ""


def test_build_prompt_citations_has_domain_and_refusal_language():
    prompt = build_prompt("citations", "What are the ID rules?", build_context(RESULTS))
    assert "CEIT Library" in prompt
    assert "I don't have enough information" in prompt
    assert "[1], [2]" in prompt
    assert "Analysis of Groundwater Depletion" in prompt


def test_build_prompt_recommendation_requires_grounded_reasons():
    prompt = build_prompt(
        "citations", "recommend me a book", build_context(RESULTS), recommendation=True
    )
    assert "academic paper in the catalog" in prompt
    assert "one brief reason per recommendation" in prompt
    assert "Do not invent abstracts" in prompt


def test_build_prompt_question_is_conversational():
    prompt = build_prompt("question", "hi what's the borrowing limit?", build_context(RESULTS))
    assert "casual and conversational" in prompt
    assert "borrowing limit" in prompt


def test_build_prompt_rag_forbids_inventing_answers():
    prompt = build_prompt("rag", "Who may borrow?", build_context(RESULTS))
    assert "only from the provided documents" in prompt
    assert "Who may borrow?" in prompt


@pytest.mark.parametrize("mode", ["citations", "question", "rag"])
def test_build_prompt_supports_all_ported_modes(mode):
    prompt = build_prompt(mode, "q", build_context(RESULTS))
    assert "q" in prompt


def test_answer_one_shot_returns_provider_text(fake_client):
    service = RagService(client=fake_client, model="test-model", max_tokens=64)
    answer = service.answer("What are the ID rules?", RESULTS, mode="citations")

    assert answer == "CEIT policy states students must present their school ID. "
    call = fake_client.chat.completions.calls[0]
    assert call["model"] == "test-model"
    assert call["max_tokens"] == 64
    assert call["stream"] is False
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert "CEIT Library" in messages[0]["content"]
    assert "Groundwater" in messages[-1]["content"]


def test_stream_answer_yields_delta_chunks(fake_client):
    service = RagService(client=fake_client, model="test-model", max_tokens=64)
    chunks = list(service.stream_answer("What are the ID rules?", RESULTS, mode="citations"))

    assert chunks == [
        "CEIT ",
        "policy ",
        "states ",
        "students ",
        "must ",
        "present ",
        "their ",
        "school ",
        "ID. ",
    ]
    assert fake_client.chat.completions.calls[0]["stream"] is True


def test_stream_events_frames_sse_with_done(fake_client):
    service = RagService(client=fake_client, model="test-model", max_tokens=64)
    events = list(service.stream_events("q", RESULTS, mode="citations"))

    assert events[0] == 'data: {"c": "CEIT "}\n\n'
    assert events[-1] == "data: [DONE]\n\n"
    assert events[-2].startswith("data: ")


def test_stream_events_emits_error_event_on_provider_failure(fake_client):
    class Boom:
        def create(self, **kwargs):
            raise RuntimeError("provider exploded")

    fake_client.chat.completions = Boom()
    service = RagService(client=fake_client, model="test-model", max_tokens=64)

    events = list(service.stream_events("q", RESULTS, mode="citations"))

    error_events = [e for e in events if e.startswith("event: error")]
    assert error_events, "expected an error event"
    data_line = next(line for line in error_events[0].splitlines() if line.startswith("data: "))
    payload = json.loads(data_line[len("data: ") :])
    assert payload["code"] == "provider_error"
    assert events[-1] == "data: [DONE]\n\n"


def test_stream_events_refuses_on_empty_results_without_llm_call(fake_client):
    service = RagService(client=fake_client, model="test-model", max_tokens=64)

    events = list(service.stream_events("q", [], mode="citations"))

    assert events == [
        "data: I don't have enough information\n\n",
        "data: [DONE]\n\n",
    ]
    assert fake_client.chat.completions.calls == []


def test_answer_refuses_on_empty_results_without_llm_call(fake_client):
    service = RagService(client=fake_client, model="test-model", max_tokens=64)

    answer = service.answer("q", [])

    assert answer == "I don't have enough information"
    assert fake_client.chat.completions.calls == []
