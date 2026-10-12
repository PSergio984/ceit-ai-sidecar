"""Failover tests for app.llm.FailoverOpenAI — fake clients, no network."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

import app.llm as llm_mod
from app.config import Settings
from app.llm import FailoverOpenAI, ensure_openai_client


@pytest.fixture(autouse=True)
def _reset_shared_index():
    llm_mod._good_index.clear()
    yield
    llm_mod._good_index.clear()


def _status_error(status: int) -> openai.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "http://test.invalid"))
    if status == 429:
        return openai.RateLimitError("rate limited", response=response, body=None)
    if status == 400:
        return openai.BadRequestError("bad request", response=response, body=None)
    if status == 401:
        return openai.AuthenticationError("unauthorized", response=response, body=None)
    if status >= 500:
        return openai.InternalServerError("server error", response=response, body=None)
    return openai.APIStatusError(f"status {status}", response=response, body=None)


class _FakeCompletions:
    def __init__(self, errors: list[Exception] | None = None, result: object = "ok"):
        self.errors = list(errors or [])
        self.result = result
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return self.result


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=_FakeCompletions(**kwargs))


def _patch_factory(monkeypatch, clients: dict[str, _FakeClient]) -> list[tuple[str, str]]:
    made: list[tuple[str, str]] = []

    def factory(base_url: str, key: str):
        made.append((base_url, key))
        return clients[key]

    monkeypatch.setattr(llm_mod, "_make_client", factory)
    return made


def test_first_key_rate_limited_rotates_to_second(monkeypatch):
    clients = {
        "k1": _FakeClient(errors=[_status_error(429)]),
        "k2": _FakeClient(result="answer"),
    }
    made = _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["k1", "k2"])

    assert client.chat.completions.create(model="m") == "answer"
    assert len(clients["k1"].chat.completions.calls) == 1
    assert len(clients["k2"].chat.completions.calls) == 1
    assert made == [("http://base", "k1"), ("http://base", "k2")]


@pytest.mark.parametrize("status", [401, 402])
def test_auth_and_payment_errors_rotate(monkeypatch, status):
    clients = {
        "k1": _FakeClient(errors=[_status_error(status)]),
        "k2": _FakeClient(result="answer"),
    }
    _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["k1", "k2"])

    assert client.chat.completions.create() == "answer"
    assert len(clients["k2"].chat.completions.calls) == 1


@pytest.mark.parametrize("status", [400, 500])
def test_non_limit_errors_do_not_rotate(monkeypatch, status):
    clients = {
        "k1": _FakeClient(errors=[_status_error(status)]),
        "k2": _FakeClient(result="answer"),
    }
    _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["k1", "k2"])

    with pytest.raises(openai.APIStatusError):
        client.chat.completions.create()
    assert len(clients["k1"].chat.completions.calls) == 1
    assert len(clients["k2"].chat.completions.calls) == 0


def test_all_keys_fail_raises_last_error_each_tried_once(monkeypatch):
    first, last = _status_error(429), _status_error(402)
    clients = {
        "k1": _FakeClient(errors=[first]),
        "k2": _FakeClient(errors=[_status_error(429)]),
        "k3": _FakeClient(errors=[last]),
    }
    _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["k1", "k2", "k3"])

    with pytest.raises(openai.APIStatusError) as excinfo:
        client.chat.completions.create()

    assert excinfo.value is last
    for fake in clients.values():
        assert len(fake.chat.completions.calls) == 1


def test_next_call_after_total_failure_recovers(monkeypatch):
    clients = {
        "k1": _FakeClient(errors=[_status_error(429)]),
        "k2": _FakeClient(errors=[_status_error(429)]),
    }
    _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["k1", "k2"])
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create()

    # Limits "reset": the same keys succeed on the next call.
    clients["k1"].chat.completions.errors = []
    clients["k2"].chat.completions.errors = []

    assert client.chat.completions.create() == "ok"


def test_shared_index_remembers_working_key_across_calls(monkeypatch):
    clients = {
        "k1": _FakeClient(errors=[_status_error(429)]),
        "k2": _FakeClient(result="answer"),
    }
    _patch_factory(monkeypatch, clients)

    assert FailoverOpenAI("http://base", ["k1", "k2"]).chat.completions.create() == "answer"

    # A fresh wrapper for the same (base_url, keys) starts from k2 — k1 is
    # not retried again.
    fresh = FailoverOpenAI("http://base", ["k1", "k2"])
    assert fresh.chat.completions.create() == "answer"
    assert len(clients["k1"].chat.completions.calls) == 1
    assert len(clients["k2"].chat.completions.calls) == 2


def test_single_key_behaves_as_plain_client(monkeypatch):
    clients = {"only": _FakeClient(result="answer")}
    _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["only"])
    assert client.chat.completions.create() == "answer"

    clients["only"].chat.completions.errors = [_status_error(429)]
    with pytest.raises(openai.RateLimitError):
        client.chat.completions.create()
    assert len(clients["only"].chat.completions.calls) == 2


def test_llm_key_list_parsing():
    settings = Settings(sidecar_token="t", llm_api_keys=" a , b ,, a ,c ", llm_api_key="x")
    assert settings.llm_key_list == ["a", "b", "c", "x"]

    assert Settings(sidecar_token="t", llm_api_keys="", llm_api_key="k").llm_key_list == ["k"]
    assert Settings(sidecar_token="t", llm_api_keys="k2", llm_api_key="k2").llm_key_list == ["k2"]
    assert Settings(sidecar_token="t", llm_api_keys="", llm_api_key="").llm_key_list == [""]


def test_ensure_openai_client_returns_injected_untouched():
    injected = object()
    assert ensure_openai_client(injected, "http://base", "k") is injected


def test_ensure_openai_client_uses_settings_key_list(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_keys", "ka, kb")
    monkeypatch.setattr(settings, "llm_api_key", "kc")

    client = ensure_openai_client(None, "http://base", settings.llm_api_key)
    assert isinstance(client, FailoverOpenAI)
    assert client._api_keys == ["ka", "kb", "kc"]


def test_ensure_openai_client_explicit_key_gets_single_key_wrapper(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_keys", "ka, kb")

    other_key = f"{settings.llm_api_key}-different"
    client = ensure_openai_client(None, "http://base", other_key)
    assert isinstance(client, FailoverOpenAI)
    assert client._api_keys == [other_key]


def test_key_values_never_appear_in_logs_or_exceptions(monkeypatch, caplog):
    clients = {
        "sk-secret-one": _FakeClient(errors=[_status_error(429)]),
        "sk-secret-two": _FakeClient(errors=[_status_error(401)]),
    }
    _patch_factory(monkeypatch, clients)

    client = FailoverOpenAI("http://base", ["sk-secret-one", "sk-secret-two"])

    with caplog.at_level(logging.WARNING), pytest.raises(openai.APIStatusError) as excinfo:
        client.chat.completions.create()

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "key 1/2" in logged
    assert "key 2/2" in logged
    assert "sk-secret-one" not in logged
    assert "sk-secret-two" not in logged
    assert "sk-secret-one" not in str(excinfo.value)
    assert "sk-secret-two" not in str(excinfo.value)
