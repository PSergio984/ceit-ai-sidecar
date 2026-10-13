"""QueryRewriter: LLM query rewriting with a safe fallback (deliverable D6a).

Seam: `QueryRewriter.rewrite(query)` — the public interface the /search flow
calls. Expected rewritten values come from the fake provider (independent of
the implementation). The contract:
- enabled + provider returns text -> that text is used.
- provider fails / no key / disabled -> the ORIGINAL query is returned
  unchanged (never a hard failure, never a crash).
"""

from __future__ import annotations

from conftest import FakeClient

from app.rewrite import REWRITE_PROMPT, QueryRewriter


def test_rewrite_uses_provider_rewrite_when_enabled():
    rw = QueryRewriter(client=FakeClient("groundwater depletion water pumps"), enabled=True)
    assert rw.rewrite("may alam ba kayo tungkol sa water pumps na groundwater?") == (
        "groundwater depletion water pumps"
    )


def test_rewrite_falls_back_to_original_on_provider_failure():
    rw = QueryRewriter(client=FakeClient("ignored", fail=True), enabled=True)
    original = "anong paper about flood monitoring?"
    assert rw.rewrite(original) == original


def test_rewrite_disabled_returns_original_without_calling_provider():
    client = FakeClient("should not be used")
    rw = QueryRewriter(client=client, enabled=False)
    original = "papers by Lisandro Grimes"
    assert rw.rewrite(original) == original
    assert client.chat.completions.calls == []


def test_rewrite_without_key_or_client_returns_original_without_provider(monkeypatch):
    """No API key configured -> fast fallback, zero provider attempts."""
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_api_keys", "")
    rw = QueryRewriter(enabled=True)
    assert rw.rewrite("any query") == "any query"
    assert rw._client is None


def test_rewrite_with_llm_api_keys_only_calls_provider(monkeypatch):
    """LLM_API_KEYS alone (LLM_API_KEY empty) must still enable rewriting."""
    import app.llm as llm_mod
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_api_keys", "k1,k2")
    client = FakeClient("rewritten query")
    monkeypatch.setattr(llm_mod, "_make_client", lambda base_url, key: client)

    rw = QueryRewriter(enabled=True)
    assert rw.rewrite("a verbose user query") == "rewritten query"
    assert len(client.chat.completions.calls) == 1
    assert rw._client._api_keys == ["k1", "k2"]


def test_rewrite_with_explicit_api_key_ignores_settings_list(monkeypatch):
    """An explicit api_key stays a single-key client even when the settings
    list is configured — the guard must not disable it either."""
    import app.llm as llm_mod
    from app.config import settings

    monkeypatch.setattr(settings, "llm_api_key", "")
    monkeypatch.setattr(settings, "llm_api_keys", "k1,k2")
    client = FakeClient("rewritten query")
    monkeypatch.setattr(llm_mod, "_make_client", lambda base_url, key: client)

    rw = QueryRewriter(enabled=True, api_key="explicit")
    assert rw.rewrite("a verbose user query") == "rewritten query"
    assert rw._client._api_keys == ["explicit"]


def test_rewrite_prompt_mentions_keyword_optimization_and_exact_codes():
    assert "keyword-style search query" in REWRITE_PROMPT
    assert "catalog codes" in REWRITE_PROMPT


def test_rewrite_blank_query_returns_blank():
    rw = QueryRewriter(client=FakeClient("x"), enabled=True)
    assert rw.rewrite("   ") == "   "
