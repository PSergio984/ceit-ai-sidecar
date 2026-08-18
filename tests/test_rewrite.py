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


def test_rewrite_without_key_or_client_returns_original_without_provider():
    """No API key configured -> fast fallback, zero provider attempts."""
    import app.rewrite as rewrite_mod

    original_client = rewrite_mod.settings
    try:
        rewrite_mod.settings = type(
            "S", (), {"llm_api_key": "", "llm_base_url": "", "llm_model": ""}
        )()
        rw = QueryRewriter(enabled=True)
        assert rw.rewrite("any query") == "any query"
    finally:
        rewrite_mod.settings = original_client


def test_rewrite_prompt_mentions_keyword_optimization_and_exact_codes():
    assert "keyword-style search query" in REWRITE_PROMPT
    assert "catalog codes" in REWRITE_PROMPT


def test_rewrite_blank_query_returns_blank():
    rw = QueryRewriter(client=FakeClient("x"), enabled=True)
    assert rw.rewrite("   ") == "   "
