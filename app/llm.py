"""Shared lazy OpenAI-compatible client for the OpenRouter provider.

The RagService/AgenticLoop client-injection shape (``client=None`` +
``base_url``/``api_key``, materialized lazily) is used by every LLM-touching
component; this helper owns the lazy-materialization step so it is not
re-implemented per module (ADR 0001: OpenRouter via the openai SDK).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


def ensure_openai_client(client: OpenAI | None, base_url: str, api_key: str) -> OpenAI:
    """Return the injected client, or lazily materialize one for the provider."""
    if client is not None:
        return client
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)
