"""Shared lazy OpenAI-compatible client for the OpenRouter provider.

The RagService/AgenticLoop client-injection shape (``client=None`` +
``base_url``/``api_key``, materialized lazily) is used by every LLM-touching
component; this helper owns the lazy-materialization step so it is not
re-implemented per module (ADR 0001: OpenRouter via the openai SDK).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

log = logging.getLogger(__name__)

# Index of the last key that succeeded, per (base_url, key set). When a key
# exhausts its limit, later calls start from the working key instead of
# retrying dead ones. On total failure the index is left alone so the next
# call starts over (limits reset).
_good_index: dict[tuple[str, tuple[str, ...]], int] = {}
_good_index_lock = threading.Lock()


def _make_client(base_url: str, api_key: str) -> OpenAI:
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _is_limit_error(exc: Exception) -> bool:
    import openai

    if isinstance(exc, openai.RateLimitError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code in (401, 402)


class FailoverOpenAI:
    """OpenAI-compatible client that rotates through API keys on limit errors.

    Only exposes ``.chat.completions.create()`` — the only OpenAI surface the
    app uses. Rate-limit (429), payment (402) and auth (401) errors rotate to
    the next key and retry the same call; every key is tried at most once per
    call, then the last error is re-raised. All other errors propagate
    unchanged. Key values are never logged or attached to exceptions.
    """

    def __init__(self, base_url: str, api_keys: list[str]) -> None:
        self._base_url = base_url
        self._api_keys = list(api_keys) or [""]
        self._clients: dict[int, OpenAI] = {}
        self.chat = _FailoverChat(self)

    def _client(self, index: int) -> OpenAI:
        client = self._clients.get(index)
        if client is None:
            client = _make_client(self._base_url, self._api_keys[index])
            self._clients[index] = client
        return client

    def _create(self, **kwargs):
        state_key = (self._base_url, tuple(self._api_keys))
        with _good_index_lock:
            start = _good_index.get(state_key, 0)
        count = len(self._api_keys)
        last_error: Exception | None = None
        for offset in range(count):
            index = (start + offset) % count
            try:
                result = self._client(index).chat.completions.create(**kwargs)
            except Exception as exc:
                if not _is_limit_error(exc):
                    raise
                last_error = exc
                log.warning(
                    "LLM key %d/%d hit its limit; rotating to the next key", index + 1, count
                )
                continue
            if index != start:
                with _good_index_lock:
                    _good_index[state_key] = index
            return result
        assert last_error is not None
        raise last_error


class _FailoverCompletions:
    def __init__(self, owner: FailoverOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class _FailoverChat:
    def __init__(self, owner: FailoverOpenAI) -> None:
        self.completions = _FailoverCompletions(owner)


def failover_keys(api_key: str) -> list[str]:
    """Keys a failover client rotates through for `api_key`: the merged
    settings list when the caller passes the settings default, else just it."""
    from .config import settings

    return settings.llm_key_list if api_key == settings.llm_api_key else [api_key]


def ensure_openai_client(client: OpenAI | None, base_url: str, api_key: str) -> OpenAI:
    """Return the injected client, or lazily materialize one for the provider."""
    if client is not None:
        return client
    return FailoverOpenAI(base_url, failover_keys(api_key))
