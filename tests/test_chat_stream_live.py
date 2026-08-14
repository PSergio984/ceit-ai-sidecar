"""Live LLM chat smoke test — skipped unless SIDECAR_LIVE_CHAT_TEST=1.

Exercises the real provider through POST /chat/stream against a running
sidecar (uv run uvicorn app.main:app). Mirrors the Phase 8 SidecarLiveTest
env-gate discipline; never runs in CI. The provider key stays in the sidecar
.env (gitignored); corpus must be exported first (php artisan ai:export-corpus)
since tests delete corpus files.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SIDECAR_LIVE_CHAT_TEST") != "1",
    reason="Set SIDECAR_LIVE_CHAT_TEST=1 to run the live LLM chat smoke.",
)


def test_live_chat_stream_round_trip():
    resp = httpx.post(
        "http://127.0.0.1:8310/chat/stream",
        headers={"X-Sidecar-Token": os.environ["SIDECAR_TOKEN"]},
        json={"query": "school ID", "mode": "citations", "corpus": "policy", "top_k": 3},
        timeout=60,
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "data: " in resp.text
    assert resp.text.endswith("data: [DONE]\n\n")
