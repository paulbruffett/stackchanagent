"""Home Assistant intent fast path.

Before an utterance goes to the LLM, hand the raw transcript to HA's built-in
intent matcher (`POST /api/conversation/process`, pinned to the local
`conversation.home_assistant` agent). It answers "turn off the office light"
or "is the floor lamp on" in tens of milliseconds, with no LLM round trip.

Only a definite hit is trusted: `action_done` (it did something) or
`query_answer` (it answered a state question). Anything else — no match, an
unknown device, an error, a malformed reply — is a miss and the turn falls
through to the LLM unchanged. HA's own error sentences ("not aware of any area
called is") are never spoken.

Needs HA_TOKEN (a long-lived access token) in the environment; without it the
fast path is off. The HA base URL is HA_URL in the environment too, not a
console knob: the token rides along to whatever URL that names.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import httpx

from config import get_config
from mcp_client import _http_headers

log = logging.getLogger("brain.ha")

# HA's local matcher answers in <100 ms when it matches (measured 3–130 ms on
# the Jetson). Anything slower means HA is struggling; give up and let the LLM
# take the turn rather than stall the user. A total deadline — httpx's own
# timeout applies per phase (connect, read, …), so it alone can run longer.
TIMEOUT_S = 2.0

TRUSTED = ("action_done", "query_answer")


@dataclass
class FastPathResult:
    # None on a miss: the caller falls through to the LLM.
    speech: str | None
    response_type: str
    latency_ms: int


_client: httpx.AsyncClient | None = None


def _make_client() -> httpx.AsyncClient:
    """Separate from _get_client so tests can swap in a mock transport."""
    return httpx.AsyncClient(timeout=TIMEOUT_S)


def _get_client() -> httpx.AsyncClient:
    # Long-lived: building a client loads the CA bundle and opens a fresh
    # connection, on the event loop, on every utterance otherwise.
    global _client
    if _client is None:
        _client = _make_client()
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _base_url() -> str:
    return (os.environ.get("HA_URL") or "http://localhost:8123").strip().rstrip("/")


async def _process(text: str) -> dict:
    r = await _get_client().post(
        _base_url() + "/api/conversation/process",
        headers=_http_headers("HA_TOKEN"),
        json={"text": text, "language": "en", "agent_id": "conversation.home_assistant"},
    )
    r.raise_for_status()
    body = r.json()
    resp = body.get("response") if isinstance(body, dict) else None
    return resp if isinstance(resp, dict) else {}


def _speech(resp: dict) -> str:
    speech = resp.get("speech")
    plain = speech.get("plain") if isinstance(speech, dict) else None
    text = plain.get("speech") if isinstance(plain, dict) else None
    return text.strip() if isinstance(text, str) else ""


async def try_handle(text: str) -> FastPathResult | None:
    """Ask HA's local intent matcher about `text`. Returns None if the fast
    path is off (no attempt made); otherwise a result whose `speech` is set
    only on a definite hit — a miss still carries the latency it cost."""
    if not get_config().get("HA_FAST_PATH") or not text.strip():
        return None
    if not (os.environ.get("HA_TOKEN") or "").strip():
        return None
    t0 = time.monotonic()
    try:
        resp = await asyncio.wait_for(_process(text), TIMEOUT_S)
        rtype = str(resp.get("response_type") or "")
        speech = _speech(resp)
        data = resp.get("data")
        code = data.get("code", "") if isinstance(data, dict) else ""
    except Exception as e:
        latency_ms = int((time.monotonic() - t0) * 1000)
        log.warning("ha fast path unavailable (%r, %d ms) — to LLM", e, latency_ms)
        return FastPathResult(None, "unavailable", latency_ms)
    latency_ms = int((time.monotonic() - t0) * 1000)
    if rtype not in TRUSTED or not speech:
        log.info("ha fast path: no match (%s %s, %d ms) — to LLM", rtype, code, latency_ms)
        return FastPathResult(None, rtype, latency_ms)
    log.info("ha fast path: %s in %d ms → %r", rtype, latency_ms, speech)
    return FastPathResult(speech, rtype, latency_ms)
