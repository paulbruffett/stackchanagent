"""Home Assistant intent fast path.

Before an utterance goes to the LLM, hand the raw transcript to HA's built-in
intent matcher (`POST /api/conversation/process`, pinned to the local
`conversation.home_assistant` agent). It answers "turn off the office light"
or "is the floor lamp on" in tens of milliseconds, with no LLM round trip.

Only a definite hit is trusted: `action_done` (it did something) or
`query_answer` (it answered a state question). Anything else — no match, an
unknown device, an error — returns None and the turn falls through to the LLM
unchanged. HA's own error sentences ("not aware of any area called is") are
never spoken.

Needs HA_TOKEN (a long-lived access token) in the environment; without it the
fast path is off.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import httpx

from config import get_config

log = logging.getLogger("brain.ha")

# HA's local matcher answers in <100 ms when it matches (measured 3–130 ms on
# the Jetson). Anything slower means HA is struggling; give up and let the LLM
# take the turn rather than stall the user.
TIMEOUT_S = 2.0

TRUSTED = ("action_done", "query_answer")


@dataclass
class FastPathResult:
    speech: str
    response_type: str
    latency_ms: int


async def try_handle(text: str) -> FastPathResult | None:
    """Return HA's spoken answer if its local intent matcher definitely
    handled `text`, else None (caller falls through to the LLM)."""
    cfg = get_config()
    token = (os.environ.get("HA_TOKEN") or "").strip()
    if not cfg.get("HA_FAST_PATH") or not token or not text.strip():
        return None
    url = str(cfg.get("HA_URL")).rstrip("/") + "/api/conversation/process"
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as http:
            r = await http.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "text": text,
                    "language": "en",
                    "agent_id": "conversation.home_assistant",
                },
            )
            r.raise_for_status()
            resp = r.json().get("response", {})
    except Exception as e:
        log.warning("ha fast path unavailable (%s) — falling through", e)
        return None
    latency_ms = int((time.monotonic() - t0) * 1000)
    rtype = resp.get("response_type", "")
    speech = (resp.get("speech", {}).get("plain", {}).get("speech") or "").strip()
    if rtype not in TRUSTED or not speech:
        code = resp.get("data", {}).get("code", "")
        log.info("ha fast path: no match (%s %s, %d ms) — to LLM", rtype, code, latency_ms)
        return None
    log.info("ha fast path: %s in %d ms → %r", rtype, latency_ms, speech)
    return FastPathResult(speech=speech, response_type=rtype, latency_ms=latency_ms)
