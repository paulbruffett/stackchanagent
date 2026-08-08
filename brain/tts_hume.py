"""Hume AI cloud TTS backend for Rocky mode (Milestone 4).

Mirrors `tts.Synthesizer`: `synthesize(text) -> bytes` returns 16 kHz s16le
mono PCM, so it's a drop-in alternative at the `run_speaker` call site. Hume
streams 48 kHz PCM, which we resample with the shared `tts.resample_to_16k`
helper.

Configuration (all from `.env`, per the established secrets pattern):
  HUME_API_KEY        — required; without it the backend reports unavailable.
  HUME_VOICE_ID       — a voice id, typically a cloned Rocky voice.
  HUME_VOICE_PROVIDER — provider for HUME_VOICE_ID: CUSTOM_VOICE (a voice you
                        cloned/saved, the default) or HUME_AI (a library voice
                        referenced by id).
  HUME_FALLBACK_VOICE — a Hume library voice *name* used when no id is set
                        (provider HUME_AI).

Any failure here (missing key, network, quota, malformed stream) raises, and
the speaker path catches it and falls back to Piper for that sentence — so
the system runs as-is with no Hume account configured.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time

import httpx
import numpy as np

from tts import resample_to_16k

log = logging.getLogger("brain.tts.hume")

HUME_TTS_URL = "https://api.hume.ai/v0/tts/stream/json"
# Hume streams 16-bit mono PCM at 48 kHz when format.type == "pcm".
HUME_SR = 48000

# Wall-clock cap on one sentence's stream. httpx's read timeout is per socket
# read, so a dribbling stream refreshes it forever and the turn hangs with the
# device stuck in SPEAKING. Generous on purpose: a long sentence legitimately
# takes several seconds, and cutting one short only trades dead air for a Piper
# fallback mid-reply (an audible voice flip).
STREAM_BUDGET_S = 20.0

# How long a generation_id stays worth resending as prosody context. Hume's
# context is for continuity within a conversation; carrying one across an idle
# gap buys nothing and risks the server rejecting an id it has aged out — which
# would poison every later sentence, since the id is only replaced on success.
CONTEXT_TTL_S = 300.0

# Default acting instructions for the Rocky persona's voice. Steers prosody
# only — the grammar/word-choice lives in claude_agent.DEFAULT_ROCKY_PROMPT.
DEFAULT_DESCRIPTION = "Alien engineer. Broken English. Deliberate. Warm but strange."


class HumeSynthesizer:
    """Lazy, reusable Hume TTS client. One persistent httpx connection."""

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str | None = None,
        voice_provider: str | None = None,
        fallback_voice: str | None = None,
        description: str = DEFAULT_DESCRIPTION,
    ) -> None:
        self.api_key = api_key or os.environ.get("HUME_API_KEY") or ""
        self.voice_id = voice_id or os.environ.get("HUME_VOICE_ID") or ""
        self.voice_provider = (
            voice_provider or os.environ.get("HUME_VOICE_PROVIDER") or "CUSTOM_VOICE"
        )
        self.fallback_voice = (
            fallback_voice or os.environ.get("HUME_FALLBACK_VOICE") or ""
        )
        self.description = description
        self._client: httpx.Client | None = None
        # Carried across sentences within a conversation for prosody
        # continuity (Hume "context"). Best-effort; reset() clears it.
        self._last_generation_id: str | None = None
        self._last_generation_s: float = 0.0

    @property
    def available(self) -> bool:
        """True when Hume can be attempted: an API key plus some voice
        (an explicit id or a fallback library voice name)."""
        return bool(self.api_key and (self.voice_id or self.fallback_voice))

    def reset(self) -> None:
        """Forget the rolling generation_id so the next utterance starts a
        fresh prosody context (called at conversation boundaries: an idle gap
        longer than CONTEXT_TTL_S, and any failed request)."""
        self._last_generation_id = None
        self._last_generation_s = 0.0

    def _voice(self) -> dict[str, str]:
        if self.voice_id:
            return {"id": self.voice_id, "provider": self.voice_provider}
        return {"name": self.fallback_voice, "provider": "HUME_AI"}

    def synthesize(self, text: str, speed: float = 1.0) -> bytes:
        """Synthesize text → 16 kHz s16le mono PCM bytes. Raises on any
        error (missing key/voice, HTTP failure, empty stream)."""
        if not text.strip():
            return b""
        if not self.available:
            raise RuntimeError("hume unavailable: no API key or voice configured")

        if self._client is None:
            # Split, tighter budget than the old blanket 30 s: this call sits
            # in the speaking path, so every second spent waiting on a dead
            # Hume is a second the device holds the SPEAKING face in silence.
            self._client = httpx.Client(
                timeout=httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
            )

        t0 = time.monotonic()
        # An id from before the last idle gap is no longer prosody continuity,
        # just an id the server may have aged out.
        if self._last_generation_id and t0 - self._last_generation_s > CONTEXT_TTL_S:
            self.reset()

        utterance: dict[str, object] = {
            "text": text,
            "description": self.description,
            "voice": self._voice(),
            "speed": speed,
        }
        body: dict[str, object] = {
            "utterances": [utterance],
            "format": {"type": "pcm"},
            "instant_mode": True,
            "strip_headers": True,
        }
        if self._last_generation_id:
            body["context"] = {"generation_id": self._last_generation_id}

        pcm = bytearray()
        gen_id: str | None = None
        try:
            with self._client.stream(
                "POST",
                HUME_TTS_URL,
                headers={"X-Hume-Api-Key": self.api_key},
                json=body,
            ) as resp:
                if resp.status_code >= 400:
                    # A streamed response carries no body yet; read it so the
                    # fallback log says *why* Hume refused (quota, unknown
                    # voice, rejected context) and not just a status code.
                    resp.read()
                    raise RuntimeError(
                        f"hume http {resp.status_code}: {resp.text[:200]}"
                    )
                for line in resp.iter_lines():
                    # The read timeout is per socket read, so a stream that
                    # dribbles one chunk every few seconds refreshes it
                    # indefinitely. This is the only wall-clock bound.
                    if time.monotonic() - t0 > STREAM_BUDGET_S:
                        raise RuntimeError(
                            f"hume stream exceeded {STREAM_BUDGET_S:.0f}s budget"
                        )
                    if not line:
                        continue
                    obj = json.loads(line)
                    audio_b64 = obj.get("audio")
                    if audio_b64:
                        pcm += base64.standard_b64decode(audio_b64)
                    if obj.get("generation_id"):
                        gen_id = obj["generation_id"]

            if not pcm:
                raise RuntimeError("hume returned no audio")
        except Exception:
            # Drop the prosody context on any failure. It is only ever
            # *replaced* on success, so a stale or rejected id kept here would
            # be resent on the next sentence, and the next — one bad request
            # demoting the Rocky voice to Piper for the rest of the process.
            self.reset()
            raise

        if gen_id:
            self._last_generation_id = gen_id
            self._last_generation_s = time.monotonic()

        pcm_int16 = np.frombuffer(bytes(pcm), dtype=np.int16)
        pcm_int16 = resample_to_16k(pcm_int16, HUME_SR)
        ms = int((time.monotonic() - t0) * 1000)
        log.info(
            "hume tts: %d ms (%d Hz → 16000 Hz), %d samples",
            ms, HUME_SR, len(pcm_int16),
        )
        return pcm_int16.tobytes()
