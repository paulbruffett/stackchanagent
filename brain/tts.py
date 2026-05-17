"""Piper TTS wrapper.

Returns 16 kHz s16le mono PCM bytes, resampled from Piper's native rate
(typically 22050 Hz) so the audio matches the firmware's codec rate.

Voices are downloaded on first use into ~/.cache/piper-voices.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.download_voices import download_voice

log = logging.getLogger("brain.tts")

DEFAULT_VOICE = "en_US-amy-medium"
TARGET_SR = 16000


class Synthesizer:
    """Lazy-loaded Piper voice. Resamples to 16 kHz on every call."""

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        cache_dir: Path | None = None,
    ) -> None:
        self.voice_name = voice
        self.cache_dir = cache_dir or Path.home() / ".cache" / "piper-voices"
        self._voice: PiperVoice | None = None

    def _load(self) -> PiperVoice:
        if self._voice is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            onnx = self.cache_dir / f"{self.voice_name}.onnx"
            if not onnx.exists():
                log.info("downloading piper voice %s …", self.voice_name)
                download_voice(self.voice_name, self.cache_dir)
            log.info("loading piper voice %s", self.voice_name)
            t0 = time.monotonic()
            self._voice = PiperVoice.load(onnx)
            log.info("piper loaded in %.1fs", time.monotonic() - t0)
        return self._voice

    def synthesize(self, text: str) -> bytes:
        """Synthesize text → 16 kHz s16le mono PCM bytes."""
        if not text.strip():
            return b""
        voice = self._load()
        t0 = time.monotonic()
        chunks = list(voice.synthesize(text))
        if not chunks:
            return b""

        src_sr = chunks[0].sample_rate
        # Concatenate int16 PCM from all chunks.
        pcm_int16 = np.concatenate([c.audio_int16_array for c in chunks])

        if src_sr != TARGET_SR:
            # Linear resample via numpy. Adequate for speech voices.
            new_len = int(round(len(pcm_int16) * TARGET_SR / src_sr))
            x = np.linspace(0, 1, len(pcm_int16), endpoint=False)
            xi = np.linspace(0, 1, new_len, endpoint=False)
            pcm_int16 = np.interp(xi, x, pcm_int16.astype(np.float32)).astype(
                np.int16
            )

        ms = int((time.monotonic() - t0) * 1000)
        log.info(
            "tts: %d ms (%d Hz → %d Hz), %d samples → %d samples",
            ms,
            src_sr,
            TARGET_SR,
            sum(len(c.audio_int16_array) for c in chunks),
            len(pcm_int16),
        )
        return pcm_int16.tobytes()
