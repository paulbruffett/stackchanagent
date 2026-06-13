"""Piper TTS wrapper.

Returns 16 kHz s16le mono PCM bytes, resampled from Piper's native rate
(typically 22050 Hz) so the audio matches the firmware's codec rate.

Voices are downloaded on first use into ~/.cache/piper-voices.
"""

from __future__ import annotations

import logging
import os
import time
from math import gcd
from pathlib import Path

import numpy as np
from piper import PiperVoice
from piper.download_voices import download_voice
from scipy.signal import resample_poly

log = logging.getLogger("brain.tts")

# Piper voice. `libritts_r-medium` has more prosodic variation than the
# original `amy-medium` and is a strict drop-in. Override via env to A/B.
DEFAULT_VOICE = os.environ.get("PIPER_VOICE", "en_US-libritts_r-medium")
TARGET_SR = 16000


def resample_to_16k(pcm_int16: np.ndarray, src_sr: int) -> np.ndarray:
    """Polyphase resample an int16 PCM array to 16 kHz (the firmware codec
    rate). Returns the input unchanged when it's already 16 kHz. Shared by
    every TTS backend (Piper at 22050 Hz, Hume at 48000 Hz) so they all hit
    the same anti-aliased path — linear interpolation aliases audibly on the
    AW88298."""
    if src_sr == TARGET_SR:
        return pcm_int16
    g = gcd(TARGET_SR, src_sr)
    up, down = TARGET_SR // g, src_sr // g
    return (
        resample_poly(pcm_int16.astype(np.float32), up, down)
        .clip(-32768, 32767)
        .astype(np.int16)
    )


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

        # Polyphase resample with a windowed-sinc anti-alias filter (no-op
        # when already 16 kHz). For 22050 → 16000 the ratio reduces to
        # 320/441; linear interp (the old path) aliased audibly on the AW88298.
        pcm_int16 = resample_to_16k(pcm_int16, src_sr)

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
