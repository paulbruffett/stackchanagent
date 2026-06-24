"""Speech-to-text wrapper around faster-whisper.

Brain accumulates 16 kHz s16le PCM frames during the LISTENING state,
then calls `Transcriber.transcribe(pcm_bytes)` once the utterance
ends. Returns the text.

Default model is `small.en` on CUDA float16 — the brain runs on the
Orin Nano. Override via env for CPU dev:
    STT_DEVICE=cpu STT_COMPUTE_TYPE=int8
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("brain.stt")

DEFAULT_DEVICE = os.environ.get("STT_DEVICE", "cuda")
DEFAULT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "float16")


@dataclass
class Transcript:
    text: str
    latency_ms: int
    # Worst-case per-segment confidence from faster-whisper, plus the
    # captured signal level. Surfaced so the caller can gate noise/
    # hallucinations on follow-up turns (see agent_server.respond). Neutral
    # defaults (high confidence, no signal) for the empty-audio path.
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    peak_pct: float = 0.0
    rms_pct: float = 0.0


def should_drop_follow_up(
    t: Transcript,
    voiced_ms: float,
    *,
    max_no_speech_prob: float,
    min_avg_logprob: float,
    clip_peak_pct: float,
    min_voiced_ms: float,
) -> tuple[bool, str]:
    """Follow-up false-trigger gate (M6.7), as a pure predicate so it can be
    unit-tested in isolation. A follow-up turn takes no wakeword, so a noise
    blip hallucinated into text would otherwise start a turn (and open yet
    another window — a self-perpetuating loop). Returns (drop, reason): drop
    when Whisper is unconfident (weak no_speech_prob/avg_logprob) OR the
    capture is a clipping blip too short to be real speech (the AND keeps a
    genuinely loud-but-real or clean-but-short word). reason is "" when kept.
    Callers gate follow-up turns only; wakeword turns are never dropped."""
    if t.no_speech_prob >= max_no_speech_prob or t.avg_logprob <= min_avg_logprob:
        return True, "low confidence"
    if t.peak_pct >= clip_peak_pct and voiced_ms < min_voiced_ms:
        return True, "noise blip"
    return False, ""


class Transcriber:
    """Lazy-loads the model on first transcribe; subsequent calls reuse it."""

    def __init__(
        self,
        model_name: str = "small.en",
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None

    def _load(self) -> WhisperModel:
        if self._model is None:
            log.info(
                "loading whisper model %s (device=%s, compute_type=%s)",
                self.model_name,
                self.device,
                self.compute_type,
            )
            t0 = time.monotonic()
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
            log.info("whisper loaded in %.1fs", time.monotonic() - t0)
        return self._model

    def transcribe(self, pcm: bytes) -> Transcript:
        """Transcribe a buffer of 16 kHz s16le mono PCM samples."""
        if not pcm:
            return Transcript(text="", latency_ms=0)

        model = self._load()
        audio = (
            np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        )
        # Signal level, as % of full scale. A healthy speech utterance peaks
        # ~30-90% with rms ~5-20%; peak below ~5% means the mic gain is too
        # low (Whisper hallucinates plausible-but-wrong text on weak audio),
        # near 100% means clipping. Logged every turn to diagnose STT errors.
        peak_pct = float(np.max(np.abs(audio))) * 100.0 if audio.size else 0.0
        rms_pct = float(np.sqrt(np.mean(audio**2))) * 100.0 if audio.size else 0.0
        t0 = time.monotonic()
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=1,
            # Short utterances; word timestamps and VAD aren't needed.
            vad_filter=False,
            condition_on_previous_text=False,
        )
        # Materialize the generator once so we can both join the text and read
        # per-segment confidence. Aggregate worst-case (a single bad segment is
        # what we want to catch): highest no_speech_prob, lowest avg_logprob.
        segs = list(segments)
        text = " ".join(seg.text.strip() for seg in segs).strip()
        no_speech_prob = max((s.no_speech_prob for s in segs), default=0.0)
        avg_logprob = min((s.avg_logprob for s in segs), default=0.0)
        ms = int((time.monotonic() - t0) * 1000)
        log.info(
            "stt: %d ms, %d bytes (peak=%.1f%% rms=%.1f%%) → %r",
            ms, len(pcm), peak_pct, rms_pct, text,
        )
        return Transcript(
            text=text,
            latency_ms=ms,
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
            peak_pct=peak_pct,
            rms_pct=rms_pct,
        )
