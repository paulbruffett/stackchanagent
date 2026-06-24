"""M6.7 follow-up false-trigger gate (should_drop_follow_up). The gate runs on
follow-up turns only — these cover the drop/keep decision in isolation."""

from __future__ import annotations

from stt import Transcript, should_drop_follow_up

# The shipped defaults (config.py FOLLOWUP_* knobs).
K = dict(
    max_no_speech_prob=0.6,
    min_avg_logprob=-0.8,
    clip_peak_pct=99.0,
    min_voiced_ms=400,
)


def _t(**kw):
    base = dict(text="x", latency_ms=1, no_speech_prob=0.1,
                avg_logprob=-0.3, peak_pct=60.0, rms_pct=10.0)
    base.update(kw)
    return Transcript(**base)


def test_clean_followup_passes():
    assert should_drop_follow_up(_t(text="turn the light on"), 800, **K) == (False, "")


def test_high_no_speech_prob_dropped():
    drop, reason = should_drop_follow_up(_t(no_speech_prob=0.9), 800, **K)
    assert drop and reason == "low confidence"


def test_low_avg_logprob_dropped():
    drop, reason = should_drop_follow_up(_t(avg_logprob=-1.2), 800, **K)
    assert drop and reason == "low confidence"


def test_clipping_blip_dropped():
    # The exact signature from the bug: clipped peak + too-short voiced span.
    drop, reason = should_drop_follow_up(_t(peak_pct=100.0), 300, **K)
    assert drop and reason == "noise blip"


def test_loud_but_real_word_passes():
    # Clipped peak but enough voiced speech — the AND keeps it.
    assert should_drop_follow_up(_t(peak_pct=100.0), 800, **K) == (False, "")


def test_clean_but_short_word_passes():
    # Short voiced span but not clipping — not the blip signature.
    assert should_drop_follow_up(_t(peak_pct=60.0), 200, **K) == (False, "")


def test_thresholds_are_boundary_inclusive():
    # >= / <= boundaries: exactly at the threshold drops.
    assert should_drop_follow_up(_t(no_speech_prob=0.6), 800, **K)[0] is True
    assert should_drop_follow_up(_t(avg_logprob=-0.8), 800, **K)[0] is True
