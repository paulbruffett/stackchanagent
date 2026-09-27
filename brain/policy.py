"""Pure decision helpers used by agent_server.

Kept dependency-free (no heavy imports) so they're unit-testable offline,
unlike agent_server which pulls in Jetson-only deps (faster-whisper, piper, …).
"""
from __future__ import annotations


def effective_sleep_timeout(
    base_s: float, prompt_timeout_s: float, buddy_prompt_pending: bool
) -> float:
    """Idle-sleep timeout in seconds.

    While a BLE buddy approve prompt is pending, hold off sleeping for the
    (longer) prompt timeout so the device doesn't sleep out from under an
    unanswered prompt. Never shortens the base timeout — a pending prompt can
    only ever delay sleep, not hasten it.
    """
    if buddy_prompt_pending:
        return max(base_s, prompt_timeout_s)
    return base_s



def capture_is_stale(
    listening: bool,
    started_at_s: float,
    now_s: float,
    max_utterance_ms: float,
    follow_up: bool,
    follow_up_window_s: float,
    grace_s: float = 3.0,
) -> bool:
    """True when an open capture has outlived anything legitimate.

    Utterance end is otherwise only ever judged on an incoming audio frame, so
    if the firmware stops streaming (it hit its listening watchdog, or the
    listen_timeout event was lost) the brain would stay "listening" forever:
    no sleep, no summarising, no set_buddy. A follow-up window may legitimately
    stay open longer than MAX_UTTERANCE_MS when that is set low, so it gets
    whichever limit is longer.
    """
    if not listening:
        return False
    limit_s = max_utterance_ms / 1000.0
    if follow_up:
        limit_s = max(limit_s, follow_up_window_s)
    return now_s - started_at_s > limit_s + grace_s


def buddy_sync_action(
    knob: object,
    reported: bool | None,
    last_sent: bool | None,
    busy: bool,
) -> str | None:
    """What to do about the BUDDY_ENABLED knob on this connection.

    Returns None (nothing), "send" (send set_buddy, keep the link) or
    "send_close" (send set_buddy, then close the link: the firmware is about to
    reboot into the other mode).

    ``reported`` is the mode the firmware said it booted in (boot event's
    "buddy" field), or None for firmware that doesn't report it — then fall
    back to sending whenever the knob differs from what this connection last
    sent, without closing (we can't know whether it will reboot). ``busy`` (a
    conversation in progress) defers everything: the change reboots the robot.
    """
    if busy:
        return None
    enabled = bool(knob)
    if reported is None:
        return "send" if last_sent != enabled else None
    if enabled == reported or last_sent == enabled:
        return None
    return "send_close"
