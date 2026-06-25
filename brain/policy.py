"""Pure decision helpers used by agent_server.

Kept dependency-free (no heavy imports) so they're unit-testable offline,
unlike agent_server which pulls in Jetson-only deps (faster-whisper, piper, …).
"""
from __future__ import annotations


def skin_for_rocky_mode(rocky_mode) -> str:
    """The firmware avatar skin that should be active for a ROCKY_MODE value.

    ROCKY_MODE is the single source of truth for "be Rocky" (voice + face);
    the brain emits ``set_skin`` so the firmware follows it.
    """
    return "rocky" if rocky_mode else "default"


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
