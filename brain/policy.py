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


def buddy_sync_command(knob: object, last_sent: bool | None) -> dict | None:
    """The set_buddy command to send for the BUDDY_ENABLED knob, or None
    when the firmware was already told this value on the current connection.

    ``last_sent`` is None before the connect-time push, so the first call
    always sends. Only a change is worth a frame: the firmware reboots on a
    differing value, and a repeat is a no-op there anyway.
    """
    enabled = bool(knob)
    if last_sent is not None and enabled == last_sent:
        return None
    return {"cmd": "set_buddy", "enabled": enabled}
