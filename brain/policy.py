"""Pure decision helpers used by agent_server.

Kept dependency-free (no heavy imports) so they're unit-testable offline,
unlike agent_server which pulls in Jetson-only deps (faster-whisper, piper, …).
"""
from __future__ import annotations

from dataclasses import dataclass


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


# --- device status (battery / charging / volume) ----------------------------

LOW_BATTERY_PCT = 15
VOLUME_STEP = 15


@dataclass
class DeviceStatus:
    """What the firmware last reported about itself (boot and status events),
    shared by the connection, the tools and the console. None = not reported
    (older firmware, or no battery reading)."""
    battery: int | None = None
    charging: bool | None = None
    volume: int | None = None
    # Last volume sent as set_volume on this connection (the sync or the tool).
    volume_sent: int | None = None
    # A low-battery warning was logged this discharge cycle.
    low_warned: bool = False
    # time.time() of the last report; None until the first one.
    updated_at: float | None = None

    def update(self, payload: dict, now: float) -> None:
        """Fold a boot/status event's fields in. Absent fields leave the
        cached value alone; a field sent as null (or garbage) clears it."""
        if "battery" in payload:
            b = payload["battery"]
            ok = isinstance(b, int) and not isinstance(b, bool) and 0 <= b <= 100
            self.battery = b if ok else None
        if "charging" in payload:
            c = payload["charging"]
            self.charging = c if isinstance(c, bool) else None
        if "volume" in payload:
            v = payload["volume"]
            ok = isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100
            self.volume = v if ok else None
        self.updated_at = now

    @property
    def low_battery(self) -> bool:
        return (self.battery is not None and self.battery <= LOW_BATTERY_PCT
                and self.charging is not True)


def volume_sync_action(
    knob: object, reported: int | None, last_sent: int | None, busy: bool,
    knob_set: bool = True,
) -> int | None:
    """The volume to send as set_volume for the SPEAKER_VOLUME knob, or None.

    Like buddy_sync_action but nothing reboots: only between conversations,
    only once the firmware has reported its volume (older firmware without
    set_volume never does, so it is never sent a command it would reject), and
    not again for a value already sent on this connection — the firmware
    answers set_volume with a status event, which updates ``reported``.
    """
    # A knob nobody has set is just the default: the volume the robot
    # already holds (set by voice, or from before this knob existed) wins,
    # rather than every fresh brain resetting it to the default.
    if busy or reported is None or not knob_set:
        return None
    target = int(knob)  # type: ignore[call-overload]
    if target == reported or target == last_sent:
        return None
    return target


def step_volume(current: int, level: int | None, change: str | None) -> int:
    """New volume for the set_volume tool: an absolute ``level``, else
    ``change`` "up"/"down" by VOLUME_STEP from ``current``; clamped to
    0..100. Raises ValueError when neither is given."""
    if level is not None:
        target = int(level)
    elif change == "up":
        target = current + VOLUME_STEP
    elif change == "down":
        target = current - VOLUME_STEP
    else:
        raise ValueError("give a level (0-100) or change 'up'/'down'")
    return max(0, min(100, target))


def describe_device_status(status: DeviceStatus) -> str:
    """One or two short sentences the model can relay aloud."""
    if status.updated_at is None:
        return "No status from the robot's body yet."
    if status.battery is None:
        battery = "Battery level unknown"
    else:
        battery = f"Battery at {status.battery} percent"
    if status.charging is True:
        battery += ", charging."
    elif status.charging is False:
        battery += ", not charging."
    else:
        battery += "."
    volume = ("Volume unknown." if status.volume is None
              else f"Volume at {status.volume} out of 100.")
    return f"{battery} {volume}"


def low_battery_check(
    battery: int | None, charging: bool | None, warned: bool,
) -> tuple[bool, bool]:
    """(warn now, warned afterwards) for the once-per-discharge-cycle
    low-battery warning. Charging re-arms it; an unknown reading changes
    nothing."""
    if charging is True:
        return False, False
    if battery is None or battery > LOW_BATTERY_PCT or warned:
        return False, warned
    return True, True
