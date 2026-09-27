"""Voice timers and reminders: the pure half.

Timers live in the brain, in memory.db's `timers` table (memory.py), keyed by
id with a wall-clock fire time — so a brain restart neither loses nor drifts
them. This module holds everything about them that doesn't touch the socket or
the DB: turning the set_timer arguments into a fire time, picking which timers
are due, matching a cancel request, and the sentences the robot says. The
scheduler that fires them is agent_server._timer_loop; the tools are in
tools.py.

"Local time" is the Jetson's own zone (America/Los_Angeles, via the system
clock): `datetime.now().astimezone()`. Fire times are stored as Unix
timestamps, so a DST change between setting and firing can't move a timer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

MIN_DURATION_S = 1
MAX_DURATION_S = 24 * 3600
MAX_ACTIVE = 20
# A timer that comes due while the robot is disconnected still fires on
# reconnect if it is at most this late; older ones are dropped (logged).
LATE_GRACE_S = 600.0

KINDS = ("timer", "reminder")


class TimerError(ValueError):
    """A set_timer / cancel_timer request that can't be carried out; the
    message is written for the model to relay."""


@dataclass(frozen=True)
class Timer:
    id: int
    label: str | None
    fire_ts: float
    created_ts: float
    # The requested length for a duration timer ("your 10 minute timer");
    # None for one set for a clock time.
    duration_s: int | None
    kind: str = "timer"   # "timer" | "reminder"


@dataclass(frozen=True)
class NewTimer:
    """A validated set_timer request, ready to persist."""
    label: str | None
    fire_ts: float
    duration_s: int | None
    kind: str


_AT = re.compile(
    r"^\s*(\d{1,2})(?:[:.](\d{2}))?\s*([ap])?\.?\s*(?:m\.?)?\s*$", re.IGNORECASE
)


def parse_at(text: str, now: datetime) -> datetime:
    """The next occurrence of clock time `text` after `now` (an aware local
    datetime). Takes "17:00", "5:30", "5pm", "5:30 PM", "12 am". A time that
    has already passed today — or is exactly now — means tomorrow."""
    m = _AT.match(text or "")
    if not m:
        raise TimerError(f"Couldn't read the time {text!r}; use HH:MM, e.g. 17:00.")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    ampm = (m.group(3) or "").lower()
    if ampm:
        if not 1 <= hour <= 12:
            raise TimerError(f"{text!r} isn't a valid time.")
        hour = hour % 12 + (12 if ampm == "p" else 0)
    if hour > 23 or minute > 59:
        raise TimerError(f"{text!r} isn't a valid time.")
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
        # astimezone() hands back a fixed UTC offset, so re-derive it for
        # tomorrow's date: across a DST change the stated clock time still
        # holds. (Only when `now` is in the system zone — tests may not be.)
        if now.utcoffset() == now.astimezone().utcoffset():
            target = target.replace(tzinfo=None).astimezone()
    return target


def new_timer(
    duration_seconds: object,
    at: object,
    label: object,
    now: datetime,
    active: int,
    kind: object = None,
) -> NewTimer:
    """Validate set_timer's arguments. Exactly one of duration_seconds / at;
    1 s ≤ duration ≤ 24 h; at most MAX_ACTIVE timers."""
    # Models fill optional properties with blanks (see tools.clean_mcp_args).
    has_duration = duration_seconds not in (None, "", 0)
    has_at = at not in (None, "")
    if has_duration == has_at:
        raise TimerError("Give exactly one of duration_seconds or at.")
    if active >= MAX_ACTIVE:
        raise TimerError(f"There are already {active} timers; cancel one first.")
    text = label.strip() if isinstance(label, str) else ""
    clean_label = text or None
    if has_duration:
        try:
            seconds = int(round(float(duration_seconds)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise TimerError(f"duration_seconds must be a number, not {duration_seconds!r}.")
        if not MIN_DURATION_S <= seconds <= MAX_DURATION_S:
            raise TimerError("A timer must be between 1 second and 24 hours.")
        fire_ts = now.timestamp() + seconds
        duration: int | None = seconds
        default_kind = "timer"
    else:
        fire_ts = parse_at(str(at), now).timestamp()
        duration = None
        default_kind = "reminder"
    k = kind if kind in KINDS else default_kind
    return NewTimer(clean_label, fire_ts, duration, k)


def partition_due(
    timers: list[Timer], now_ts: float, grace_s: float = LATE_GRACE_S
) -> tuple[list[Timer], list[Timer]]:
    """(due, missed): timers whose fire time has come and are at most
    `grace_s` late — oldest first — and those later than that, which are
    dropped instead of fired (the robot was offline when they came due)."""
    due: list[Timer] = []
    missed: list[Timer] = []
    for t in sorted(timers, key=lambda t: t.fire_ts):
        if t.fire_ts > now_ts:
            continue
        (missed if now_ts - t.fire_ts > grace_s else due).append(t)
    return due, missed


_GENERIC = {"", "timer", "the timer", "my timer", "reminder", "the reminder",
            "my reminder", "it", "that"}


def match_cancel(timers: list[Timer], target: object) -> list[Timer]:
    """The timers cancel_timer(target) means: "all"; an id; a label (case-
    insensitive, either containing the other — "pasta" matches "pasta
    timer"); or, with exactly one timer set, a generic "the timer". Raises
    TimerError, listing what is set, when nothing matches."""
    text = str(target if target is not None else "").strip().lower()
    if text in ("all", "all timers", "everything", "all reminders"):
        return list(timers)
    if text.isdigit():
        hit = [t for t in timers if t.id == int(text)]
        if hit:
            return hit
    if text:
        hit = [t for t in timers
               if t.label and (text in t.label.lower() or t.label.lower() in text)]
        if hit:
            return hit
    if text in _GENERIC and len(timers) == 1:
        return list(timers)
    if not timers:
        raise TimerError("No timers are set.")
    raise TimerError(f"No timer matches {target!r}. " + format_list(timers, _now(), with_ids=True))


# --- sentences ----------------------------------------------------------------

def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}" if n == 1 else f"{n} {unit}s"


def span(seconds: float, adjective: bool = False) -> str:
    """"1 hour 30 minutes" / "45 seconds"; adjective=True gives the form
    before a noun ("10 minute", "1 hour 30 minute")."""
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    parts = [(h, "hour"), (m, "minute"), (sec, "second")]
    words = [f"{n} {u}" if adjective else _plural(n, u) for n, u in parts if n]
    return " ".join(words) or ("0 second" if adjective else "0 seconds")


def remaining(seconds: float) -> str:
    """Time left, rounded for speech: seconds under a minute, else whole
    minutes (hours and minutes past an hour)."""
    s = max(0, int(round(seconds)))
    if s < 60:
        return _plural(s, "second")
    minutes = int(round(s / 60))
    h, m = divmod(minutes, 60)
    if not h:
        return _plural(m, "minute")
    return _plural(h, "hour") + (f" {_plural(m, 'minute')}" if m else "")


def clock(ts: float, now: datetime) -> str:
    """"5 PM", "5:30 PM", "9 AM tomorrow" in local time."""
    t = datetime.fromtimestamp(ts).astimezone()
    hour = t.hour % 12 or 12
    text = f"{hour} {'AM' if t.hour < 12 else 'PM'}" if t.minute == 0 else \
        f"{hour}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'}"
    days = (t.date() - now.date()).days
    if days == 1:
        text += " tomorrow"
    elif days > 1:
        text += f" on {t:%A}"
    return text


def _name(t: Timer) -> str:
    """How the list refers to one timer."""
    if t.label:
        return t.label
    if t.duration_s is not None:
        return f"{span(t.duration_s, adjective=True)} timer"
    return "reminder"


def describe(t: Timer, now: datetime) -> str:
    """"pasta in 4 minutes" / "call mom at 5 PM"."""
    if t.duration_s is None:
        return f"{_name(t)} at {clock(t.fire_ts, now)}"
    return f"{_name(t)} in {remaining(t.fire_ts - now.timestamp())}"


def format_list(timers: list[Timer], now: datetime, with_ids: bool = False) -> str:
    """"2 timers: pasta in 4 minutes, call mom at 5 PM." The tool result
    carries ids in [brackets] (not spoken) so the model can cancel one."""
    if not timers:
        return "No timers are set."
    ordered = sorted(timers, key=lambda t: t.fire_ts)
    items = [describe(t, now) + (f" [id {t.id}]" if with_ids else "") for t in ordered]
    return f"{_plural(len(ordered), 'timer')}: {', '.join(items)}."


def confirmation(t: NewTimer, now: datetime) -> str:
    """set_timer's tool result."""
    what = f"{t.kind} '{t.label}'" if t.label else t.kind
    if t.duration_s is not None:
        return f"Set {what} for {span(t.duration_s)} (fires at {clock(t.fire_ts, now)})."
    return f"Set {what} for {clock(t.fire_ts, now)}."


def announcement(t: Timer) -> str:
    """What the robot says when `t` goes off."""
    if t.kind == "reminder":
        if t.label:
            return f"Reminder: {t.label}."
        return "This is your reminder."
    if t.label:
        return f"Your {t.label} timer is done."
    if t.duration_s is not None:
        return f"Your {span(t.duration_s, adjective=True)} timer is done."
    return "Your timer is done."


def history_note(t: Timer) -> str:
    """The bracketed system-context line recorded as the opening of the
    announcement's exchange, so the thread stays user/assistant-alternating
    (the prompt already treats [brackets] as system context, not speech)."""
    what = f"{t.kind} '{t.label}'" if t.label else t.kind
    if t.duration_s is not None:
        what += f" ({span(t.duration_s)})"
    return f"[The {what} you set went off; you announced it.]"


def _now() -> datetime:
    return datetime.now().astimezone()
