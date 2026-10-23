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
    # Resolve the wall-clock time in the system zone for that date, not with
    # `now`'s UTC offset: astimezone() on a naive datetime asks the zone
    # rules (mktime) about that date, so 9:00 the morning after "fall back"
    # is 9:00 PST, not 9:00 at yesterday's PDT offset.
    today = now.astimezone().date()
    for day in (today, today + timedelta(days=1)):
        target = datetime(day.year, day.month, day.day, hour, minute).astimezone()
        if target > now:
            return target
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
    timers: list[Timer], now_ts: float, connected_ts: float,
    grace_s: float = LATE_GRACE_S,
) -> tuple[list[Timer], list[Timer]]:
    """(due, missed), oldest first. `missed` are timers that came due more
    than `grace_s` before this connection started — the robot was offline —
    and are dropped instead of fired. Every other timer whose time has come
    is due, however late: one that waited behind a long conversation is
    still announced once it ends."""
    due: list[Timer] = []
    missed: list[Timer] = []
    for t in sorted(timers, key=lambda t: t.fire_ts):
        if t.fire_ts > now_ts:
            continue
        (missed if t.fire_ts < connected_ts - grace_s else due).append(t)
    return due, missed


_GENERIC = {"", "timer", "the timer", "my timer", "this timer", "that timer",
            "reminder", "the reminder", "my reminder", "it", "that", "this"}
_ALL = {"all": None, "everything": None, "all of them": None,
        "all timers": "timer", "all the timers": "timer", "all my timers": "timer",
        "all reminders": "reminder", "all the reminders": "reminder",
        "all my reminders": "reminder"}


def _ambiguous(hits: list[Timer], target: object) -> TimerError:
    return TimerError(
        f"{target!r} could mean more than one; ask which. "
        + format_list(hits, _now(), with_ids=True))


def _whole_words(needle: str, hay: str) -> bool:
    return bool(re.search(rf"\b{re.escape(needle)}\b", hay))


def match_cancel(timers: list[Timer], target: object) -> list[Timer]:
    """The timers cancel_timer(target) means, in order: "all" (everything),
    "all timers" / "all reminders" (that kind only); an id; a generic "the
    timer" / "it" (the only one set); a label, case-insensitive — exact
    first, else as whole words either way round ("pasta" ~ "pasta timer",
    never "pa"). A request that fits several timers, or none, raises
    TimerError listing what is set (with ids) so the model can ask."""
    text = str(target if target is not None else "").strip().lower().rstrip(".!?")
    if text in _ALL:
        kind = _ALL[text]
        return [t for t in timers if kind is None or t.kind == kind]
    if not timers:
        raise TimerError("No timers are set.")
    if text.isdigit():
        hit = [t for t in timers if t.id == int(text)]
        if hit:
            return hit
    if text in _GENERIC:
        if len(timers) == 1:
            return list(timers)
        raise _ambiguous(timers, target)
    labelled = [t for t in timers if t.label]
    exact = [t for t in labelled if t.label.lower() == text]
    hits = exact or [t for t in labelled
                     if _whole_words(t.label.lower(), text) or _whole_words(text, t.label.lower())]
    if len(hits) == 1:
        return hits
    if hits:
        raise _ambiguous(hits, target)
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
