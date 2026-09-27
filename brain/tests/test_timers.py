"""Voice timers and reminders: argument parsing and validation, persistence,
which timers are due (and which were missed while offline), the sentences the
robot says, the tools, and the HA fast-path guard. Times are built in the
system's local zone, as the brain uses them."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

import ha_fast_path
import timers
import tools
from claude_agent import validate_thread
from config import get_config
from conftest import FakeWs, model_calls, persisted_thread
from timers import Timer, TimerError

NOW = datetime(2026, 9, 26, 14, 0, 0).astimezone()   # 2 PM local


def _at(h: int, m: int = 0, days: int = 0) -> float:
    return (NOW.replace(hour=h, minute=m) + timedelta(days=days)).timestamp()


# --- parsing / validation ------------------------------------------------------

@pytest.mark.parametrize("text, hour, minute, days", [
    ("17:00", 17, 0, 0),
    ("5pm", 17, 0, 0),
    ("5:30 PM", 17, 30, 0),
    ("5 p.m.", 17, 0, 0),
    ("12 am", 0, 0, 1),        # midnight: already past today
    ("12pm", 12, 0, 1),        # noon: past
    ("9:30", 9, 30, 1),        # past today → tomorrow
    ("14:00", 14, 0, 1),       # exactly now → tomorrow
    ("14:01", 14, 1, 0),
])
def test_parse_at_next_occurrence(text, hour, minute, days):
    got = timers.parse_at(text, NOW)
    assert got.timestamp() == _at(hour, minute, days)


@pytest.mark.parametrize("text", ["", "noon", "25:00", "7:75", "13pm", "5 o'clock"])
def test_parse_at_rejects_nonsense(text):
    with pytest.raises(TimerError):
        timers.parse_at(text, NOW)


def test_new_timer_duration_and_at():
    t = timers.new_timer(600, None, " pasta ", NOW, 0)
    assert (t.fire_ts, t.duration_s, t.label, t.kind) == (NOW.timestamp() + 600, 600, "pasta", "timer")
    r = timers.new_timer(None, "17:00", "call mom", NOW, 0)
    assert (r.fire_ts, r.duration_s, r.kind) == (_at(17), None, "reminder")
    # Blank optional values are "not given", as models send them.
    assert timers.new_timer(60, "", "", NOW, 0).label is None
    assert timers.new_timer(0, "9:00", None, NOW, 0).fire_ts == _at(9, days=1)
    assert timers.new_timer(600, None, "check the oven", NOW, 0, kind="reminder").kind == "reminder"
    assert timers.new_timer(600, None, None, NOW, 0, kind="bogus").kind == "timer"


@pytest.mark.parametrize("duration, ok", [
    (1, True), (86400, True), (0.4, False), (86401, False), (-5, False), ("ten", False),
])
def test_duration_limits(duration, ok):
    if ok:
        assert timers.new_timer(duration, None, None, NOW, 0).duration_s == duration
    else:
        with pytest.raises(TimerError):
            timers.new_timer(duration, None, None, NOW, 0)


def test_exactly_one_of_duration_or_at_and_the_active_cap():
    with pytest.raises(TimerError, match="exactly one"):
        timers.new_timer(60, "17:00", None, NOW, 0)
    with pytest.raises(TimerError, match="exactly one"):
        timers.new_timer(None, None, "x", NOW, 0)
    timers.new_timer(60, None, None, NOW, timers.MAX_ACTIVE - 1)
    with pytest.raises(TimerError, match="cancel one"):
        timers.new_timer(60, None, None, NOW, timers.MAX_ACTIVE)


# --- persistence ---------------------------------------------------------------

def test_persistence_round_trip(mem, tmp_path):
    a = mem.add_timer("pasta", NOW.timestamp() + 240, 600, "timer")
    b = mem.add_timer("call mom", _at(17), None, "reminder")
    c = mem.add_timer(None, NOW.timestamp() + 30, 30, "timer")
    from memory import Memory
    again = Memory(tmp_path / "memory.db")   # a brain restart
    try:
        got = again.list_timers()
    finally:
        again.close()
    assert [t.id for t in got] == [c.id, a.id, b.id]          # soonest first
    assert got[1] == a and got[2] == b
    assert mem.delete_timer(a.id) and not mem.delete_timer(a.id)
    assert [t.id for t in mem.list_timers()] == [c.id, b.id]


# --- due selection -------------------------------------------------------------

def _t(id_: int, fire_ts: float, label: str | None = None) -> Timer:
    return Timer(id_, label, fire_ts, 0.0, 60)


def test_partition_due_and_the_late_reconnect_rule():
    now = NOW.timestamp()
    future = _t(1, now + 5)
    just_due = _t(2, now)
    late_ok = _t(3, now - timers.LATE_GRACE_S)          # exactly at the limit
    too_late = _t(4, now - timers.LATE_GRACE_S - 1)
    # Connected just now: the two old ones came due while we were offline.
    due, missed = timers.partition_due([future, just_due, too_late, late_ok], now, now)
    assert due == [late_ok, just_due]                     # oldest first
    assert missed == [too_late]
    assert timers.partition_due([], now, now) == ([], [])


def test_a_timer_that_waited_behind_a_conversation_is_never_missed():
    # Connected an hour ago; this came due 30 minutes ago, while a long
    # conversation held the announcement. Late, but not offline: announce it.
    now = NOW.timestamp()
    waited = _t(1, now - 1800)
    offline = _t(2, now - 3600 - timers.LATE_GRACE_S - 1)
    due, missed = timers.partition_due([waited, offline], now, now - 3600)
    assert due == [waited] and missed == [offline]


# --- sentences -----------------------------------------------------------------

def test_list_formatting():
    pasta = Timer(1, "pasta", NOW.timestamp() + 240, 0, 600)
    mom = Timer(2, "call mom", _at(17), 0, None, "reminder")
    assert timers.format_list([mom, pasta], NOW) == "2 timers: pasta in 4 minutes, call mom at 5 PM."
    assert timers.format_list([pasta], NOW, with_ids=True) == "1 timer: pasta in 4 minutes [id 1]."
    assert timers.format_list([], NOW) == "No timers are set."
    plain = Timer(3, None, NOW.timestamp() + 3900, 0, 5400)
    assert timers.describe(plain, NOW) == "1 hour 30 minute timer in 1 hour 5 minutes"
    tomorrow = Timer(4, None, _at(9, 30, days=1), 0, None, "reminder")
    assert timers.describe(tomorrow, NOW) == "reminder at 9:30 AM tomorrow"
    assert timers.describe(Timer(5, "tea", NOW.timestamp() + 45, 0, 60), NOW) == "tea in 45 seconds"


def test_announcements():
    assert timers.announcement(Timer(1, None, 0, 0, 600)) == "Your 10 minute timer is done."
    assert timers.announcement(Timer(1, "pasta", 0, 0, 600)) == "Your pasta timer is done."
    assert timers.announcement(Timer(1, "call mom", 0, 0, None, "reminder")) == "Reminder: call mom."
    assert timers.announcement(Timer(1, "check the oven", 0, 0, 600, "reminder")) == \
        "Reminder: check the oven."
    assert timers.announcement(Timer(1, None, 0, 0, 90)) == "Your 1 minute 30 second timer is done."


def test_history_note_keeps_the_thread_valid():
    note = timers.history_note(Timer(1, "pasta", 0, 0, 600))
    assert note.startswith("[") and note.endswith("]")
    assert validate_thread([{"role": "user", "content": note},
                            {"role": "assistant", "content": "Your pasta timer is done."}]) == []


PASTA = Timer(1, "pasta", 0, 0, 600)
MOM = Timer(2, "call mom", 0, 0, None, "reminder")
TEA = Timer(3, "tea", 0, 0, 300)


def test_match_cancel_by_id_label_and_generic():
    both = [PASTA, MOM]
    assert timers.match_cancel(both, "2") == [MOM]
    assert timers.match_cancel(both, "Pasta") == [PASTA]           # exact, any case
    assert timers.match_cancel(both, "the pasta timer") == [PASTA]  # whole words
    assert timers.match_cancel(both, "mom") == [MOM]
    for generic in ("the timer", "it", "timer", "that"):
        assert timers.match_cancel([PASTA], generic) == [PASTA]
    with pytest.raises(TimerError, match="No timers"):
        timers.match_cancel([], "pasta")


@pytest.mark.parametrize("target", ["pa", "past", "om", "call m"])
def test_match_cancel_never_matches_a_fragment(target):
    with pytest.raises(TimerError, match="No timer matches"):
        timers.match_cancel([PASTA, MOM], target)


def test_match_cancel_generic_words_come_before_labels():
    # A timer labelled "timer" must not turn "the timer" into a pick when
    # several are set: the generic word asks which.
    odd = Timer(4, "timer", 0, 0, 60)
    with pytest.raises(TimerError, match=r"ask which.*\[id 1\].*\[id 4\]"):
        timers.match_cancel([PASTA, odd], "the timer")


def test_match_cancel_ambiguous_label_lists_the_candidates():
    red, blue = Timer(5, "red pasta", 0, 0, 60), Timer(6, "blue pasta", 0, 0, 60)
    with pytest.raises(TimerError, match=r"ask which.*red pasta.*\[id 5\].*blue pasta.*\[id 6\]"):
        timers.match_cancel([red, blue, MOM], "pasta")
    assert timers.match_cancel([red, blue], "red pasta") == [red]     # exact wins


def test_match_cancel_all_variants():
    every = [PASTA, MOM, TEA]
    assert timers.match_cancel(every, "all") == every
    assert timers.match_cancel(every, "All.") == every
    assert timers.match_cancel(every, "all timers") == [PASTA, TEA]
    assert timers.match_cancel(every, "all reminders") == [MOM]
    assert timers.match_cancel([], "all") == []


# --- DST ---------------------------------------------------------------------------

@pytest.fixture
def los_angeles(monkeypatch):
    """Run in America/Los_Angeles whatever the host's zone is."""
    import time
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _la(*args) -> datetime:
    from zoneinfo import ZoneInfo
    return datetime(*args, tzinfo=ZoneInfo("America/Los_Angeles"))


@pytest.mark.parametrize("now, at, expected", [
    # Fall back (Sun Nov 1 2026, 2:00 PDT -> 1:00 PST).
    (_la(2026, 10, 31, 23, 0), "9:00", _la(2026, 11, 1, 9, 0)),    # tomorrow, 11 h away
    (_la(2026, 11, 1, 0, 30), "9:00", _la(2026, 11, 1, 9, 0)),     # today, across the change
    (_la(2026, 11, 1, 10, 0), "9:00", _la(2026, 11, 2, 9, 0)),
    # Spring forward (Sun Mar 8 2026, 2:00 PST -> 3:00 PDT).
    (_la(2026, 3, 7, 23, 0), "9:00", _la(2026, 3, 8, 9, 0)),       # tomorrow, 9 h away
    (_la(2026, 3, 8, 0, 30), "17:00", _la(2026, 3, 8, 17, 0)),     # today, across the change
])
def test_parse_at_uses_the_zone_rules_for_that_date(los_angeles, now, at, expected):
    got = timers.parse_at(at, now.astimezone())   # a fixed-offset "now", as the brain has
    assert got.timestamp() == expected.timestamp()
    assert (got.hour, got.minute) == (expected.hour, expected.minute)


def test_the_fall_back_morning_is_eleven_hours_after_11pm(los_angeles):
    now = _la(2026, 10, 31, 23, 0).astimezone()
    assert timers.parse_at("9:00", now).timestamp() - now.timestamp() == 11 * 3600


# --- tools ---------------------------------------------------------------------

def _ctx(mem):
    return tools.ToolContext(ws=FakeWs(), memory=mem)


async def test_set_list_cancel_through_dispatch(mem):
    ctx = _ctx(mem)
    got = await tools.dispatch("set_timer", {"duration_seconds": 600, "label": "pasta"}, ctx)
    assert got.startswith("Set timer 'pasta' for 10 minutes")
    await tools.dispatch("set_timer", {"at": "23:59", "label": "call mom"}, ctx)
    listing = await tools.dispatch("list_timers", {}, ctx)
    assert listing.startswith("2 timers: ")
    assert "pasta in 10 minutes [id 1]" in listing and "call mom at 11:59 PM" in listing
    assert (await tools.dispatch("cancel_timer", {"label_or_id": "pasta"}, ctx)).startswith(
        "Cancelled 1: pasta")
    assert [t.label for t in mem.list_timers()] == ["call mom"]


@pytest.mark.parametrize("name, args", [
    ("set_timer", {"duration_seconds": 90000}),
    ("set_timer", {"duration_seconds": 60, "at": "17:00"}),
    ("set_timer", {}),
    ("cancel_timer", {"label_or_id": "pasta"}),      # nothing set
    ("cancel_timer", {"label_or_id": "all"}),        # nothing set
])
async def test_timer_tool_errors_are_error_results(mem, name, args):
    result = await tools.dispatch(name, args, _ctx(mem))
    assert tools.is_error_result(result)
    assert mem.list_timers() == []


async def test_set_timer_with_confirmation_is_one_round(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("tool", "set_timer", "c1", "Ten minute timer, starting now.",
                        '{"duration_seconds": 600}')])
    await sess.respond("set a timer for ten minutes", speak)
    assert len(model_calls(sess)) == 1
    assert spoken == ["Ten minute timer, starting now."]
    assert [t.duration_s for t in mem.list_timers()] == [600]
    assert validate_thread(persisted_thread(mem)) == []


async def test_invalid_set_timer_gets_a_second_round(mem, make_agent, speaker):
    spoken, speak = speaker
    sess = make_agent([("tool", "set_timer", "c1", "Timer set for two days.",
                        '{"duration_seconds": 172800}'),
                       ("text", "Sorry, timers can only run up to a day.")])
    await sess.respond("set a timer for two days", speak)
    assert len(model_calls(sess)) == 2
    assert mem.list_timers() == []


async def test_list_timers_gets_a_second_round(mem, make_agent, speaker):
    spoken, speak = speaker
    mem.add_timer("pasta", NOW.timestamp() + 10**6, 600, "timer")
    sess = make_agent([("tool", "list_timers", "c1", "Let me check."),
                       ("text", "Your pasta timer is still running.")])
    await sess.respond("what timers do I have", speak)
    assert len(model_calls(sess)) == 2
    tool_msg = [m for m in model_calls(sess)[1]["messages"] if m.get("role") == "tool"]
    assert tool_msg[0]["content"].startswith("1 timer: pasta in")


def test_home_assistants_timer_intents_are_never_offered(mem, make_agent):
    import claude_agent

    class FakeMcp:
        def tool_defs(self):
            names = ["HassTurnOn", "HassStartTimer", "HassCancelTimer",
                     "HassCancelAllTimers", "HassIncreaseTimer", "HassDecreaseTimer",
                     "HassPauseTimer", "HassUnpauseTimer", "HassTimerStatus"]
            return [{"name": f"mcp__homeassistant__{n}", "description": "",
                     "input_schema": {"type": "object", "properties": {}}} for n in names]

    sess = make_agent([])
    sess._tool_ctx.mcp = FakeMcp()
    offered = {d["function"]["name"] for d in sess._tool_defs()}
    assert "mcp__homeassistant__HassTurnOn" in offered
    assert not any("Timer" in n for n in offered if n.startswith("mcp__"))
    assert {"set_timer", "list_timers", "cancel_timer"} <= offered
    assert not claude_agent._is_ha_action("mcp__homeassistant__HassCancelAllTimers")


def test_restore_timer_puts_a_claimed_timer_back(mem):
    t = mem.add_timer("pasta", NOW.timestamp(), 600, "timer")
    assert mem.delete_timer(t.id)
    mem.restore_timer(t)
    assert mem.list_timers() == [t]


def test_ends_turn_with_timer_tools():
    import claude_agent
    ends = claude_agent._ends_turn
    assert ends(["set_timer"]) and ends(["cancel_timer"])
    assert ends(["set_expression", "set_timer"])
    assert ends(["set_timer", "mcp__homeassistant__HassTurnOn"])
    assert not ends(["list_timers"])
    assert not ends(["set_timer", "list_timers"])
    assert not ends(["set_timer", "mcp__weather__get_weather"])
    # Timer tools are local and instant: no busy bubble / ack filler.
    assert not claude_agent._has_slow_tool(["set_timer", "list_timers", "cancel_timer"])


# --- Home Assistant fast path guard ----------------------------------------------

@pytest.mark.parametrize("text, is_timer", [
    ("set a timer for 5 minutes", True),
    ("5 minute timer", True),
    ("cancel all timers", True),
    ("remind me at 5 pm to call mom", True),
    ("set an alarm for 7", True),
    ("turn off the office light", False),
    ("what's the time", False),
    ("is the timeroom light on", False),
])
def test_is_timer_request(text, is_timer):
    assert ha_fast_path.is_timer_request(text) is is_timer


async def test_timer_wording_never_reaches_home_assistant(mem, monkeypatch):
    # HA recognises "cancel all timers" (HassCancelAllTimers) and, with no
    # Assist satellite behind the request, answers action_done for HA's own
    # (empty) timer list — which would be spoken while ours keep running.
    monkeypatch.setenv("HA_TOKEN", "t")
    get_config().set("HA_FAST_PATH", 1)

    async def boom(text):
        raise AssertionError("HA was asked about a timer phrase")

    monkeypatch.setattr(ha_fast_path, "_process", boom)
    assert await ha_fast_path.try_handle("cancel all timers") is None
    assert await ha_fast_path.try_handle("remind me in ten minutes to stir") is None
