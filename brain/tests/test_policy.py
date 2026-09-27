"""Unit tests for the pure decision helpers in policy.py.

These are kept dependency-free (no agent_server import) so they run offline on
any machine, unlike the server module which pulls in Jetson-only deps.
"""
import pytest

from policy import (
    DeviceStatus,
    buddy_sync_action,
    capture_is_stale,
    describe_device_status,
    effective_sleep_timeout,
    low_battery_check,
    step_volume,
    volume_sync_action,
)


class TestEffectiveSleepTimeout:
    def test_no_prompt_returns_base(self):
        assert effective_sleep_timeout(300.0, 1800.0, False) == 300.0

    def test_prompt_pending_uses_longer_prompt_timeout(self):
        assert effective_sleep_timeout(300.0, 1800.0, True) == 1800.0

    def test_prompt_pending_never_shortens_a_longer_base(self):
        # If the base idle timeout is already longer than the prompt timeout,
        # keep the base — a pending prompt should only ever delay sleep.
        assert effective_sleep_timeout(3600.0, 1800.0, True) == 3600.0

    def test_base_zero_with_prompt_still_holds_off(self):
        # SLEEP_TIMEOUT_S==0 (sleep disabled) is handled by the caller before
        # this helper, so here a 0 base with a pending prompt still elongates.
        assert effective_sleep_timeout(0.0, 1800.0, True) == 1800.0



class TestBuddySyncAction:
    def test_matching_report_sends_nothing(self):
        assert buddy_sync_action(0, False, None, busy=False) is None
        assert buddy_sync_action(1, True, None, busy=False) is None

    def test_differing_report_sends_and_closes(self):
        assert buddy_sync_action(1, False, None, busy=False) == "send_close"
        assert buddy_sync_action(0, True, None, busy=False) == "send_close"

    def test_already_sent_waits_for_the_reboot(self):
        assert buddy_sync_action(1, False, True, busy=False) is None

    def test_busy_defers(self):
        assert buddy_sync_action(1, False, None, busy=True) is None
        assert buddy_sync_action(1, None, None, busy=True) is None

    def test_no_report_sends_once_without_closing(self):
        assert buddy_sync_action(0, None, None, busy=False) == "send"
        assert buddy_sync_action(0, None, False, busy=False) is None

    def test_no_report_resends_on_knob_change(self):
        assert buddy_sync_action(1, None, False, busy=False) == "send"


class TestCaptureIsStale:
    def test_not_listening_is_never_stale(self):
        assert not capture_is_stale(False, 0.0, 1000.0, 10000, False, 4.5)

    def test_within_max_utterance_plus_grace(self):
        assert not capture_is_stale(True, 100.0, 112.9, 10000, False, 4.5)

    def test_past_max_utterance_plus_grace(self):
        assert capture_is_stale(True, 100.0, 113.1, 10000, False, 4.5)

    def test_follow_up_window_longer_than_max_utterance(self):
        # MAX_UTTERANCE_MS set low: a follow-up window may stay open longer.
        assert not capture_is_stale(True, 100.0, 110.0, 3000, True, 9.0)
        assert capture_is_stale(True, 100.0, 112.1, 3000, True, 9.0)
        # ...but a wakeword capture still uses the utterance cap.
        assert capture_is_stale(True, 100.0, 106.1, 3000, False, 9.0)


class TestVolumeSyncAction:
    def test_matching_report_sends_nothing(self):
        assert volume_sync_action(70, 70, None, busy=False) is None

    def test_differing_report_sends_the_knob(self):
        assert volume_sync_action(40, 70, None, busy=False) == 40

    def test_already_sent_waits_for_the_report(self):
        # The tool (or an earlier tick) sent it; the status event will catch up.
        assert volume_sync_action(40, 70, 40, busy=False) is None

    def test_knob_changed_again_after_a_send(self):
        assert volume_sync_action(55, 40, 40, busy=False) == 55

    def test_busy_defers(self):
        assert volume_sync_action(40, 70, None, busy=True) is None

    def test_no_report_never_sends(self):
        # Firmware without set_volume never reports a volume.
        assert volume_sync_action(40, None, None, busy=False) is None


class TestStepVolume:
    def test_up_and_down_step_by_15(self):
        assert step_volume(50, None, "up") == 65
        assert step_volume(50, None, "down") == 35

    def test_clamped(self):
        assert step_volume(95, None, "up") == 100
        assert step_volume(10, None, "down") == 0
        assert step_volume(50, 140, None) == 100
        assert step_volume(50, -5, None) == 0

    def test_level_wins_over_change(self):
        assert step_volume(50, 30, "up") == 30

    def test_neither_is_an_error(self):
        with pytest.raises(ValueError):
            step_volume(50, None, None)


class TestDeviceStatus:
    def test_nothing_reported_yet(self):
        assert describe_device_status(DeviceStatus()) == "No status from the robot's body yet."

    def test_unknown_battery_and_volume(self):
        d = DeviceStatus()
        d.update({"battery": None, "charging": None}, now=1.0)
        assert describe_device_status(d) == "Battery level unknown. Volume unknown."

    def test_full_report(self):
        d = DeviceStatus()
        d.update({"battery": 82, "charging": True, "volume": 55}, now=1.0)
        assert describe_device_status(d) == (
            "Battery at 82 percent, charging. Volume at 55 out of 100.")
        d.update({"charging": False}, now=2.0)
        assert "not charging" in describe_device_status(d)

    def test_garbage_fields_read_as_unknown(self):
        d = DeviceStatus()
        d.update({"battery": 255, "charging": "yes", "volume": True}, now=1.0)
        assert (d.battery, d.charging, d.volume) == (None, None, None)

    def test_low_battery_flag(self):
        d = DeviceStatus(battery=15, charging=False)
        assert d.low_battery
        assert not DeviceStatus(battery=15, charging=True).low_battery
        assert not DeviceStatus(battery=16, charging=False).low_battery
        assert not DeviceStatus().low_battery


class TestLowBatteryCheck:
    def test_warns_once_per_discharge_cycle(self):
        warned = False
        seen = []
        for pct in (20, 15, 14, 12, 10):
            warn, warned = low_battery_check(pct, False, warned)
            seen.append(warn)
        assert seen == [False, True, False, False, False]

    def test_charging_rearms(self):
        warn, warned = low_battery_check(10, True, True)
        assert (warn, warned) == (False, False)
        warn, warned = low_battery_check(10, False, warned)
        assert (warn, warned) == (True, True)

    def test_unknown_reading_changes_nothing(self):
        assert low_battery_check(None, False, True) == (False, True)
        assert low_battery_check(None, None, False) == (False, False)


def test_volume_sync_leaves_the_robot_alone_while_the_knob_is_default():
    from policy import volume_sync_action

    assert volume_sync_action(70, 35, None, busy=False, knob_set=False) is None
    assert volume_sync_action(70, 35, None, busy=False, knob_set=True) == 70
