"""Unit tests for the pure decision helpers in policy.py.

These are kept dependency-free (no agent_server import) so they run offline on
any machine, unlike the server module which pulls in Jetson-only deps.
"""
from policy import buddy_sync_action, capture_is_stale, effective_sleep_timeout


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
