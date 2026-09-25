"""Unit tests for the pure decision helpers in policy.py.

These are kept dependency-free (no agent_server import) so they run offline on
any machine, unlike the server module which pulls in Jetson-only deps.
"""
from policy import effective_sleep_timeout


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
