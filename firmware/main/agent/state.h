/*
 * Agent state machine. Three states:
 *
 *   IDLE       — wakeword detection active; mic isn't streamed; speaker idle.
 *   LISTENING  — mic streams to brain; wakeword paused; speaker idle.
 *   SPEAKING   — speaker plays brain audio; mic + wakeword paused.
 *
 * Transitions are driven by:
 *   - wakeword callback / head tap (IDLE → LISTENING)
 *   - brain commands "stop_listening" / "start_listening" / "start_speaking" /
 *     "stop_speaking"
 *   - brain link drop (any → IDLE, transport.cpp)
 *   - the turn watchdog (commands::check_turn_watchdog): LISTENING or SPEAKING
 *     → IDLE when the brain goes quiet for too long while still connected
 *
 * The state is read by mic_pump (gating mic→brain), speaker_play (whether
 * to drain the queue), and wakeword (pause/resume).
 */
#pragma once

#include <cstdint>

namespace agent::state {

enum class Mode {
    Idle,
    Listening,
    Speaking,
};

Mode current();
void transition(Mode next);

// Monotonic milliseconds since boot (esp_timer). The one clock the agent
// layer uses for its timeouts.
int64_t now_ms();

// Watchdog helper: atomically (w.r.t. transition()) move to IDLE if the mode
// is still `mode` and more than `timeout_ms` has passed since the later of
// entering that mode and `last_activity_ms`. Re-reads both under the
// transition lock, so a transition that lands concurrently (a new episode)
// is never cut short. Returns true if it transitioned.
bool expire_if_stale(Mode mode, int64_t last_activity_ms, int64_t timeout_ms);

}  // namespace agent::state
