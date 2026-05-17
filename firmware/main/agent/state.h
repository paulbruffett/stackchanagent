/*
 * Agent state machine. Three states:
 *
 *   IDLE       — wakeword detection active; mic isn't streamed; speaker idle.
 *   LISTENING  — mic streams to brain; wakeword paused; speaker idle.
 *   SPEAKING   — speaker plays brain audio; mic + wakeword paused.
 *
 * Transitions are driven by:
 *   - wakeword callback (IDLE → LISTENING)
 *   - brain commands "stop_listening" / "start_speaking" / "stop_speaking"
 *
 * The state is read by mic_pump (gating mic→brain), speaker_play (whether
 * to drain the queue), and wakeword (pause/resume).
 */
#pragma once

namespace agent::state {

enum class Mode {
    Idle,
    Listening,
    Speaking,
};

Mode current();
void transition(Mode next);

}  // namespace agent::state
