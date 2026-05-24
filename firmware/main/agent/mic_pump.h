/*
 * Mic pump: continuously read 20 ms PCM frames from the codec's mic and
 * push them to the brain over the transport. Phase 1 runs unconditionally
 * for the echo loop; later phases will gate this on wakeword state.
 */
#pragma once

namespace agent::mic_pump {

void start();

}  // namespace agent::mic_pump
