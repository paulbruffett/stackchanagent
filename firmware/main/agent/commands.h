/*
 * Dispatch JSON commands from the brain to firmware state changes.
 * Registered with transport::set_on_json() during startup.
 *
 * Recognized commands (Phase 2):
 *   {"cmd":"stop_listening"}   LISTENING → IDLE  (cancel without speech)
 *   {"cmd":"start_speaking"}   LISTENING or IDLE → SPEAKING
 *   {"cmd":"stop_speaking"}    SPEAKING → IDLE
 *
 * Sleep (brain inactivity timer):
 *   {"cmd":"sleep"}            screen off + sleepy face (wake word/tap wakes)
 *   {"cmd":"wake"}             restore screen (also done locally on input)
 */
#pragma once

#include <string_view>

namespace agent::commands {

void dispatch(std::string_view json);

// Relight the screen if it was turned off for sleep; no-op otherwise.
// Called locally from the wake word / head-tap handlers so waking is
// instant and works even if the brain link is down. Idempotent.
void wake_face();

}  // namespace agent::commands
