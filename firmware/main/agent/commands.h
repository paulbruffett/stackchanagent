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
 *
 * The screen is also relit automatically by any activity command
 * (set_expression / look_at / set_busy / start_speaking), so the device
 * can never move or speak with the screen off even if the brain's notion
 * of sleep has drifted from the firmware's (e.g. after a brain restart).
 */
#pragma once

#include <string_view>

namespace agent::commands {

void dispatch(std::string_view json);

// Relight the screen if it was turned off for sleep; no-op otherwise.
// Called locally from the wake word / head-tap handlers so waking is
// instant and works even if the brain link is down. Idempotent.
void wake_face();

// Turn the screen off + show the sleepy face. Idempotent. Used by the
// brain's {"cmd":"sleep"} and by the BLE buddy to restore the off state
// after a prompt that arrived while asleep resolves.
void sleep_face();

// True while the screen is currently off for sleep.
bool face_is_off();

}  // namespace agent::commands
