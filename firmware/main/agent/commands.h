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
 * BLE buddy (follows the brain's BUDDY_ENABLED; emitted on connect + change):
 *   {"cmd":"set_buddy","enabled":true|false}   persist to NVS; reboot if changed
 *
 * Speaker volume (follows the brain's SPEAKER_VOLUME; also the set_volume tool):
 *   {"cmd":"set_volume","value":0-100}   clamp, apply, persist to NVS; replies
 *                                        with a status event
 *
 * Dance (the brain's dance tool):
 *   {"cmd":"dance","style":"happy"|"robot"|"panic"}   one bounded (< 6 s) run of
 *       the stock DanceModifier keyframes, then back to the rest pose; ignored
 *       while a dance is already running
 * Firmware update (agent/ota.h):
 *   {"cmd":"ota","id":n,"url":"http://<brain ip>:…/firmware/<token>.bin",
 *              "size":N,"sha256":"<hex>"}
 *
 * The screen is also relit automatically by any activity command
 * (set_expression / look_at / set_busy / dance / start_speaking), so the device
 * can never move or speak with the screen off even if the brain's notion
 * of sleep has drifted from the firmware's (e.g. after a brain restart).
 */
#pragma once

#include <string>
#include <string_view>

namespace agent::commands {

void dispatch(std::string_view json);

// Registered as the transport's on_json handler: copies the frame onto a
// queue instead of dispatching it inline. Frames arrive on the esp-ml307
// "tcp_receive" task, and look_at drives the SCS servo bus that the main loop
// is already driving at 50 Hz — see the note above the queue in commands.cpp.
void enqueue(std::string_view json);

// Run every queued command on the calling task. Call from the main idle loop,
// OUTSIDE the LVGL lock (dispatch takes it itself).
void drain();

// Drop back to Idle if the device has been LISTENING for 15 s, or SPEAKING
// for 60 s, with no inbound brain frame in that time — a connected but wedged
// brain would otherwise leave the wakeword paused forever — and tell the
// brain ({"event":"listen_timeout"} / {"event":"speak_timeout"}). Call from
// the main idle loop, after drain().
void check_turn_watchdog();

// Device status fields for the boot and status events, without braces:
// "battery":<0-100|null>,"charging":<bool|null>,"volume":<0-100>.
// Reads the PMIC over I2C; safe from any task.
std::string status_fields();

// Start the status reporter task: {"event":"status",...} every 60 s, at once
// when the charging state flips (polled every 2 s), and after set_volume. Runs
// off the main loop so PMIC reads and WebSocket sends never stall it.
void start_status_reporter();

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
