/*
 * buddy_ble — Claude Desktop "Hardware Buddy" integration (firmware side).
 *
 * Sits on top of the NUS transport (buddy_ble_nus.c) and implements the
 * REFERENCE.md JSON protocol: it parses heartbeat snapshots + commands from
 * the desktop app, replies with acks / permission decisions, and arbitrates
 * the avatar face so a waiting tool-approval prompt shows up as a "Doubt"
 * face + "approve: <tool>" bubble — but only when the agent is otherwise
 * idle (a live conversation always wins) and coherently with sleep.
 *
 * Tap-to-approve: while a permission prompt is pending, a head tap approves
 * it (decision "once") instead of starting a listening turn.
 */
#pragma once

namespace agent::buddy_ble {

// Bring up the BLE peripheral and start advertising. Call once after Wi-Fi,
// and only when enabled(). tick() / prompt_pending() / approve_pending() are
// safe (inert) if it never ran.
void start();

// The persisted on/off setting (NVS "stackchan"/"buddy_ble", default off).
bool enabled();

// Brain's {"cmd":"set_buddy"}: if `on` differs from the stored setting, save
// it and restart the device (BLE is only brought up at boot). No-op when
// equal. Blocks ~300 ms before restarting; call from a task, never with the
// LVGL lock held.
void set_enabled(bool on);

// Drive the face/bubble arbitration. Call from the main idle loop, OUTSIDE
// the LVGL lock (it takes the lock itself when it needs to draw).
void tick();

// True while a desktop permission prompt is waiting for a decision.
bool prompt_pending();

// Approve the pending prompt (decision "once"). Called from the tap handler.
void approve_pending();

}  // namespace agent::buddy_ble
