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

// Bring up the BLE peripheral and start advertising. Call once after Wi-Fi.
void start();

// Drive the face/bubble arbitration. Call from the main idle loop, OUTSIDE
// the LVGL lock (it takes the lock itself when it needs to draw).
void tick();

// True while a desktop permission prompt is waiting for a decision.
bool prompt_pending();

// Approve the pending prompt (decision "once"). Called from the tap handler.
void approve_pending();

// Invalidate the buddy render state after the avatar skin is swapped, so any
// pending prompt/PIN bubble (wiped by the rebuild) is redrawn on the next tick.
void notify_avatar_swapped();

}  // namespace agent::buddy_ble
