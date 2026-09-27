/*
 * Over-the-air firmware update, pushed by the brain.
 *
 *   {"cmd":"ota","url":"http://<brain>:8080/firmware/<token>.bin",
 *    "size":N,"sha256":"<hex>"}
 *
 * The download runs on its own task (never the main loop): plain HTTP GET via
 * the board's network stack, streamed into the next OTA slot while hashing,
 * then size + SHA-256 + esp_ota_end's image check, set boot partition,
 * restart. Reports {"event":"ota","state":"downloading","pct":n} about every
 * 10 %, then "rebooting"; any failure sends "failed" + "error", aborts the
 * write and leaves the running firmware in place.
 *
 * Rollback: CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE boots a new image in
 * PENDING_VERIFY. confirm_tick() marks it valid only once the brain
 * WebSocket has connected; an image that can't reach the brain within
 * kConfirmDeadlineMs of boot rolls itself back and restarts into the
 * previous one (as does any crash/reset before that point).
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace agent::ota {

// Start an update. Refuses (sending a "failed" event) unless the device is
// IDLE and no update is running. Call from the command dispatcher.
void start(std::string_view url, size_t size, std::string_view sha256_hex);

// True from start() until the update fails (success restarts the device).
// Wakeword and head tap are ignored meanwhile.
bool in_progress();

// Boot-time image confirmation; call from the main loop every iteration.
void confirm_tick();

// Running firmware for the boot event: esp_app_desc_t version, and
// "<date> <time>" of the build (the version string alone doesn't change
// between builds).
std::string running_version();
std::string running_build();

}  // namespace agent::ota
