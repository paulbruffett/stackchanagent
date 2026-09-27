/*
 * Over-the-air firmware update, pushed by the brain.
 *
 *   {"cmd":"ota","id":n,"url":"http://<brain ip>:8080/firmware/<token>.bin",
 *    "size":N,"sha256":"<hex>"}
 *
 * The download runs on its own task (never the main loop): plain HTTP GET via
 * the board's network stack, streamed into the next OTA slot while hashing,
 * then size + SHA-256 + esp_ota_end's image check — which, with
 * CONFIG_SECURE_SIGNED_ON_UPDATE_NO_SECURE_BOOT, includes the RSA signature
 * against the key the running app was signed with — then set boot partition
 * and restart. Reports {"event":"ota","id":n,"state":"downloading","pct":p}
 * about every 10 %, then "rebooting"; any failure sends "failed" + "error",
 * aborts the write and leaves the running firmware in place. The URL must
 * point at the brain this robot is connected to (transport::brain_ip()).
 *
 * Rollback: CONFIG_BOOTLOADER_APP_ROLLBACK_ENABLE boots a new image in
 * PENDING_VERIFY; confirm_tick() decides whether it stays (see there).
 */
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace agent::ota {

// Start an update. Refuses (sending a "failed" event) unless the device is
// IDLE, no update is running, and the running image is itself confirmed.
// Call from the command dispatcher.
void start(std::string_view url, size_t size, std::string_view sha256_hex, int id);

// True from start() until the update fails (success restarts the device).
// Wakeword and head tap are ignored meanwhile.
bool in_progress();

// Boot-time image confirmation; call from the main loop every iteration.
void confirm_tick();

// True while the running image is a fresh OTA still on probation.
bool pending_verify();

// Running firmware for the boot event: esp_app_desc_t version, "<date>
// <time>" of the build, and the app ELF's SHA-256 (hex) — the one field that
// reliably tells two builds apart.
std::string running_version();
std::string running_build();
std::string running_elf_sha256();

}  // namespace agent::ota
