/*
 * buddy_ble_nus — low-level NimBLE Nordic UART Service peripheral for the
 * Claude Desktop "Hardware Buddy" link (REFERENCE.md protocol).
 *
 * This is the transport layer only: it owns the NimBLE host, advertises as
 * "Claude StackChan", exposes the Nordic UART Service (NUS), pairs with LE
 * Secure Connections bonding (DisplayOnly), and chops the inbound RX stream
 * into newline-delimited JSON lines. All protocol parsing/serialization and
 * the UI live in the C++ layer (buddy_ble.cpp), which registers an on_line
 * callback and calls buddy_nus_notify() to reply.
 *
 * Written in C (not C++) so the NimBLE GATT service/UUID tables can use C
 * designated initializers freely, mirroring the factory bleprph scaffold.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Called once per complete '\n'-terminated line received on NUS RX. The
 * pointer is a NUL-terminated copy owned by the caller and only valid for
 * the duration of the call. Invoked from the NimBLE host task context. */
typedef void (*buddy_line_cb_t)(const char *line, int len);

/* Bring up NimBLE + NUS and start advertising. Call once after NVS/Wi-Fi
 * init (HAL::init has already initialised NVS). Safe to call from app_main. */
void buddy_nus_init(buddy_line_cb_t on_line);

/* True once a central is connected (regardless of encryption). */
bool buddy_nus_connected(void);

/* True once the current connection's link is encrypted (bonded/paired). */
bool buddy_nus_encrypted(void);

/* Send one line to the desktop over NUS TX (notification). The caller must
 * include the trailing '\n'. Returns 0 on success. No-op (returns nonzero)
 * if no central is subscribed. */
int buddy_nus_notify(const char *data, int len);

/* Erase all stored bonds (handles the {"cmd":"unpair"} request). */
void buddy_nus_unpair(void);

/* ---- Hooks implemented in the C++ layer (buddy_ble.cpp) ---- */

/* DisplayOnly pairing: show this 6-digit passkey to the user so they can
 * type it into the desktop app. */
void buddy_on_passkey(uint32_t passkey);

/* Central connected / disconnected — lets the C++ layer reset protocol
 * state (snapshot atomics, pending prompt) on link changes. */
void buddy_on_connect(void);
void buddy_on_disconnect(void);

#ifdef __cplusplus
}
#endif
