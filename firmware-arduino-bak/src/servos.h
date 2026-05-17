#pragma once
#include <stdint.h>

// Minimal Feetech SCS serial-bus servo driver for the StackChan Kawaii base.
// Sends "Write Goal Position" (register 0x2A) frames on Serial2.
// No read-back, no firmware-version detection, no easing — just open-loop
// position commands. Phase 0 verifies the wire; smoothing comes from the
// brain-side gaze controller in Phase 4.

namespace scs {

// Powers the servo rail (5 V on the Kawaii base) and opens Serial2.
// Call once in setup() before any move_to().
bool begin();

// Send a position command (0..1023, clamped). Returns false if begin()
// hasn't been called.
bool move_to(uint8_t servo_id, int16_t position);

}  // namespace scs
