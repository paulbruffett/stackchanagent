#include "servos.h"
#include <Arduino.h>
#include <M5Unified.h>
#include <Wire.h>
#include "config.h"

namespace scs {

namespace {

// --- Feetech SCS protocol ---------------------------------------------------
constexpr uint8_t INSTRUCTION_WRITE = 0x03;
constexpr uint8_t REG_GOAL_POSITION = 0x2A;

// --- PY32 I/O expander on the Kawaii base ----------------------------------
// Custom M5Stack chip at I2C 0x6F on Wire1 (internal bus). Pin 0 gates the
// 5 V servo rail. Registers come from m5stack/StackChan firmware,
// hal/drivers/PY32IOExpander_Class/PY32IOExpander_Class.cpp.
constexpr uint8_t IOE_I2C_ADDR     = 0x6F;
constexpr uint8_t IOE_REG_DIR_L    = 0x03;  // direction (1=output)
constexpr uint8_t IOE_REG_OUT_L    = 0x05;  // output level
constexpr uint8_t IOE_REG_PU_L     = 0x09;  // pull-up enable
constexpr uint8_t IOE_REG_PD_L     = 0x0B;  // pull-down enable
constexpr uint8_t IOE_SERVO_POWER_PIN = 0;

bool initialised = false;

// Read-modify-write a single bit. Set==true clears the bit otherwise.
bool ioe_write_bit(uint8_t reg, uint8_t bit, bool set) {
    Wire1.beginTransmission(IOE_I2C_ADDR);
    Wire1.write(reg);
    if (Wire1.endTransmission(false) != 0) return false;
    if (Wire1.requestFrom(IOE_I2C_ADDR, (uint8_t)1) != 1) return false;
    uint8_t v = Wire1.read();
    if (set) v |= (1 << bit);
    else     v &= ~(1 << bit);
    Wire1.beginTransmission(IOE_I2C_ADDR);
    Wire1.write(reg);
    Wire1.write(v);
    return Wire1.endTransmission() == 0;
}

// Build and send a Feetech SCS WRITE_DATA frame.
//
//   0xFF 0xFF | ID | LEN | INSTR | PARAM... | CHECKSUM
//
// LEN     = (#PARAM) + 2
// CHECKSUM = ~(ID + LEN + INSTR + sum(PARAM)) & 0xFF
void send_write(uint8_t id, uint8_t reg, const uint8_t* data, uint8_t n) {
    const uint8_t len = n + 3;  // reg byte + n data bytes + instr + checksum
    uint8_t sum = id + len + INSTRUCTION_WRITE + reg;
    for (uint8_t i = 0; i < n; i++) sum += data[i];
    const uint8_t checksum = ~sum;

    Serial2.write(0xFF);
    Serial2.write(0xFF);
    Serial2.write(id);
    Serial2.write(len);
    Serial2.write(INSTRUCTION_WRITE);
    Serial2.write(reg);
    Serial2.write(data, n);
    Serial2.write(checksum);
    Serial2.flush();
}

}  // namespace

bool begin() {
    // Power on the 5 V servo rail by configuring PY32 IOE pin 0 the same
    // way the factory firmware does: output + pull-up enabled + high.
    // The pull-up is essential — without it pin 0 floats (open-drain
    // default) and the rail never comes up.
    const auto pin = IOE_SERVO_POWER_PIN;
    bool ok = true;
    ok &= ioe_write_bit(IOE_REG_DIR_L, pin, true);   // direction = OUTPUT
    ok &= ioe_write_bit(IOE_REG_PD_L,  pin, false);  // pull-down off
    ok &= ioe_write_bit(IOE_REG_PU_L,  pin, true);   // pull-up   on
    ok &= ioe_write_bit(IOE_REG_OUT_L, pin, true);   // level     HIGH
    if (!ok) {
        Serial.println("[scs] PY32 IOE register writes failed");
        return false;
    }
    Serial.println("[scs] servo rail enabled via PY32 IOE pin 0");
    delay(200);  // let the 5 V servo rail settle

    Serial2.begin(cfg::SERVO_BUS_BAUD, SERIAL_8N1,
                  cfg::SERVO_BUS_RX_PIN, cfg::SERVO_BUS_TX_PIN);
    delay(50);
    initialised = true;
    return true;
}

bool move_to(uint8_t servo_id, int16_t position) {
    if (!initialised) return false;
    if (position < 0) position = 0;
    if (position > cfg::SERVO_POS_MAX) position = cfg::SERVO_POS_MAX;
    const uint8_t pos[2] = {
        static_cast<uint8_t>(position & 0xFF),
        static_cast<uint8_t>((position >> 8) & 0xFF),
    };
    send_write(servo_id, REG_GOAL_POSITION, pos, sizeof(pos));
    return true;
}

}  // namespace scs
