#pragma once

namespace cfg {

// Servos on the M5StackChan Kawaii base are Feetech STS-series serial-bus
// servos. Shared half-duplex UART on Serial2 @ 1 Mbaud. Pinout matches
// the official BSP (m5stack/StackChan firmware/main/hal/hal_servo.cpp):
//   RX = GPIO 6, TX = GPIO 7.
constexpr int SERVO_BUS_RX_PIN = 6;
constexpr int SERVO_BUS_TX_PIN = 7;
constexpr uint32_t SERVO_BUS_BAUD = 1000000;
constexpr uint8_t SERVO_ID_X = 1;  // yaw / pan
constexpr uint8_t SERVO_ID_Y = 2;  // pitch / tilt

// STS position is 12-bit (0..4095). Center = 2048. Small amplitude for
// safe bring-up; tune up once we can confirm motion direction.
constexpr int16_t SERVO_POS_MAX = 4095;
constexpr int16_t SERVO_X_CENTER = 2048;
constexpr int16_t SERVO_X_AMPLITUDE = 300;  // ~ ±26° at 11.4 ticks/°
constexpr int16_t SERVO_Y_CENTER = 2048;
constexpr int16_t SERVO_Y_AMPLITUDE = 200;

// Camera FOV — GC0308 0.3MP module shipped with M5StackChan Kawaii.
// Used by the brain's gaze controller to map face-pixel → servo angle.
constexpr float CAMERA_HFOV_DEG = 60.0f;
constexpr float CAMERA_VFOV_DEG = 45.0f;
constexpr int CAMERA_W = 640;
constexpr int CAMERA_H = 480;

// Audio — I2S in/out share the CoreS3's onboard ES7210 (mic) + AW88298 (spk).
// M5Unified handles pin routing; we only set sample rates.
constexpr int AUDIO_SAMPLE_RATE = 16000;  // matches Whisper's native rate
constexpr int AUDIO_BITS_PER_SAMPLE = 16;

// Network — brain discovered via mDNS; no hardcoded IP.
constexpr const char* BRAIN_MDNS_HOST = "stackchan-brain";
constexpr int BRAIN_WS_PORT = 8765;

}  // namespace cfg
