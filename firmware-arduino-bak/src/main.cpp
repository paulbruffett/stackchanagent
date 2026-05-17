// Phase 0 — hardware bring-up.
// Cycles through I2C scan → servos → mic record then speaker playback.
// (Mic and speaker share I2S0 on CoreS3, so we can't full-duplex; we
// record into a buffer, then play it back sequentially.)
// Camera bring-up is deferred to Phase 4.

#include <M5Unified.h>
#include <Wire.h>
#include <Avatar.h>
#include "config.h"
#include "servos.h"

using namespace m5avatar;

namespace {

Avatar avatar;

// 5 s of mono 16-bit @ 16 kHz = 160 000 bytes. Must live in internal SRAM
// (I2S DMA on ESP32-S3 can't access PSRAM — was the cause of the earlier
// garbled-hammering playback).
constexpr int kRecordSeconds = 5;
constexpr size_t kRecordSamples = cfg::AUDIO_SAMPLE_RATE * kRecordSeconds;
int16_t record_buf[kRecordSamples];  // 160 KB static; fits in ~320 KB DRAM

void scan_i2c_bus(const char* label, TwoWire& bus) {
    Serial.printf("[i2c] %s: ", label);
    int found = 0;
    for (uint8_t addr = 0x08; addr < 0x78; addr++) {
        bus.beginTransmission(addr);
        if (bus.endTransmission() == 0) {
            Serial.printf("0x%02X ", addr);
            found++;
        }
    }
    Serial.printf("(%d device%s)\n", found, found == 1 ? "" : "s");
}

void test_display() {
    M5.Display.fillScreen(BLACK);
    avatar.init();
    avatar.setExpression(Expression::Happy);
    avatar.setSpeechText("phase 0: bring-up");
    delay(2000);
    avatar.setSpeechText("");
}

void test_servos() {
    avatar.setSpeechText("servos");
    if (!scs::begin()) {
        avatar.setSpeechText("servo bus init failed");
        delay(1500);
        return;
    }

    // Center both servos so we have a known starting point.
    scs::move_to(cfg::SERVO_ID_X, cfg::SERVO_X_CENTER);
    scs::move_to(cfg::SERVO_ID_Y, cfg::SERVO_Y_CENTER);
    delay(800);

    // Tilt up then down within range.
    scs::move_to(cfg::SERVO_ID_Y, cfg::SERVO_Y_CENTER + cfg::SERVO_Y_AMPLITUDE);
    delay(800);
    scs::move_to(cfg::SERVO_ID_Y, cfg::SERVO_Y_CENTER - cfg::SERVO_Y_AMPLITUDE);
    delay(800);
    scs::move_to(cfg::SERVO_ID_Y, cfg::SERVO_Y_CENTER);
    delay(400);

    // Pan left then right.
    scs::move_to(cfg::SERVO_ID_X, cfg::SERVO_X_CENTER - cfg::SERVO_X_AMPLITUDE);
    delay(800);
    scs::move_to(cfg::SERVO_ID_X, cfg::SERVO_X_CENTER + cfg::SERVO_X_AMPLITUDE);
    delay(800);
    scs::move_to(cfg::SERVO_ID_X, cfg::SERVO_X_CENTER);
    delay(400);

    avatar.setSpeechText("");
}

void test_audio() {
    avatar.setSpeechText("listening 5s...");
    M5.Mic.begin();
    M5.Mic.record(record_buf, kRecordSamples, cfg::AUDIO_SAMPLE_RATE);
    while (M5.Mic.isRecording()) delay(10);
    M5.Mic.end();

    avatar.setSpeechText("playing back");
    M5.Speaker.begin();
    M5.Speaker.setVolume(200);
    M5.Speaker.playRaw(record_buf, kRecordSamples, cfg::AUDIO_SAMPLE_RATE, false, 1, 0);
    while (M5.Speaker.isPlaying()) delay(10);
    M5.Speaker.end();

    avatar.setSpeechText("");
}

}  // namespace

void setup() {
    auto m5cfg = M5.config();
    M5.begin(m5cfg);

    Serial.begin(115200);
    delay(200);
    Serial.println("[stackchan] phase 0 bring-up");

    // Ensure the BUS 5 V is on so the Kawaii base's PY32 IOE has power.
    M5.Power.setExtOutput(true);
    delay(300);

    // PY32 IOE (servo power gate) lives at 0x6F on Wire1; confirmed by
    // bring-up scan. Wire (external Port.A) isn't initialised by default
    // on CoreS3 — skip scanning it to avoid NULL TX buffer warnings.
    scan_i2c_bus("Wire1 (internal)", Wire1);

    test_display();
    test_servos();
    test_audio();

    avatar.setSpeechText("bring-up ok");
    avatar.setExpression(Expression::Neutral);
}

void loop() {
    M5.update();
    delay(20);
}
