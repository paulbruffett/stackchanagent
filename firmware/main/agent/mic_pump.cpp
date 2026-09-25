#include "mic_pump.h"

#include <atomic>
#include <vector>

#include <audio_codec.h>
#include <board.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>
#include <mooncake_log.h>

#include "state.h"
#include "transport.h"
#include "wakeword.h"

namespace agent::mic_pump {

namespace {

constexpr const char* TAG = "agent.mic";

// 20 ms frames at the codec's input sample rate. At 24 kHz that's 480 samples;
// at 16 kHz it would be 320.
constexpr int kFrameMs = 20;

// Frames buffered between the reader and the sender: 50 × 20 ms = 1 s. The
// reader used to call send_audio() itself, and the only slack behind it was
// the I2S DMA ring (~90 ms). A socket write that stalled longer than that —
// a camera JPEG holding the WebSocket send mutex, a TCP window waiting on an
// ACK through modem sleep — let the codec overwrite audio with no trace, and
// the brain received ~1 s of speech per 10 s of listening. Now a slow write
// only delays audio; a full queue is counted and logged.
constexpr int kQueueFrames = 50;

QueueHandle_t g_queue = nullptr;
int g_samples_per_frame = 0;

// Per-second counters, logged from the reader while LISTENING.
std::atomic<uint32_t> g_read{0};
std::atomic<uint32_t> g_dropped{0};
std::atomic<uint32_t> g_sent{0};
std::atomic<uint32_t> g_send_failed{0};
std::atomic<uint32_t> g_max_send_us{0};

void log_stats_if_due(int64_t& last_log_us)
{
    int64_t now = esp_timer_get_time();
    if (now - last_log_us < 1000000) return;
    last_log_us = now;
    uint32_t read = g_read.exchange(0);
    if (read == 0) return;
    mclog::tagInfo(TAG, "uplink/s: read={} dropped={} sent={} failed={} max_send={}ms queued={}",
                   read, g_dropped.exchange(0), g_sent.exchange(0), g_send_failed.exchange(0),
                   g_max_send_us.exchange(0) / 1000, uxQueueMessagesWaiting(g_queue));
}

void sender_task(void*)
{
    std::vector<int16_t> frame(g_samples_per_frame);
    while (true) {
        xQueueReceive(g_queue, frame.data(), portMAX_DELAY);
        int64_t t0 = esp_timer_get_time();
        bool ok = transport::send_audio(frame.data(), frame.size());
        uint32_t us = static_cast<uint32_t>(esp_timer_get_time() - t0);
        (ok ? g_sent : g_send_failed).fetch_add(1);
        uint32_t prev = g_max_send_us.load();
        while (us > prev && !g_max_send_us.compare_exchange_weak(prev, us)) {
        }
    }
}

void task(void*)
{
    auto* codec = Board::GetInstance().GetAudioCodec();
    if (!codec) {
        mclog::tagError(TAG, "no audio codec; task exiting");
        vTaskDelete(nullptr);
        return;
    }

    codec->Start();
    codec->EnableInput(true);

    int sample_rate = codec->input_sample_rate();
    g_samples_per_frame = (sample_rate * kFrameMs) / 1000;
    mclog::tagInfo(TAG, "mic pump: {} Hz, {} samples / frame", sample_rate,
                   g_samples_per_frame);

    // PSRAM: 50 frames is ~32 KB, too much to take from internal RAM.
    g_queue = xQueueCreateWithCaps(kQueueFrames, g_samples_per_frame * sizeof(int16_t),
                                   MALLOC_CAP_SPIRAM);
    if (!g_queue) {
        mclog::tagError(TAG, "mic queue alloc failed; task exiting");
        vTaskDelete(nullptr);
        return;
    }
    // Core 0 beside the WebSocket task, so a blocked write never competes
    // with this reader on core 1.
    xTaskCreatePinnedToCore(sender_task, "agent_mic_tx", 4096, nullptr, 5, nullptr, 0);

    std::vector<int16_t> buf(g_samples_per_frame);
    int64_t last_log_us = esp_timer_get_time();

    while (true) {
        auto mode = state::current();
        // Mic stays open during IDLE so wakeword can hear; muted during
        // SPEAKING to avoid feeding our own TTS back to STT.
        if (mode == state::Mode::Speaking) {
            vTaskDelay(pdMS_TO_TICKS(kFrameMs));
            continue;
        }
        if (!codec->InputData(buf)) {
            vTaskDelay(pdMS_TO_TICKS(kFrameMs));
            continue;
        }
        if (mode == state::Mode::Idle) {
            wakeword::feed(buf);
        } else if (mode == state::Mode::Listening) {
            g_read.fetch_add(1);
            if (xQueueSend(g_queue, buf.data(), 0) != pdTRUE) g_dropped.fetch_add(1);
        }
        log_stats_if_due(last_log_us);
    }
}

}  // namespace

void start()
{
    xTaskCreatePinnedToCore(task, "agent_mic", 4096, nullptr, 6, nullptr, 1);
}

}  // namespace agent::mic_pump
