#include "speaker_play.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include <audio_codec.h>
#include <board.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#include <mooncake_log.h>

namespace agent::speaker_play {

namespace {

constexpr const char* TAG = "agent.spk";

// Cap the playback queue so a stuck speaker can't run us out of heap.
// At 24 kHz × 20 ms frames × 50 frames = 1 s of audio buffered max.
constexpr size_t kMaxQueuedFrames = 50;

struct State {
    std::mutex mu;
    std::queue<std::vector<int16_t>> q;
    // Created with the rest of the module state rather than in start():
    // main.cpp registers push() as the transport's audio handler before the
    // speaker task exists, so a frame arriving in that window would reach
    // xSemaphoreGive(nullptr) and trip xQueueGenericSend's configASSERT.
    SemaphoreHandle_t sem = xSemaphoreCreateCounting(kMaxQueuedFrames + 4, 0);
    std::atomic<size_t> drops{0};
};

State& state()
{
    static State s;
    return s;
}

void task(void*)
{
    auto& s = state();
    auto* codec = Board::GetInstance().GetAudioCodec();
    if (!codec) {
        mclog::tagError(TAG, "no audio codec; task exiting");
        vTaskDelete(nullptr);
        return;
    }
    codec->EnableOutput(true);

    while (true) {
        // Block until something arrives.
        xSemaphoreTake(s.sem, portMAX_DELAY);

        std::vector<int16_t> frame;
        {
            std::lock_guard<std::mutex> lock(s.mu);
            if (s.q.empty()) continue;
            frame = std::move(s.q.front());
            s.q.pop();
        }
        codec->OutputData(frame);
    }
}

}  // namespace

void start()
{
    xTaskCreatePinnedToCore(task, "agent_spk", 4096, nullptr, 6, nullptr, 1);
}

void push(const int16_t* samples, size_t sample_count)
{
    auto& s = state();
    bool dropped = false;
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (s.q.size() >= kMaxQueuedFrames) {
            // Drop oldest to keep latency bounded.
            s.q.pop();
            s.drops.fetch_add(1, std::memory_order_relaxed);
            dropped = true;
        }
        s.q.emplace(samples, samples + sample_count);
    }
    if (dropped) {
        // The brain paces frames at 0.8x real time so it leads the queue, and
        // we drain at exactly real time — a long uninterrupted synthesis span
        // saturates this 1 s buffer and then deletes 20 ms of speech per
        // arriving frame. `drops` was write-only, so that was silent. Warn,
        // rate-limited: at saturation the drop branch fires every 20 ms.
        static TickType_t last_warn = 0;
        TickType_t now = xTaskGetTickCount();
        if (last_warn == 0 || now - last_warn >= pdMS_TO_TICKS(5000)) {
            last_warn = now;
            mclog::tagWarn(TAG, "playback queue overflow; {} frames dropped so far",
                           s.drops.load());
        }
    }
    xSemaphoreGive(s.sem);
}

}  // namespace agent::speaker_play
