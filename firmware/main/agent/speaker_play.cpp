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
    SemaphoreHandle_t sem = nullptr;
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
    auto& s = state();
    s.sem = xSemaphoreCreateCounting(kMaxQueuedFrames + 4, 0);
    xTaskCreatePinnedToCore(task, "agent_spk", 4096, nullptr, 6, nullptr, 1);
}

void push(const int16_t* samples, size_t sample_count)
{
    auto& s = state();
    {
        std::lock_guard<std::mutex> lock(s.mu);
        if (s.q.size() >= kMaxQueuedFrames) {
            // Drop oldest to keep latency bounded.
            s.q.pop();
            s.drops.fetch_add(1, std::memory_order_relaxed);
        }
        s.q.emplace(samples, samples + sample_count);
    }
    xSemaphoreGive(s.sem);
}

}  // namespace agent::speaker_play
