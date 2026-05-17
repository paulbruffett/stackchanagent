#include "mic_pump.h"

#include <vector>

#include <audio_codec.h>
#include <board.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <mooncake_log.h>

#include "transport.h"

namespace agent::mic_pump {

namespace {

constexpr const char* TAG = "agent.mic";

// 20 ms frames at the codec's input sample rate. At 24 kHz that's 480 samples;
// at 16 kHz it would be 320.
constexpr int kFrameMs = 20;

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
    int samples_per_frame = (sample_rate * kFrameMs) / 1000;
    mclog::tagInfo(TAG, "mic pump: {} Hz, {} samples / frame", sample_rate,
                   samples_per_frame);

    std::vector<int16_t> buf(samples_per_frame);

    while (true) {
        if (!codec->InputData(buf)) {
            vTaskDelay(pdMS_TO_TICKS(kFrameMs));
            continue;
        }
        // Send unconditionally; transport drops silently if not connected.
        transport::send_audio(buf.data(), buf.size());
    }
}

}  // namespace

void start()
{
    xTaskCreatePinnedToCore(task, "agent_mic", 4096, nullptr, 6, nullptr, 1);
}

}  // namespace agent::mic_pump
