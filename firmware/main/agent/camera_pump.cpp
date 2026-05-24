#include "camera_pump.h"

#include <cstddef>
#include <cstdint>

#include <board.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <mooncake_log.h>

#include "hal/board/stackchan_camera.h"
#include "image_to_jpeg.h"
#include "state.h"
#include "transport.h"

namespace agent::camera_pump {

namespace {

constexpr const char* TAG = "agent.cam";

// One frame every 1.5 s — matches the brain-side face detector cadence.
// Tune in Phase 4 if YOLO/MediaPipe wants more headroom.
constexpr uint32_t kIntervalMs = 1500;

// JPEG quality 60 keeps each frame ~8–15 KB at 320×240, fine over LAN.
constexpr uint8_t kJpegQuality = 60;

void task(void*)
{
    auto* base = Board::GetInstance().GetCamera();
    auto* cam = static_cast<StackChanCamera*>(base);
    if (!cam) {
        mclog::tagError(TAG, "no camera; task exiting");
        vTaskDelete(nullptr);
        return;
    }

    while (true) {
        if (state::current() == state::Mode::Speaking
            || !transport::is_connected()) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        // StreamCaptures (not Capture) — skips the shutter sfx + LVGL preview
        // overlay, which would clobber the avatar every 1.5 s.
        if (!cam->StreamCaptures()) {
            mclog::tagWarn(TAG, "StreamCaptures failed");
            vTaskDelay(pdMS_TO_TICKS(kIntervalMs));
            continue;
        }

        uint8_t* jpeg = nullptr;
        size_t jpeg_len = 0;
        bool ok = image_to_jpeg(
            const_cast<uint8_t*>(cam->GetFrameData()),
            cam->GetFrameSize(),
            static_cast<uint16_t>(cam->GetFrameWidth()),
            static_cast<uint16_t>(cam->GetFrameHeight()),
            static_cast<v4l2_pix_fmt_t>(cam->GetFrameFormat()),
            kJpegQuality, &jpeg, &jpeg_len);
        if (!ok || !jpeg || jpeg_len == 0) {
            mclog::tagWarn(TAG, "image_to_jpeg failed");
            if (jpeg) heap_caps_free(jpeg);
            vTaskDelay(pdMS_TO_TICKS(kIntervalMs));
            continue;
        }

        if (!transport::send_jpeg(jpeg, jpeg_len)) {
            mclog::tagWarn(TAG, "send_jpeg dropped {} bytes", jpeg_len);
        }
        heap_caps_free(jpeg);

        vTaskDelay(pdMS_TO_TICKS(kIntervalMs));
    }
}

}  // namespace

void start()
{
    // 6 KB stack: image_to_jpeg uses heap, but the encoder still needs
    // a few KB for locals.
    xTaskCreatePinnedToCore(task, "agent_cam", 6144, nullptr, 4, nullptr, 1);
}

}  // namespace agent::camera_pump
