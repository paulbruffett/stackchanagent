#include "wakeword.h"

#include <memory>

#include <assets.h>
#include <audio_codec.h>
#include <board.h>
#include <model_path.h>
#include <mooncake_log.h>
#include <wake_words/afe_wake_word.h>

namespace agent::wakeword {

namespace {

constexpr const char* TAG = "agent.ww";

struct State {
    std::unique_ptr<AfeWakeWord> impl;
    DetectedCallback cb;
    srmodel_list_t* models = nullptr;
};

State& state()
{
    static State s;
    return s;
}

// Load srmodels from the mmap'd `assets` partition (built by xiaozhi's
// build_default_assets.py with the wakenet selected in sdkconfig).
// xiaozhi's normal path is Assets::Apply() → AudioService::SetModelsList,
// but Apply() is gated on Application::GetInstance() which we don't run,
// so we replicate the relevant slice manually.
srmodel_list_t* load_models()
{
    auto& assets = Assets::GetInstance();
    void* ptr = nullptr;
    size_t size = 0;
    // The assets index.json names srmodels.bin specifically.
    if (!assets.GetAssetData("srmodels.bin", ptr, size)) {
        mclog::tagError(TAG, "srmodels.bin not found in assets partition");
        return nullptr;
    }
    auto* models = srmodel_load(static_cast<uint8_t*>(ptr));
    if (!models) {
        mclog::tagError(TAG, "srmodel_load failed");
        return nullptr;
    }
    mclog::tagInfo(TAG, "srmodels loaded ({} models)", models->num);
    return models;
}

}  // namespace

void start()
{
    auto& s = state();
    if (s.impl) return;

    s.models = load_models();
    if (!s.models) {
        return;
    }

    s.impl = std::make_unique<AfeWakeWord>();
    auto* codec = Board::GetInstance().GetAudioCodec();
    if (!s.impl->Initialize(codec, s.models)) {
        mclog::tagError(TAG, "AfeWakeWord init failed");
        s.impl.reset();
        return;
    }
    s.impl->OnWakeWordDetected([](const std::string& w) {
        mclog::tagInfo(TAG, "wakeword fired: {}", w);
        auto cb = state().cb;
        if (cb) cb(w);
    });
    s.impl->Start();
    mclog::tagInfo(TAG, "wakeword armed");
}

void pause()
{
    auto& s = state();
    if (s.impl) s.impl->Stop();
}

void resume()
{
    auto& s = state();
    if (s.impl) s.impl->Start();
}

void feed(const std::vector<int16_t>& pcm)
{
    auto& s = state();
    if (s.impl) s.impl->Feed(pcm);
}

void on_detected(DetectedCallback cb)
{
    state().cb = std::move(cb);
}

}  // namespace agent::wakeword
