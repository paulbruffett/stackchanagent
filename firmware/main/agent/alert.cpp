#include "alert.h"

#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>

#include <hal/hal.h>
#include <mooncake_log.h>
#include <stackchan/avatar/avatar/elements/emotion.h>
#include <stackchan/stackchan.h>

#include "commands.h"
#include "speaker_play.h"
#include "state.h"
#include "transport.h"

namespace agent::alert {

namespace {

constexpr const char* TAG = "agent.alert";

// Matches the codec output rate (hal/board/config.h AUDIO_OUTPUT_SAMPLE_RATE)
// and the brain's 20 ms frames.
constexpr int kSampleRate = 16000;
constexpr int kFrameSamples = kSampleRate / 50;

// Three rising notes (C6, E6, G6), each struck 150 ms after the last and left
// to ring out: ~0.75 s, 38 frames — inside speaker_play's 50-frame queue, so
// the whole chime is pushed at once without dropping any of it.
constexpr float kNotesHz[] = {1046.5f, 1318.5f, 1568.0f};
constexpr int kNoteStepSamples = kSampleRate * 150 / 1000;
constexpr int kChimeSamples = kSampleRate * 750 / 1000;
constexpr float kDecayS = 0.12f;
constexpr float kAttackS = 0.004f;
constexpr float kPeak = 0.30f * 32767.0f / 3.0f;  // per note; three may overlap

constexpr float kTwoPi = 6.28318531f;

constexpr int kMaxChimes = 3;
constexpr int64_t kRepeatGapMs = 4000;
// Hard stop, whatever the brain is doing meanwhile.
constexpr int64_t kMaxAlertMs = 20000;

std::mutex mu;
bool active = false;
int chimes = 0;
int64_t started_ms = 0;
int64_t next_chime_ms = 0;

void push_chime()
{
    int16_t frame[kFrameSamples];
    for (int base = 0; base < kChimeSamples; base += kFrameSamples) {
        for (int i = 0; i < kFrameSamples; ++i) {
            int n = base + i;
            float v = 0.0f;
            for (int k = 0; k < 3; ++k) {
                int since = n - k * kNoteStepSamples;
                if (since < 0) continue;
                float t = static_cast<float>(since) / kSampleRate;
                float env = std::exp(-t / kDecayS);
                if (t < kAttackS) env *= t / kAttackS;
                v += kPeak * env * std::sin(kTwoPi * kNotesHz[k] * t);
            }
            frame[i] = static_cast<int16_t>(v);
        }
        speaker_play::push(frame, kFrameSamples);
    }
}

void set_face(stackchan::avatar::Emotion e)
{
    if (commands::face_is_off()) return;
    LvglLockGuard lock;
    GetStackChan().avatar().setEmotion(e);
}

// Caller holds mu.
void finish_locked(const char* why)
{
    active = false;
    mclog::tagInfo(TAG, "alert over: {} ({} chime(s))", why, chimes);
}

// The user acknowledged it (tap / wake word). Outside mu: a socket write.
void tell_brain_dismissed()
{
    transport::send_event_json("{\"event\":\"alert_dismissed\"}");
}

}  // namespace

void start(std::string_view style)
{
    commands::wake_face();
    set_face(stackchan::avatar::Emotion::Happy);
    std::lock_guard<std::mutex> lock(mu);
    active = true;
    chimes = 0;
    started_ms = state::now_ms();
    next_chime_ms = started_ms;  // first chime on the next tick, if idle
    mclog::tagInfo(TAG, "alert: {}", std::string(style));
}

bool dismiss()
{
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!active) return false;
        finish_locked("head tap");
    }
    tell_brain_dismissed();
    return true;
}

void tick()
{
    bool chime = false;
    bool ended = false;
    bool dismissed = false;
    {
        std::lock_guard<std::mutex> lock(mu);
        if (!active) return;
        int64_t now = state::now_ms();
        state::Mode mode = state::current();
        if (mode == state::Mode::Listening) {
            // Wake word (or a follow-up window): the user is here.
            finish_locked("listening");
            dismissed = true;
        } else if (now - started_ms > kMaxAlertMs) {
            finish_locked("timed out");
            ended = true;
        } else if (mode == state::Mode::Speaking) {
            // The brain is speaking the announcement: delivered, no more
            // chimes. Repeats only run while that never happens.
            finish_locked("announced");
        } else if (now >= next_chime_ms) {
            chime = true;
            ++chimes;
            next_chime_ms = now + kRepeatGapMs;
            if (chimes >= kMaxChimes) {
                finish_locked("chimes done");
                ended = true;
            }
        }
    }
    if (dismissed) tell_brain_dismissed();
    if (chime) push_chime();
    if (ended) set_face(stackchan::avatar::Emotion::Neutral);
}

}  // namespace agent::alert
