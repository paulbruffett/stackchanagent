#include "state.h"

#include <atomic>
#include <mutex>

#include <esp_timer.h>
#include <mooncake_log.h>

#include "wakeword.h"

namespace agent::state {

namespace {

constexpr const char* TAG = "agent.state";

std::atomic<Mode> mode_{Mode::Idle};
std::atomic<int64_t> entered_at_ms_{0};

// Serialises transition() end to end. The exchange alone is atomic but the
// wakeword pause/resume that follows is not part of it, and there is a UART
// log line in between: two tasks (WS dispatch, headtouch, audio_detection,
// agent_ws) can land their exchanges in one order and their wakeword calls in
// the other, ending at mode_ == Idle with the detector stopped. AfeWakeWord
// then swallows every Feed() and the device silently stops answering to its
// wake word.
std::mutex mu_;

const char* name(Mode m)
{
    switch (m) {
        case Mode::Idle: return "IDLE";
        case Mode::Listening: return "LISTENING";
        case Mode::Speaking: return "SPEAKING";
    }
    return "?";
}

// Caller holds mu_.
void transition_locked(Mode next)
{
    if (mode_.load(std::memory_order_acquire) == next) return;
    // Stamp before publishing the mode, so a reader that sees the new mode
    // never pairs it with the previous episode's entry time.
    entered_at_ms_.store(now_ms());
    Mode prev = mode_.exchange(next, std::memory_order_acq_rel);
    mclog::tagInfo(TAG, "{} -> {}", name(prev), name(next));

    switch (next) {
        case Mode::Idle:
            wakeword::resume();
            break;
        case Mode::Listening:
        case Mode::Speaking:
            wakeword::pause();
            break;
    }
}

}  // namespace

int64_t now_ms()
{
    return esp_timer_get_time() / 1000;
}

Mode current()
{
    return mode_.load(std::memory_order_relaxed);
}

void transition(Mode next)
{
    std::lock_guard<std::mutex> lock(mu_);
    transition_locked(next);
}

bool expire_if_stale(Mode mode, int64_t last_activity_ms, int64_t timeout_ms)
{
    std::lock_guard<std::mutex> lock(mu_);
    if (mode_.load(std::memory_order_acquire) != mode) return false;
    int64_t since = entered_at_ms_.load();
    if (last_activity_ms > since) since = last_activity_ms;
    if (now_ms() - since <= timeout_ms) return false;
    transition_locked(Mode::Idle);
    return true;
}

}  // namespace agent::state
