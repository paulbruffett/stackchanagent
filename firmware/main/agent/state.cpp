#include "state.h"

#include <atomic>
#include <mutex>

#include <mooncake_log.h>

#include "wakeword.h"

namespace agent::state {

namespace {

constexpr const char* TAG = "agent.state";

std::atomic<Mode> mode_{Mode::Idle};

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

}  // namespace

Mode current()
{
    return mode_.load(std::memory_order_relaxed);
}

void transition(Mode next)
{
    std::lock_guard<std::mutex> lock(mu_);
    Mode prev = mode_.exchange(next, std::memory_order_acq_rel);
    if (prev == next) return;
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

}  // namespace agent::state
