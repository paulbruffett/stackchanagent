#include "state.h"

#include <atomic>

#include <mooncake_log.h>

#include "wakeword.h"

namespace agent::state {

namespace {

constexpr const char* TAG = "agent.state";

std::atomic<Mode> mode_{Mode::Idle};

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
