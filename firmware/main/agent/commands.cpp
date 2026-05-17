#include "commands.h"

#include <string>

#include <ArduinoJson.h>
#include <mooncake_log.h>

#include "state.h"

namespace agent::commands {

namespace {

constexpr const char* TAG = "agent.cmd";

}  // namespace

void dispatch(std::string_view json)
{
    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, json.data(), json.size());
    if (err) {
        mclog::tagWarn(TAG, "bad json: {}", err.c_str());
        return;
    }
    const char* cmd = doc["cmd"] | static_cast<const char*>(nullptr);
    if (!cmd) {
        mclog::tagWarn(TAG, "no cmd field: {}", std::string(json));
        return;
    }
    std::string_view c{cmd};
    if (c == "stop_listening") {
        state::transition(state::Mode::Idle);
    } else if (c == "start_speaking") {
        state::transition(state::Mode::Speaking);
    } else if (c == "stop_speaking") {
        state::transition(state::Mode::Idle);
    } else {
        mclog::tagWarn(TAG, "unknown cmd: {}", c);
    }
}

}  // namespace agent::commands
