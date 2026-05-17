#include "commands.h"

#include <string>
#include <string_view>

#include <ArduinoJson.h>
#include <hal/hal.h>
#include <mooncake_log.h>
#include <stackchan/avatar/avatar/elements/emotion.h>
#include <stackchan/stackchan.h>

#include "state.h"

namespace agent::commands {

namespace {

constexpr const char* TAG = "agent.cmd";

using stackchan::avatar::Emotion;

Emotion parse_emotion(std::string_view name)
{
    if (name == "happy") return Emotion::Happy;
    if (name == "sad") return Emotion::Sad;
    if (name == "angry") return Emotion::Angry;
    if (name == "sleepy") return Emotion::Sleepy;
    // The brain's "surprised" maps to Doubt (closest available expression).
    if (name == "surprised") return Emotion::Doubt;
    return Emotion::Neutral;
}

void apply_set_expression(JsonDocument& doc)
{
    const char* value = doc["value"] | "neutral";
    Emotion e = parse_emotion(value);
    LvglLockGuard lock;
    GetStackChan().avatar().setEmotion(e);
    mclog::tagInfo(TAG, "expression: {}", value);
}

void apply_look_at(JsonDocument& doc)
{
    // Claude speaks degrees; servos take tenths-of-degree.
    // Yaw ±128° (clamped to ±1280 tenths); pitch 3..87° (30..870 tenths).
    float yaw_deg = doc["yaw_deg"] | 0.0f;
    float pitch_deg = doc["pitch_deg"] | 30.0f;
    int yaw = static_cast<int>(yaw_deg * 10);
    int pitch = static_cast<int>(pitch_deg * 10);
    if (yaw < -1280) yaw = -1280;
    if (yaw > 1280) yaw = 1280;
    if (pitch < 30) pitch = 30;
    if (pitch > 870) pitch = 870;
    GetStackChan().motion().move(yaw, pitch);
    mclog::tagInfo(TAG, "look_at: yaw={}° pitch={}°", yaw_deg, pitch_deg);
}

void apply_set_motion_rate(JsonDocument& doc)
{
    int per_min = doc["per_minute"] | 4;
    // Phase 3 stub: IdleMotionModifier isn't installed yet — log the request
    // so brain-side tool feedback is meaningful, and wire it for real in a
    // follow-up. (The mod is in stackchan/modifiers/idle_motion.h but we
    // haven't attached it; main.cpp would need to install + retain a handle.)
    mclog::tagWarn(TAG, "set_motion_rate {} per/min — not yet implemented",
                   per_min);
}

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
    } else if (c == "set_expression") {
        apply_set_expression(doc);
    } else if (c == "look_at") {
        apply_look_at(doc);
    } else if (c == "set_motion_rate") {
        apply_set_motion_rate(doc);
    } else {
        mclog::tagWarn(TAG, "unknown cmd: {}", c);
    }
}

}  // namespace agent::commands
