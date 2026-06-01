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

// Spring speed for agent-initiated look_at. The motion library maps
// speed → stiffness via k = 10 + (speed/1000)^2 * 640. 500 ≈ stock M5
// default (snappy, k=170), 300 is gentler (k≈68), 250 is moderately
// gentle (k≈50), 200 is the slowest that still tracks (k≈36).
// Lower numbers smooth out the visible "snap" on small gaze corrections.
static constexpr int kLookAtSpeed = 200;

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
    // Optional per-command spring speed (0..1000). Absent → kLookAtSpeed
    // (the gentle default used for face-centering and agent look_at). The
    // brain sends a higher speed for look-around sweep poses.
    int speed = doc["speed"] | kLookAtSpeed;
    if (speed < 0) speed = 0;
    if (speed > 1000) speed = 1000;
    GetStackChan().motion().moveWithSpeed(yaw, pitch, speed);
    mclog::tagInfo(TAG, "look_at: yaw={}° pitch={}° speed={}", yaw_deg, pitch_deg, speed);
}

// On-screen "thinking" indicator the brain raises while a slow tool call
// runs and clears when the reply starts. Uses the avatar's speech bubble
// (otherwise unused in the agent flow) so it doesn't clobber whatever
// emotion the agent set via set_expression.
void apply_set_busy(JsonDocument& doc)
{
    bool on = doc["on"] | false;
    LvglLockGuard lock;
    if (on) {
        GetStackChan().avatar().setSpeech("...");
    } else {
        GetStackChan().avatar().clearSpeech();
    }
    mclog::tagInfo(TAG, "busy: {}", on);
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
    } else if (c == "start_listening") {
        // Brain-initiated listening: follow-up window after a reply
        // (no wakeword needed for the second utterance). Transition
        // pauses the wakeword so it doesn't double-fire on speech.
        state::transition(state::Mode::Listening);
    } else if (c == "start_speaking") {
        state::transition(state::Mode::Speaking);
    } else if (c == "stop_speaking") {
        state::transition(state::Mode::Idle);
    } else if (c == "set_expression") {
        apply_set_expression(doc);
    } else if (c == "look_at") {
        apply_look_at(doc);
    } else if (c == "set_busy") {
        apply_set_busy(doc);
    } else {
        mclog::tagWarn(TAG, "unknown cmd: {}", c);
    }
}

}  // namespace agent::commands
