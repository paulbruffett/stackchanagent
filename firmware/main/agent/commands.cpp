#include "commands.h"

#include <atomic>
#include <memory>
#include <string>
#include <string_view>

#include <ArduinoJson.h>
#include <hal/hal.h>
#include <mooncake_log.h>
#include <stackchan/avatar/avatar/elements/emotion.h>
#include <stackchan/avatar/skins/default/default.h>
#include <stackchan/avatar/skins/rocky/rocky.h>
#include <stackchan/stackchan.h>

#include "buddy_ble.h"
#include "state.h"

namespace agent::commands {

namespace {

constexpr const char* TAG = "agent.cmd";

using stackchan::avatar::Emotion;

// Sleep state. The brain's JSON task sets it on the inactivity timeout;
// the wake word / head-touch tasks clear it via wake_face(). Atomic so the
// cross-task reads/writes are well-defined.
std::atomic<bool> g_face_off{false};
std::atomic<uint8_t> g_saved_brightness{255};

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
    LvglLockGuard lock;
    if (std::string_view{value} == "celebrate") {
        // No "celebrate" Emotion; skins decide how to render it (Rocky adds
        // confetti, the default maps it to Happy).
        GetStackChan().avatar().celebrate();
    } else {
        GetStackChan().avatar().setEmotion(parse_emotion(value));
    }
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
    GetStackChan().avatar().setBusy(on);
    if (on) {
        GetStackChan().avatar().setSpeech("...");
    } else {
        GetStackChan().avatar().clearSpeech();
    }
    mclog::tagInfo(TAG, "busy: {}", on);
}

// Active skin. Boot brings up DefaultAvatar (hal/board/stackchan_display.cc
// SetupUI), so this starts at Default; the brain syncs it via set_skin on
// connect and whenever ROCKY_MODE changes.
enum class Skin { Default, Rocky };
Skin g_current_skin = Skin::Default;

// Runtime skin swap. Both skins ship in the build; ROCKY_MODE (a brain knob)
// is the single source of truth and the brain emits set_skin. The framework
// swap is pointer-safe: the Breath/Blink/HeadPet modifiers re-fetch
// avatar().leftEye()/mouth()/getEmotion() fresh every tick, so they need no
// teardown — they simply poke the new avatar next tick.
void apply_set_skin(JsonDocument& doc)
{
    const char* value = doc["value"] | "default";
    Skin want         = (std::string_view{value} == "rocky") ? Skin::Rocky : Skin::Default;
    if (want == g_current_skin) {
        return;
    }
    LvglLockGuard lock;
    auto& stackchan = GetStackChan();
    if (want == Skin::Rocky) {
        auto a = std::make_unique<stackchan::avatar::RockyAvatar>();
        a->init(lv_screen_active());
        stackchan.attachAvatar(std::move(a));
    } else {
        auto a = std::make_unique<stackchan::avatar::DefaultAvatar>();
        a->init(lv_screen_active());
        stackchan.attachAvatar(std::move(a));
    }
    g_current_skin = want;
    // The new avatar starts blank — let the BLE buddy redraw any pending
    // prompt/PIN bubble that the rebuild wiped.
    buddy_ble::notify_avatar_swapped();
    mclog::tagInfo(TAG, "skin: {}", value);
}

}  // namespace

void wake_face()
{
    if (!g_face_off.exchange(false)) return;  // wasn't asleep
    LvglLockGuard lock;
    GetHAL().setBackLightBrightness(g_saved_brightness.load());
    GetStackChan().avatar().setEmotion(Emotion::Neutral);
    mclog::tagInfo(TAG, "wake: screen on (brightness {})",
                   g_saved_brightness.load());
}

void sleep_face()
{
    if (g_face_off.exchange(true)) return;  // already asleep
    LvglLockGuard lock;
    uint8_t cur = GetHAL().getBackLightBrightness();
    if (cur > 0) g_saved_brightness.store(cur);
    GetStackChan().avatar().setEmotion(Emotion::Sleepy);
    GetHAL().setBackLightBrightness(0);
    mclog::tagInfo(TAG, "sleep: screen off (saved brightness {})",
                   g_saved_brightness.load());
}

bool face_is_off()
{
    return g_face_off.load();
}

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
        // Any visible/audible activity implies the screen must be on. The
        // firmware owns the backlight, so relight here regardless of what
        // the brain believes its sleep state to be — this keeps "screen on
        // while moving/speaking" true even if brain and firmware desync
        // (e.g. the brain restarted while the device was asleep). wake_face()
        // is a no-op when already awake, so it never clobbers a live turn.
        wake_face();
        state::transition(state::Mode::Speaking);
    } else if (c == "stop_speaking") {
        state::transition(state::Mode::Idle);
    } else if (c == "set_expression") {
        wake_face();
        apply_set_expression(doc);
    } else if (c == "look_at") {
        wake_face();
        apply_look_at(doc);
    } else if (c == "set_busy") {
        wake_face();
        apply_set_busy(doc);
    } else if (c == "set_skin") {
        apply_set_skin(doc);
    } else if (c == "sleep") {
        sleep_face();
    } else if (c == "wake") {
        wake_face();
    } else {
        mclog::tagWarn(TAG, "unknown cmd: {}", c);
    }
}

}  // namespace agent::commands
