#include "commands.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <ArduinoJson.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <hal/board/hal_bridge.h>
#include <hal/hal.h>
#include <mooncake_log.h>
#include <stackchan/animation/animation.h>
#include <stackchan/avatar/avatar/elements/emotion.h>
#include <stackchan/modifiers/dance.h>
#include <stackchan/stackchan.h>

#include "buddy_ble.h"
#include "ota.h"
#include "state.h"
#include "transport.h"

namespace agent::commands {

namespace {

constexpr const char* TAG = "agent.cmd";

using stackchan::avatar::Emotion;

// Sleep state. The brain's JSON task sets it on the inactivity timeout;
// the wake word / head-touch tasks clear it via wake_face(). Atomic so the
// cross-task reads/writes are well-defined.
std::atomic<bool> g_face_off{false};
std::atomic<uint8_t> g_saved_brightness{255};

// The atomic makes each flag flip well-defined, but not the flag-plus-backlight
// pair. sleep_face() runs on the WS dispatch path while wake_face() runs from
// the head-touch and wakeword tasks, and both take the (contended) LVGL lock
// only *after* their exchange — so sleep can win the flag and wake can win the
// lock, leaving g_face_off false with the backlight at 0. After that every
// wake_face() early-returns and no tap or wake word can relight the screen.
// Serialise the whole check-and-act. Deliberately not the LVGL lock itself:
// dispatch() calls wake_face() for every activity command and its common
// no-op path must not queue behind a 20 ms avatar update.
std::mutex g_face_mu;

// Brain commands are parsed on the esp-ml307 "tcp_receive" task, but they must
// not be *executed* there: apply_look_at ends in ScsServo::getCurrentAngle() /
// set_angle_impl(), i.e. request/response transactions on the SCS UART bus
// that the main loop is already driving at 50 Hz from Servo::update(). A
// ReadPos yields for tens of ms mid-transaction (SCSerial::readSCS delays per
// byte poll), so a WritePos frame interleaves, both sides fail their
// checksums, hal_servo's bad-read guard substitutes the stale angle and the
// spring teleports — a visible head jump plus a dropped pose. The same two
// tasks were also racing on _angle_anim, which no bus lock would cover. So
// queue the raw frame here and let the idle loop run it: one task owns
// StackChan. It also keeps set_buddy's NVS write + restart off tcp_receive.
constexpr size_t kMaxQueuedCommands = 16;

std::mutex g_queue_mu;
std::deque<std::string> g_queue;

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
        // No "celebrate" Emotion; the avatar decides how to render it (the
        // default maps it to Happy).
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

// Turn watchdog. Only the brain ends a LISTENING or SPEAKING turn, and the
// wakeword is paused and the head tap gated to Idle meanwhile. transport
// already drops to Idle when the link goes down, but a brain that is connected
// yet wedged would leave the robot deaf. Both clocks run from the later of
// entering the mode and the last inbound brain frame.
//
// LISTENING: the brain's captures are capped below this — MAX_UTTERANCE_MS
// max 12000 and FOLLOW_UP_WINDOW_S max 10 (brain/config.py); the follow-up
// window is opened by start_listening, itself a frame. Keep those caps below
// this value if either changes.
constexpr int64_t kListeningWatchdogMs = 15000;
// SPEAKING: a slow MCP tool (45 s dispatch timeout) can legitimately hold the
// device in SPEAKING with no frames after the spoken ack filler.
constexpr int64_t kSpeakingWatchdogMs = 60000;

void apply_set_buddy(JsonDocument& doc)
{
    if (!doc["enabled"].is<bool>()) {
        mclog::tagWarn(TAG, "set_buddy: missing/non-bool enabled");
        return;
    }
    // May not return: restarts the device when the setting changes.
    buddy_ble::set_enabled(doc["enabled"].as<bool>());
}

// Speaker volume as last applied, 0..100; -1 until first read. Tracked here
// rather than re-read with getSpeakerVolume(): that one maps a stored 0 to 10
// (xiaozhi's boot floor), so a brain that set 0 would see 10 reported and keep
// re-sending. Written on the main task, read from the transport task (boot)
// and the status task.
std::atomic<int> g_volume{-1};

int current_volume()
{
    int v = g_volume.load();
    if (v < 0) {
        v = GetHAL().getSpeakerVolume();
        g_volume.store(v);
    }
    return v;
}

// "battery":..,"charging":..,"volume":.. from values already read.
std::string format_status(bool read_ok, int pct, bool charging)
{
    std::string out = "\"battery\":";
    // The AXP2101 fuel gauge reads 0..100; anything else means no usable
    // reading (no battery, gauge not ready), reported as unknown.
    if (read_ok && pct >= 0 && pct <= 100) {
        out += std::to_string(pct);
        out += charging ? ",\"charging\":true" : ",\"charging\":false";
    } else {
        out += "null,\"charging\":null";
    }
    out += ",\"volume\":" + std::to_string(current_volume());
    return out;
}

// --- status reporter --------------------------------------------------------
//
// Its own low-priority task, so neither the PMIC's I2C reads nor a WebSocket
// send that stalls on a full TCP window ever holds up the main loop (servos,
// avatar, command drain). It polls only the charging bit every 2 s and reads
// the level only when it actually reports: every 60 s, on a charging flip, or
// when asked (request_status, after set_volume).
constexpr uint32_t kStatusPollMs   = 2000;
constexpr int64_t kStatusReportMs  = 60000;
TaskHandle_t g_status_task         = nullptr;

void status_task(void*)
{
    int last_charging      = -1;  // -1 unknown, else 0/1
    int64_t next_report_ms = 0;
    while (true) {
        bool requested = ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(kStatusPollMs)) > 0;
        bool charging  = false;
        int ch         = GetHAL().readBatteryCharging(charging) ? (charging ? 1 : 0) : -1;
        bool flipped   = last_charging != -1 && ch != -1 && ch != last_charging;
        if (ch != -1) last_charging = ch;
        if (flipped) mclog::tagInfo(TAG, "charging: {}", ch == 1);
        int64_t now = state::now_ms();
        if (!requested && !flipped && now < next_report_ms) continue;
        // Not connected: the boot event carries it on (re)connect.
        if (!transport::is_connected()) continue;
        next_report_ms = now + kStatusReportMs;
        int pct = 0;
        bool ok = GetHAL().readBatteryStatus(pct, charging);
        transport::send_event_json("{\"event\":\"status\"," + format_status(ok, pct, charging) + "}");
    }
}

void request_status()
{
    if (g_status_task) xTaskNotifyGive(g_status_task);
}

void apply_set_volume(JsonDocument& doc)
{
    if (!doc["value"].is<int>()) {
        mclog::tagWarn(TAG, "set_volume: missing/non-int value");
        return;
    }
    int v = std::clamp(doc["value"].as<int>(), 0, 100);
    // permanent: AudioCodec::SetOutputVolume writes NVS "audio"/"output_volume",
    // which the codec reads back at boot.
    GetHAL().setSpeakerVolume(static_cast<uint8_t>(v), true);
    g_volume.store(v);
    mclog::tagInfo(TAG, "volume: {}", v);
    // Confirm at once (from the status task), so the brain's reported value
    // is corrected if we clamped, instead of 60 s later.
    request_status();
}

// --- dance ------------------------------------------------------------------
//
// Plays one of DanceModifier's stock keyframe sequences with our own bounded
// modifier instead of DanceModifier itself: that one destroys itself when its
// timeline ends, and a modifier id is reused as soon as it is freed, so the
// caller could never tell "still dancing" from "some other modifier now has
// that id".
//
// Only the fields this build wants are read from the stock keyframes, into
// DanceStep:
//  - Servos: pitch offset by the rest pose (the sequences were written around
//    M5's home (0, 0); here pitch 0 is 3°, chin on the desk, and rest is 20°),
//    and spring speed capped at kDanceMaxSpeed — Robot (800) and Panic (1000,
//    reversing every 100 ms) otherwise command the snap moves 7d761aa/d976ad9
//    worked to get rid of; capped, Panic reads as a fast wobble, not a jerk.
//  - Features: weight, rotation and x only. Not size: FeatureKeyframe leaves
//    it uninitialised, and no size value maps to the skin's default eye size
//    (16 px; size 0 is 20 px), so it is left alone rather than reset. Not y:
//    BreathModifier moves the features' y by deltas from its own running
//    offset, and an absolute y write would silently shift that baseline (and
//    drift a little further every dance). x is ours alone; every stock
//    sequence ends at x = 0, and finish() puts it there too.
constexpr int kRestYaw         = 0;
constexpr int kRestPitch       = 200;  // tenths of a degree; matches main.cpp's resting pose
constexpr int kDanceMaxSpeed   = 400;  // k ≈ 112, vs 200 (k ≈ 36) for look_at
constexpr uint32_t kDanceMaxMs = 6000;

bool g_dancing = false;  // main task only (dispatch + StackChan::update)

struct FeatureStep {
    int x, rotation, weight;
};

struct DanceStep {
    FeatureStep left_eye, right_eye, mouth;
    int yaw, yaw_speed, pitch, pitch_speed;
    uint32_t duration_ms;
};

FeatureStep feature_step(const stackchan::animation::FeatureKeyframe& k)
{
    return {k.position.x, k.rotation, k.weight};
}

DanceStep dance_step(const stackchan::animation::Keyframe& kf)
{
    return {feature_step(kf.leftEye), feature_step(kf.rightEye), feature_step(kf.mouth),
            kf.yawServo.angle, std::min(kf.yawServo.speed, kDanceMaxSpeed),
            kf.pitchServo.angle + kRestPitch, std::min(kf.pitchServo.speed, kDanceMaxSpeed),
            kf.durationMs};
}

void set_feature_x(stackchan::avatar::Feature& f, int x)
{
    auto pos = f.getPosition();
    pos.x    = x;
    f.setPosition(pos);
}

class BoundedDance : public stackchan::Modifier {
public:
    // Applies the first step at once: construct on the main task, LVGL lock held.
    explicit BoundedDance(std::vector<DanceStep> steps)
        : _steps(std::move(steps)), _deadline(GetHAL().millis() + kDanceMaxMs)
    {
        _step_at = GetHAL().millis();
        apply(GetStackChan(), _steps[0]);
    }

    void _update(stackchan::Modifiable& sc) override
    {
        if (isDestroyRequested()) return;
        uint32_t now = GetHAL().millis();
        if (static_cast<int32_t>(now - _deadline) >= 0) {
            finish(sc);
            return;
        }
        if (now - _step_at < _steps[_index].duration_ms) return;
        if (++_index >= _steps.size()) {
            finish(sc);
            return;
        }
        _step_at = now;
        apply(sc, _steps[_index]);
    }

private:
    static void apply_feature(stackchan::avatar::Feature& f, const FeatureStep& s)
    {
        set_feature_x(f, s.x);
        f.setRotation(s.rotation);
        f.setWeight(s.weight);
    }

    static void apply(stackchan::Modifiable& sc, const DanceStep& s)
    {
        if (sc.hasAvatar()) {
            auto& avatar = sc.avatar();
            apply_feature(avatar.leftEye(), s.left_eye);
            apply_feature(avatar.rightEye(), s.right_eye);
            apply_feature(avatar.mouth(), s.mouth);
        }
        sc.motion().yawServo().moveWithSpeed(s.yaw, s.yaw_speed);
        sc.motion().pitchServo().moveWithSpeed(s.pitch, s.pitch_speed);
    }

    void finish(stackchan::Modifiable& sc)
    {
        // Home at the gentle look_at speed; put the face back to whatever
        // emotion was set (it restores the eye weights/rotations the steps
        // wrote), then re-sync the blink modifier's base weights exactly as
        // StackChanAvatarDisplay::SetEmotion does.
        sc.motion().moveWithSpeed(kRestYaw, kRestPitch, kLookAtSpeed);
        if (sc.hasAvatar()) {
            auto& avatar = sc.avatar();
            set_feature_x(avatar.leftEye(), 0);
            set_feature_x(avatar.rightEye(), 0);
            set_feature_x(avatar.mouth(), 0);
            avatar.setEmotion(avatar.getEmotion());
            hal_bridge::display_resync_blink();
        }
        g_dancing = false;
        requestDestroy();
        mclog::tagInfo(TAG, "dance: done");
    }

    std::vector<DanceStep> _steps;
    size_t _index = 0;
    uint32_t _step_at = 0;
    uint32_t _deadline;
};

void apply_dance(JsonDocument& doc)
{
    std::string_view style = doc["style"] | "happy";
    if (g_dancing) {
        mclog::tagInfo(TAG, "dance: {} ignored, already dancing", style);
        return;
    }
    const stackchan::animation::KeyframeSequence* src = &stackchan::DanceModifier::Happy;
    if (style == "robot") {
        src = &stackchan::DanceModifier::Robot;
    } else if (style == "panic") {
        src = &stackchan::DanceModifier::Panic;
    } else if (style != "happy") {
        mclog::tagWarn(TAG, "dance: unknown style {}, dancing happy", style);
        style = "happy";
    }
    std::vector<DanceStep> steps;
    steps.reserve(src->size());
    for (const auto& kf : *src) steps.push_back(dance_step(kf));
    if (steps.empty()) return;
    LvglLockGuard lock;
    GetStackChan().addModifier(std::make_unique<BoundedDance>(std::move(steps)));
    g_dancing = true;
    mclog::tagInfo(TAG, "dance: {}", style);
}

}  // namespace

std::string status_fields()
{
    int pct       = 0;
    bool charging = false;
    bool ok       = GetHAL().readBatteryStatus(pct, charging);
    return format_status(ok, pct, charging);
}

void start_status_reporter()
{
    if (g_status_task) return;
    xTaskCreatePinnedToCore(status_task, "agent_status", 4096, nullptr, 1, &g_status_task, 0);
}

void wake_face()
{
    std::lock_guard<std::mutex> face_lock(g_face_mu);
    if (!g_face_off.exchange(false)) return;  // wasn't asleep
    LvglLockGuard lock;
    GetHAL().setBackLightBrightness(g_saved_brightness.load());
    GetStackChan().avatar().setEmotion(Emotion::Neutral);
    mclog::tagInfo(TAG, "wake: screen on (brightness {})",
                   g_saved_brightness.load());
}

void sleep_face()
{
    std::lock_guard<std::mutex> face_lock(g_face_mu);
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

void enqueue(std::string_view json)
{
    std::lock_guard<std::mutex> lock(g_queue_mu);
    if (g_queue.size() >= kMaxQueuedCommands) {
        // The idle loop drains everything every ~20 ms, so this only fires if
        // it has stopped running — worth a line in the log either way.
        mclog::tagWarn(TAG, "command queue full; dropping {}", g_queue.front());
        g_queue.pop_front();
    }
    g_queue.emplace_back(json);
}

void drain()
{
    while (true) {
        std::string json;
        {
            std::lock_guard<std::mutex> lock(g_queue_mu);
            if (g_queue.empty()) return;
            json = std::move(g_queue.front());
            g_queue.pop_front();
        }
        // Outside g_queue_mu: dispatch takes the LVGL lock and can block.
        dispatch(json);
    }
}

void check_turn_watchdog()
{
    // Same Idle transition as the brain's stop_listening / stop_speaking,
    // then tell the brain so it drops its side of the turn too.
    int64_t last_rx = transport::last_rx_ms();
    if (state::expire_if_stale(state::Mode::Listening, last_rx, kListeningWatchdogMs)) {
        mclog::tagWarn(TAG, "LISTENING > {} ms with no brain frame; back to idle",
                       kListeningWatchdogMs);
        transport::send_event_json("{\"event\":\"listen_timeout\"}");
    } else if (state::expire_if_stale(state::Mode::Speaking, last_rx, kSpeakingWatchdogMs)) {
        mclog::tagWarn(TAG, "SPEAKING > {} ms with no brain frame; back to idle",
                       kSpeakingWatchdogMs);
        transport::send_event_json("{\"event\":\"speak_timeout\"}");
    }
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
    } else if (c == "set_buddy") {
        apply_set_buddy(doc);
    } else if (c == "set_volume") {
        apply_set_volume(doc);
    } else if (c == "dance") {
        wake_face();
        apply_dance(doc);
    } else if (c == "ota") {
        // Only kicks off the download task; it never blocks this loop.
        ota::start(doc["url"] | "", doc["size"] | static_cast<size_t>(0), doc["sha256"] | "");
    } else if (c == "sleep") {
        sleep_face();
    } else if (c == "wake") {
        wake_face();
    } else {
        mclog::tagWarn(TAG, "unknown cmd: {}", c);
    }
}

}  // namespace agent::commands
