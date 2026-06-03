/*
 * stackchan agentic firmware — main entry point.
 *
 * Built on top of m5stack/StackChan HAL (kept verbatim, minus apps/ and the
 * unused factory HAL modules). Replaces M5's mooncake/apps app-launcher layer
 * with a single agentic loop that connects to the Python brain on the LAN.
 *
 * Through Phase 3: HAL init → avatar → servo sweep → Wi-Fi → WebSocket →
 * wakeword arm → state machine running. Brain handles wakeword event →
 * mic stream → STT → Claude tool-use turn → TTS playback. Tools the brain
 * can call: set_expression, look_at (set_motion_rate not yet wired
 * firmware-side; IdleMotionModifier install is Phase 6).
 */
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include <mooncake_log.h>
#include <hal/hal.h>
#include <stackchan/stackchan.h>
#include <board.h>
#include <display/display.h>

#include "agent/camera_pump.h"
#include "agent/commands.h"
#include "agent/mic_pump.h"
#include "agent/speaker_play.h"
#include "agent/state.h"
#include "agent/transport.h"
#include "agent/wakeword.h"

static constexpr const char* TAG = "stackchan";

// Brain hostname + port come from Kconfig — change via
// `idf.py menuconfig` → "Stackchan Brain" → "Brain host".
static constexpr const char* BRAIN_HOST = CONFIG_BRAIN_HOST;
static constexpr int BRAIN_PORT = CONFIG_BRAIN_PORT;

extern "C" void app_main(void)
{
    mclog::set_level(mclog::level_info);
    mclog::set_time_format(mclog::time_format_unix_milliseconds);

    // Init everything — display, LVGL, audio codec, mic/speaker, servos,
    // IMU, RTC, IOE, head touch. Heavy but it's M5's proven path.
    GetHAL().init();

    // Create the avatar. xiaozhi normally calls SetupUI() from
    // Application::Start(); we don't call Application, so do it ourselves.
    // StackChanAvatarDisplay::SetupUI() instantiates DefaultAvatar, attaches
    // it to GetStackChan(), and registers the Breath/Blink/HeadPet/ImuEvent
    // modifiers.
    Board::GetInstance().GetDisplay()->SetupUI();

    // HAL's init puts a "STACKCHAN / Starting up..." BootLogo overlay on
    // screen and expects the launcher app to dismiss it. Drop it now so the
    // avatar is visible.
    {
        GetHAL().lvglLock();
        GetHAL().bootLogo.reset();
        GetHAL().lvglUnlock();
    }

    mclog::tagInfo(TAG, "bring-up check");

    // Visible servo sweep. Angles are tenths of a degree (yaw limit ±1280
    // = ±128°, pitch limit 30..870 = 3..87°).
    //
    // GetStackChan().update() pumps modifiers, avatar animation, and the
    // servo spring under one call. We need to hold the LVGL lock around it
    // because avatar/modifier code mutates LVGL objects.
    auto& motion = GetStackChan().motion();
    motion.setTorqueEnabled(true);

    auto pump_for = [](uint32_t ms) {
        uint32_t deadline = GetHAL().millis() + ms;
        while (GetHAL().millis() < deadline) {
            {
                LvglLockGuard lock;
                GetStackChan().update();
            }
            GetHAL().delay(20);
        }
    };

    motion.move(0, 0);         pump_for(1500);   // home (pitch clamps to 3°, max down)
    motion.move(-300, 450);    pump_for(1500);   // look up-left
    motion.move(300, 450);     pump_for(1500);   // look up-right
    motion.move(0, 200);       pump_for(1500);   // resting pose: pitch=20° (slightly up)
    motion.move(0, 200);       pump_for(1000);   // settle in resting pose

    mclog::tagInfo(TAG, "bring-up ok — bringing up network");

    // Block until Wi-Fi connects (uses board's stored creds; on first boot
    // this enters WifiManager config-AP mode and the user joins the hotspot
    // to set credentials).
    GetHAL().startNetwork([](std::string_view msg) {
        mclog::tagInfo(TAG, "net: {}", msg);
    });

    // Wakeword → LISTENING → STT → SPEAKING → TTS playback. JSON commands
    // from the brain land in commands::dispatch (set_expression, look_at,
    // start/stop_speaking, stop_listening).
    agent::transport::set_on_audio(
        [](const int16_t* samples, size_t n) { agent::speaker_play::push(samples, n); });
    agent::transport::set_on_json(
        [](std::string_view json) { agent::commands::dispatch(json); });
    agent::transport::start(BRAIN_HOST, BRAIN_PORT);
    agent::speaker_play::start();
    agent::wakeword::on_detected([](const std::string& w) {
        // Relight the screen first if we were asleep (instant, local —
        // doesn't wait on the brain round-trip).
        agent::commands::wake_face();
        agent::transport::send_event_json(
            std::string("{\"event\":\"wakeword\",\"word\":\"") + w + "\"}");
        agent::state::transition(agent::state::Mode::Listening);
    });
    agent::wakeword::start();
    agent::mic_pump::start();
    agent::camera_pump::start();

    // Head tap → talk. The capacitive head sensor emits HeadPetGesture::Press
    // on a touch-down (the avatar's HeadPet modifier only reacts to swipes, so
    // Press is unused). Treat a debounced Press while idle exactly like the
    // wake word: relight the screen if asleep, notify the brain, and start
    // listening. Gated to IDLE so a touch can't interrupt an active turn.
    //
    // Every gesture is logged (the sensor only emits on actual touch, so this
    // isn't spammy) to make tap detection visible in the serial monitor.
    GetHAL().onHeadPetGesture.connect([](HeadPetGesture gesture) {
        const char* gname =
            gesture == HeadPetGesture::Press ? "Press" :
            gesture == HeadPetGesture::Release ? "Release" :
            gesture == HeadPetGesture::SwipeForward ? "SwipeForward" :
            gesture == HeadPetGesture::SwipeBackward ? "SwipeBackward" : "None";
        mclog::tagInfo(TAG, "head gesture: {}", gname);
        if (gesture != HeadPetGesture::Press) return;
        if (agent::state::current() != agent::state::Mode::Idle) {
            mclog::tagInfo(TAG, "tap ignored (not idle)");
            return;
        }
        static uint32_t last_tap_ms = 0;
        uint32_t now = GetHAL().millis();
        if (now - last_tap_ms < 800) {  // debounce repeated touches
            mclog::tagInfo(TAG, "tap ignored (debounce)");
            return;
        }
        last_tap_ms = now;
        agent::commands::wake_face();
        agent::transport::send_event_json("{\"event\":\"tap\"}");
        agent::state::transition(agent::state::Mode::Listening);
        mclog::tagInfo(TAG, "head tap → listening");
    });

    mclog::tagInfo(TAG, "running — listening for wakeword");

    // Brain-offline badge: small red "OFFLINE" label in the top-right
    // corner, hidden when the WebSocket is connected. Added after the
    // avatar's SetupUI so it sits above in LVGL Z order. Visibility is
    // polled from the idle loop and only toggled on state change.
    std::unique_ptr<uitk::lvgl_cpp::Label> offline_badge;
    {
        LvglLockGuard lock;
        offline_badge = std::make_unique<uitk::lvgl_cpp::Label>(lv_screen_active());
        offline_badge->setTextFont(&lv_font_montserrat_14);
        offline_badge->setTextColor(lv_color_hex(0xFF4040));
        offline_badge->setText("OFFLINE");
        offline_badge->align(LV_ALIGN_TOP_RIGHT, -4, 4);
        offline_badge->setHidden(true);
    }
    bool last_offline_state = false;

    // Idle loop: pump stackchan (avatar blink/breath + motion spring) at 50 Hz.
    // Also poll the actual servo position every ~500 ms; log any change > 1°
    // on either axis. Helps spot modifier-driven moves (HeadPet, ImuEvent)
    // that don't go through agent.cmd::apply_look_at and so wouldn't
    // otherwise appear in the log.
    int last_logged_yaw = 0;
    int last_logged_pitch = 0;
    uint32_t next_pose_log_ms = 0;
    while (1) {
        {
            LvglLockGuard lock;
            GetStackChan().update();
        }
        uint32_t now = GetHAL().millis();
        if (now >= next_pose_log_ms) {
            next_pose_log_ms = now + 500;
            auto angles = GetStackChan().motion().getCurrentAngles();
            int yaw_t = angles.x;
            int pitch_t = angles.y;
            if (std::abs(yaw_t - last_logged_yaw) > 10
                || std::abs(pitch_t - last_logged_pitch) > 10) {
                mclog::tagInfo(TAG, "pose: yaw={}° pitch={}°",
                               yaw_t / 10.0f, pitch_t / 10.0f);
                last_logged_yaw = yaw_t;
                last_logged_pitch = pitch_t;
            }
            bool offline = !agent::transport::is_connected();
            if (offline != last_offline_state) {
                LvglLockGuard lock;
                offline_badge->setHidden(!offline);
                last_offline_state = offline;
                mclog::tagInfo(TAG, "brain link: {}",
                               offline ? "OFFLINE" : "online");
            }
        }
        GetHAL().feedTheDog();
        GetHAL().delay(20);
    }
}
