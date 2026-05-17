/*
 * stackchan agentic firmware — main entry point.
 *
 * Built on top of m5stack/StackChan HAL (kept verbatim, minus apps/ and the
 * unused factory HAL modules). Replaces M5's mooncake/apps app-launcher layer
 * with a single agentic loop that connects to the Python brain on the LAN.
 *
 * Phase 1 (current): HAL init, render the avatar, bring up Wi-Fi, open a
 * WebSocket to the brain, and pipe mic→brain / brain→speaker for an echo
 * loopback test. No wakeword or agent logic yet.
 */
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <mooncake_log.h>
#include <hal/hal.h>
#include <stackchan/stackchan.h>
#include <board.h>
#include <display/display.h>

#include "agent/mic_pump.h"
#include "agent/speaker_play.h"
#include "agent/transport.h"

static constexpr const char* TAG = "stackchan";

// Hostname + port the brain advertises (matches brain/agent_server.py).
// On macOS where zeroconf can't bind 5353, point this at the host's own
// .local hostname (e.g. "Pauls-Mac-mini.local") for local testing.
// TODO: lift to a Kconfig setting.
static constexpr const char* BRAIN_HOST = "stackchan-brain.local";
static constexpr int BRAIN_PORT = 8765;

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

    mclog::tagInfo(TAG, "phase 1 main — bring-up check");

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

    motion.move(0, 0);         pump_for(1500);   // home
    motion.move(-300, 450);    pump_for(1500);   // look up-left
    motion.move(300, 450);     pump_for(1500);   // look up-right
    motion.move(0, 200);       pump_for(1500);   // look slightly down
    motion.move(0, 0);         pump_for(1000);   // home

    mclog::tagInfo(TAG, "bring-up ok — bringing up network");

    // Block until Wi-Fi connects (uses board's stored creds; on first boot
    // this enters WifiManager config-AP mode and the user joins the hotspot
    // to set credentials).
    GetHAL().startNetwork([](std::string_view msg) {
        mclog::tagInfo(TAG, "net: {}", msg);
    });

    // Phase 1: open the WS to the brain. Audio frames from the brain go to
    // the speaker; the mic continuously streams to the brain. With the
    // brain echoing, we should hear ourselves.
    agent::transport::set_on_audio(
        [](const int16_t* samples, size_t n) { agent::speaker_play::push(samples, n); });
    agent::transport::start(BRAIN_HOST, BRAIN_PORT);
    agent::speaker_play::start();
    agent::mic_pump::start();

    mclog::tagInfo(TAG, "phase 1 running — echo loop active");

    // Idle loop: pump stackchan (avatar blink/breath + motion spring) at 50 Hz.
    while (1) {
        {
            LvglLockGuard lock;
            GetStackChan().update();
        }
        GetHAL().feedTheDog();
        GetHAL().delay(20);
    }
}
