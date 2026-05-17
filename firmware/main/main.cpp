/*
 * stackchan agentic firmware — main entry point.
 *
 * Built on top of m5stack/StackChan HAL (kept verbatim, minus apps/ and the
 * unused factory HAL modules). Replaces M5's mooncake/apps app-launcher layer
 * with a single agentic loop that (eventually) connects to the Python brain
 * on the LAN.
 *
 * Phase 0.5 (current): HAL init, render the idle avatar, do a one-time servo
 * sweep, then sit idle while pumping the stackchan modifier/animation loop.
 * No networking, audio, or wakeword yet.
 */
#include <mooncake_log.h>
#include <hal/hal.h>
#include <stackchan/stackchan.h>
#include <board.h>
#include <display/display.h>

static constexpr const char* TAG = "stackchan";

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

    mclog::tagInfo(TAG, "phase 0.5 main — bring-up check");

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

    mclog::tagInfo(TAG, "bring-up ok — idling");

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
