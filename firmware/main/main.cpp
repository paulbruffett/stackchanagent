/*
 * stackchan agentic firmware — main entry point.
 *
 * Built on top of m5stack/StackChan HAL (kept verbatim). Replaces M5's
 * mooncake/apps app-launcher layer with a single agentic loop that
 * (eventually) connects to the Python brain on the LAN.
 *
 * Phase 0: just initialise the HAL and prove servo motion works under
 * our own main. No display rendering, no networking, no audio yet.
 */
#include <mooncake_log.h>
#include <hal/hal.h>
#include <stackchan/stackchan.h>

static constexpr const char* TAG = "stackchan";

extern "C" void app_main(void)
{
    mclog::set_level(mclog::level_info);
    mclog::set_time_format(mclog::time_format_unix_milliseconds);

    // Init everything — display, LVGL, audio codec, mic/speaker, servos,
    // IMU, RTC, IOE, head touch. Heavy but it's M5's proven path.
    GetHAL().init();

    // HAL's init puts a "STACKCHAN / Starting up..." BootLogo overlay
    // on screen and expects the launcher app to dismiss it. We don't
    // install any apps, so clear it ourselves — otherwise the splash
    // hides everything (the servos run regardless, but invisibly).
    {
        GetHAL().lvglLock();
        GetHAL().bootLogo.reset();
        GetHAL().lvglUnlock();
    }

    mclog::tagInfo(TAG, "phase 0 custom main — bring-up check");

    // Visible servo sweep. Angles are tenths of a degree (yaw limit ±1280
    // = ±128°, pitch limit 30..870 = 3..87°).
    //
    // motion.move() only sets a spring-animation TARGET — motion.update()
    // is what actually ticks the animation and writes goal positions to
    // the SCS bus. M5's firmware pumps this from a background task that
    // only starts inside startXiaozhi(); since we're not starting xiaozhi
    // we pump it ourselves at 50 Hz.
    auto& motion = GetStackChan().motion();
    motion.setTorqueEnabled(true);

    auto pump_for = [&motion](uint32_t ms) {
        uint32_t deadline = GetHAL().millis() + ms;
        while (GetHAL().millis() < deadline) {
            motion.update();
            GetHAL().delay(20);
        }
    };

    motion.move(0, 0);         pump_for(1500);   // home
    motion.move(-300, 450);    pump_for(1500);   // look up-left
    motion.move(300, 450);     pump_for(1500);   // look up-right
    motion.move(0, 200);       pump_for(1500);   // look slightly down
    motion.move(0, 0);         pump_for(1000);   // home

    mclog::tagInfo(TAG, "bring-up ok — idling (no apps installed)");

    // Keep pumping motion + feeding the watchdog forever.
    while (1) {
        motion.update();
        GetHAL().feedTheDog();
        GetHAL().delay(20);
    }
}
