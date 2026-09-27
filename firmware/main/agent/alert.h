/*
 * Attention alert for a brain timer/reminder ({"cmd":"alert","style":"timer"}).
 *
 * start() relights the screen, puts on a happy face and plays a short
 * synthesized chime (three rising notes, generated as 16 kHz PCM and pushed
 * through speaker_play — no decoder needed). The chime repeats up to
 * kMaxChimes times, ~4 s apart, only while IDLE and only until the brain
 * speaks the announcement: entering SPEAKING counts as delivered and ends the
 * alert, so repeats happen only if the brain is busy or gone. It also ends
 * when the chimes run out (or after 20 s), on a head tap (dismiss(), which consumes the tap),
 * or when the device starts LISTENING (the wake word). A tap or wake word
 * tells the brain {"event":"alert_dismissed"}.
 *
 * All calls are from the main idle loop except dismiss(), which the head-tap
 * handler calls; state is guarded by a mutex.
 */
#pragma once

#include <string_view>

namespace agent::alert {

// Begin (or restart) an alert. Call from commands::dispatch, i.e. the idle
// loop, outside the LVGL lock.
void start(std::string_view style);

// Drive repeats and the end conditions. Call every idle-loop iteration,
// outside the LVGL lock.
void tick();

// Head tap: end an active alert. True if there was one (the tap is consumed).
bool dismiss();

}  // namespace agent::alert
