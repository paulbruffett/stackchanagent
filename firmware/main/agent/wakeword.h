/*
 * Wakeword detection. Wraps xiaozhi's AfeWakeWord, which loads the
 * wakenet model selected via CONFIG_SR_WN_* (currently "Hi, Stack Chan"
 * - wn9_histackchan_tts3). Feed it raw mic PCM via wakeword::feed();
 * detection fires the callback set with wakeword::on_detected().
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace agent::wakeword {

// Initialize the wakenet, start the detection task, and arm detection.
void start();

// Pause detection (e.g. while LISTENING / SPEAKING — don't want to wake
// ourselves up on TTS playback or on the user's own speech).
void pause();
void resume();

// Push 16 kHz mono PCM frames in. Safe to call from the mic task.
void feed(const std::vector<int16_t>& pcm);

using DetectedCallback = std::function<void(const std::string& word)>;
void on_detected(DetectedCallback cb);

}  // namespace agent::wakeword
