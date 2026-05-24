/*
 * Speaker playback: queue incoming OP_AUDIO frames from the brain and
 * write them to the codec's speaker output. The transport's audio
 * handler pushes; a dedicated task pops + writes (so blocking codec I/O
 * doesn't stall the WebSocket task).
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace agent::speaker_play {

void start();

// Push a PCM frame (s16le mono, sample-rate-matched to codec output).
// Called from the transport's audio callback. Drops on queue full.
void push(const int16_t* samples, size_t sample_count);

}  // namespace agent::speaker_play
