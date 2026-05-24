/*
 * Agent transport: a single WebSocket to the Jetson brain.
 *
 * Wire protocol (matches brain/agent_server.py):
 *   - Binary frame, first byte = opcode:
 *       0x01  PCM audio (s16le, mono, codec native sample rate).
 *       0x02  JPEG camera frame.
 *   - Text frame: JSON for events (ESP→brain) and commands (brain→ESP).
 *
 * Connection lifecycle is managed by a background task: connect → run
 * until disconnect → exponential backoff → retry. Callers don't need to
 * know whether the link is up; sends on a dead link just return false.
 */
#pragma once
#include <cstdint>
#include <cstddef>
#include <functional>
#include <string_view>

namespace agent::transport {

constexpr uint8_t OP_AUDIO = 0x01;
constexpr uint8_t OP_JPEG  = 0x02;

// Spin up the connection-manager task. Must be called after Wi-Fi is up.
// Caller retains ownership of host (must outlive this call's copy).
void start(const char* host, int port);

bool is_connected();

// Send a 16 kHz / codec-native-rate s16le mono PCM frame prefixed with
// OP_AUDIO. Returns false if the link is down (frame is dropped, no queue).
bool send_audio(const int16_t* samples, size_t sample_count);

// Send a JPEG frame prefixed with OP_JPEG. Returns false if the link is
// down (frame is dropped, no queue).
bool send_jpeg(const uint8_t* jpeg, size_t len);

// Send a JSON text frame. Pass a complete JSON string.
bool send_event_json(std::string_view json);

// Register handler for incoming OP_AUDIO frames. The pointer is into
// the WebSocket library's internal buffer and is only valid for the
// duration of the call — copy if you need to retain.
using AudioFrameHandler =
    std::function<void(const int16_t* samples, size_t sample_count)>;
void set_on_audio(AudioFrameHandler handler);

// Register handler for incoming JSON text frames.
using JsonFrameHandler = std::function<void(std::string_view json)>;
void set_on_json(JsonFrameHandler handler);

}  // namespace agent::transport
