/*
 * Camera pump: periodically grab a frame from the StackChan camera,
 * JPEG-encode it, and ship it to the brain over the WebSocket with
 * opcode OP_JPEG. Used by brain-side vision (face detection / gaze) and
 * the describe_view tool.
 *
 * Skipped while state == SPEAKING (frees CPU + bandwidth while TTS plays)
 * and while the transport is down.
 */
#pragma once

namespace agent::camera_pump {

void start();

}  // namespace agent::camera_pump
