"""EMA-smoothed gaze controller.

Given a face in normalized image coords (cx, cy ∈ [0, 1]), compute the
servo angle the head should move to so the camera centres on the face,
and send a `look_at` command. The camera is mounted to the head, so a
face offset from image center is a delta to apply to the head's current
yaw/pitch — over several iterations the head locks onto the face.

We:
  - EMA-smooth the offset across frames (α ≈ 0.25) so single noisy
    detections don't yank the head.
  - Deadzone tiny offsets so a near-centered face doesn't jitter.
  - Throttle commands to once per 300 ms (camera_pump is 1500 ms so
    this is a safety net rather than a tight loop).
  - Track the head's commanded yaw/pitch locally as the integrator;
    reset the EMA after each commanded move so we don't double-count.

GC0308 FOV is roughly 60° horizontal, 45° vertical (typical small
sensor lens). Tune EMPIRICALLY if tracking over/undershoots.
"""

from __future__ import annotations

import json
import logging
import time

from websockets.asyncio.server import ServerConnection

from vision import Face

log = logging.getLogger("brain.gaze")

# Camera FOV — empirical for GC0308 + StackChan lens.
FOV_H_DEG = 60.0
FOV_V_DEG = 45.0

# Servo limits (mirror firmware constants).
MAX_YAW_DEG = 128.0
MIN_PITCH_DEG = 3.0
MAX_PITCH_DEG = 87.0
CENTER_PITCH_DEG = 62.0   # default pitch zero position (62.0° = 620 tenths)

EMA_ALPHA = 0.25

# Offsets smaller than this (in normalized image units) don't trigger
# a move — keeps the head quiet when the face is roughly centered.
DEADZONE_NORM = 0.06

# Don't issue more than one look_at per this interval.
MIN_INTERVAL_S = 0.3


class GazeController:
    def __init__(self) -> None:
        self.head_yaw = 0.0
        self.head_pitch = CENTER_PITCH_DEG
        self.ema_off_x = 0.0
        self.ema_off_y = 0.0
        self.last_send_s = 0.0

    async def update(self, face: Face, ws: ServerConnection) -> None:
        off_x = face.cx - 0.5
        off_y = face.cy - 0.5
        self.ema_off_x = EMA_ALPHA * off_x + (1 - EMA_ALPHA) * self.ema_off_x
        self.ema_off_y = EMA_ALPHA * off_y + (1 - EMA_ALPHA) * self.ema_off_y

        if (
            abs(self.ema_off_x) < DEADZONE_NORM
            and abs(self.ema_off_y) < DEADZONE_NORM
        ):
            return

        now = time.monotonic()
        if now - self.last_send_s < MIN_INTERVAL_S:
            return

        target_yaw = self.head_yaw + self.ema_off_x * FOV_H_DEG
        # Pitch convention: larger value = head looks up. A face higher
        # in the image (smaller cy → negative off_y) means we need to
        # look up → larger pitch → subtract.
        target_pitch = self.head_pitch - self.ema_off_y * FOV_V_DEG
        target_yaw = max(-MAX_YAW_DEG, min(MAX_YAW_DEG, target_yaw))
        target_pitch = max(MIN_PITCH_DEG, min(MAX_PITCH_DEG, target_pitch))

        await ws.send(
            json.dumps(
                {
                    "cmd": "look_at",
                    "yaw_deg": target_yaw,
                    "pitch_deg": target_pitch,
                }
            )
        )
        log.debug(
            "gaze: face=(%.2f,%.2f) head=(%.1f,%.1f) → (%.1f,%.1f)",
            face.cx, face.cy, self.head_yaw, self.head_pitch,
            target_yaw, target_pitch,
        )
        self.head_yaw = target_yaw
        self.head_pitch = target_pitch
        self.ema_off_x = 0.0
        self.ema_off_y = 0.0
        self.last_send_s = now

    def notify_head_moved(self, yaw_deg: float, pitch_deg: float) -> None:
        """Sync internal pose to an externally-commanded head position
        (e.g. agent called look_at). Clears EMA so the next face
        detection is measured relative to the new pose, not the previous
        one — without this, gaze tracking immediately fights the agent's
        command back toward the prior pose."""
        self.head_yaw = yaw_deg
        self.head_pitch = pitch_deg
        self.ema_off_x = 0.0
        self.ema_off_y = 0.0
