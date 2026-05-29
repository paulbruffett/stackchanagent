"""Idle head behavior: periodic look-around + one-shot centering.

Replaces the continuous EMA face-tracking from gaze.py with a much
simpler model:

  - Default: head stays at REST_POSE.
  - Every LOOK_AROUND_INTERVAL_S, run a short multi-pose sweep (only
    if no face is currently in view).
  - If a face is detected and CENTERING_COOLDOWN_S has elapsed since
    the last centering, send ONE look_at to center the face. No
    follow-up tracking, no re-centering until cooldown expires.

Cancellation/interruption are intentionally absent: if a face appears
during a look-around sweep, the sweep finishes and centering happens
on the next tick. Cycle is short enough (~12 s) that the wait is
acceptable.

To slow movement down later: bump LOOK_AROUND_POSE_DURATION_S (more
dwell at each pose) and/or drop kLookAtSpeed in commands.cpp (gentler
spring).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from websockets.asyncio.server import ServerConnection

from vision import Face

log = logging.getLogger("brain.behavior")

# Tunables — pick whatever feels right; nothing here interacts.
LOOK_AROUND_INTERVAL_S = 180.0      # 3 min between sweeps
LOOK_AROUND_POSE_DURATION_S = 3.0   # seconds held at each pose; raise to slow
CENTERING_COOLDOWN_S = 180.0        # 3 min lockout after centering
CENTERING_GAIN = 0.7                # under-center slightly; FOV is approximate

# Camera FOV — nominal for GC0308 + StackChan lens.
FOV_H_DEG = 60.0
FOV_V_DEG = 45.0

# Servo limits (mirror firmware constants).
MAX_YAW_DEG = 128.0
MIN_PITCH_DEG = 3.0
MAX_PITCH_DEG = 87.0

# Rest pose — matches the firmware bring-up sweep's final position
# (motion.move(0, 200) → pitch=20°, yaw=0).
REST_YAW_DEG = 0.0
REST_PITCH_DEG = 20.0

# Sweep poses (yaw_deg, pitch_deg). Stays inside the servo limits and
# returns to rest at the end so consecutive sweeps start from the
# same place. The dwell (LOOK_AROUND_POSE_DURATION_S) must be long
# enough for the spring at kLookAtSpeed=200 to actually settle at
# each target — otherwise the next pose preempts mid-flight and the
# sweep reads as a snap-reversal instead of a smooth glance.
LOOK_AROUND_POSES: list[tuple[float, float]] = [
    (-35.0, 25.0),                          # glance up-left
    (+35.0, 25.0),                          # glance up-right
    (0.0, 40.0),                            # tip head up briefly
    (REST_YAW_DEG, REST_PITCH_DEG),         # rest
]


class IdleBehavior:
    """Per-connection idle head behavior. Owns the head pose + the
    last-centering / last-look-around timers."""

    def __init__(self) -> None:
        self.head_yaw = REST_YAW_DEG
        self.head_pitch = REST_PITCH_DEG
        now = time.monotonic()
        # Treat boot as a recent look-around so the first sweep is
        # delayed by LOOK_AROUND_INTERVAL_S, not fired immediately.
        self.last_look_around_s = now
        self.last_centering_s = 0.0   # never centered → first face fires
        self.look_around_in_progress = False

    def notify_head_moved(self, yaw_deg: float, pitch_deg: float) -> None:
        """Sync internal pose to an externally-commanded head position
        (agent's look_at tool call). Without this, the centering math
        would compute from a stale pose."""
        self.head_yaw = yaw_deg
        self.head_pitch = pitch_deg

    async def tick(
        self,
        ws: ServerConnection,
        faces: list[Face],
        in_conversation: bool,
    ) -> None:
        """Called once per JPEG/detection. Decides whether to center,
        start a look-around, or do nothing."""
        if in_conversation:
            return

        now = time.monotonic()

        # Faces present → maybe center.
        if faces:
            if self.look_around_in_progress:
                # Wait for the sweep to finish before centering; otherwise
                # the sweep's remaining poses immediately override the
                # centered position, and the cooldown timer would still
                # be set — blocking any centering for the next 3 min.
                return
            if now - self.last_centering_s >= CENTERING_COOLDOWN_S:
                await self._center_on_face(ws, faces[0])
                self.last_centering_s = now
            return

        # No faces, no sweep running, interval elapsed → fire sweep.
        if self.look_around_in_progress:
            return
        if now - self.last_look_around_s >= LOOK_AROUND_INTERVAL_S:
            self.last_look_around_s = now
            self.look_around_in_progress = True
            asyncio.create_task(self._look_around(ws))

    async def _center_on_face(
        self, ws: ServerConnection, face: Face
    ) -> None:
        """Single look_at that centers the face in the camera frame."""
        off_x = face.cx - 0.5
        off_y = face.cy - 0.5
        target_yaw = self.head_yaw + off_x * FOV_H_DEG * CENTERING_GAIN
        # Pitch convention: smaller pitch → head down. Face below center
        # (cy > 0.5, off_y > 0) → tilt head down → subtract.
        target_pitch = self.head_pitch - off_y * FOV_V_DEG * CENTERING_GAIN

        target_yaw = max(-MAX_YAW_DEG, min(MAX_YAW_DEG, target_yaw))
        target_pitch = max(MIN_PITCH_DEG, min(MAX_PITCH_DEG, target_pitch))

        log.info(
            "center: face=(%.2f,%.2f) head=(%.1f,%.1f) → (%.1f,%.1f)",
            face.cx, face.cy, self.head_yaw, self.head_pitch,
            target_yaw, target_pitch,
        )
        await self._send_look_at(ws, target_yaw, target_pitch, source="center")

    async def _look_around(self, ws: ServerConnection) -> None:
        """Run the multi-pose sweep. Always finishes — no interruption."""
        log.info("look-around: starting (%d poses)", len(LOOK_AROUND_POSES))
        try:
            for i, (yaw, pitch) in enumerate(LOOK_AROUND_POSES):
                await self._send_look_at(
                    ws, yaw, pitch, source=f"look-around[{i}]"
                )
                await asyncio.sleep(LOOK_AROUND_POSE_DURATION_S)
        finally:
            self.look_around_in_progress = False
            log.info("look-around: done")

    async def _send_look_at(
        self,
        ws: ServerConnection,
        yaw: float,
        pitch: float,
        source: str = "?",
    ) -> None:
        try:
            await ws.send(
                json.dumps(
                    {"cmd": "look_at", "yaw_deg": yaw, "pitch_deg": pitch}
                )
            )
        except Exception:
            log.exception("look_at send failed")
            return
        log.info(
            "send look_at: yaw=%.1f pitch=%.1f (source=%s)",
            yaw, pitch, source,
        )
        self.head_yaw = yaw
        self.head_pitch = pitch
