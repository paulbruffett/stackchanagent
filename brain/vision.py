"""OpenCV Haar-cascade face detection for the camera frame stream.

The firmware ships a JPEG every ~1.5 s on opcode 0x02. We decode it,
run OpenCV's bundled frontal-face Haar cascade, and surface:

  - the most recent face list (each entry is normalized to the image
    frame: cx, cy ∈ [0, 1], w, h ∈ (0, 1], score = 1.0 placeholder
    since Haar doesn't return confidences)
  - a debounced `new_face` event when a face appears that wasn't
    visible in the last NEW_FACE_DEBOUNCE_S seconds

Face identity is *not* tracked (no embedding). "New face" really
means "face seen after a stretch of no faces" — good enough to drive
proactive greetings without unwanted re-fires when the user shifts
in their chair.

Haar is less accurate than mediapipe / YuNet, especially for non-frontal
or low-light faces, but it has zero extra deps (the cascade XML ships
with opencv-python-headless) and no model download. Good enough for a
desktop robot looking at the user. Swap in YuNet (cv2.FaceDetectorYN)
if accuracy becomes an issue.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("brain.vision")

# Once a face is detected, suppress new_face events until at least this
# many seconds have passed with NO face visible. Keeps the greeting
# from re-firing every time the user looks away briefly.
NEW_FACE_DEBOUNCE_S = 60.0


@dataclass(frozen=True)
class Face:
    cx: float          # normalized [0, 1], 0=left
    cy: float          # normalized [0, 1], 0=top
    w: float           # normalized
    h: float           # normalized
    score: float


class FaceDetector:
    """Wraps cv2.CascadeClassifier with the bundled frontal-face XML."""

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"failed to load Haar cascade at {path}")

    def detect(self, jpeg_bytes: bytes) -> list[Face]:
        bgr = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None:
            log.warning("jpeg decode failed (%d bytes)", len(jpeg_bytes))
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape

        # scaleFactor=1.1: standard tradeoff between speed/accuracy.
        # minNeighbors=4: drop ~half the false positives at the cost of
        # missing a few real faces.
        # minSize=(40,40): ignore tiny faces — at 320×240 that's anything
        # smaller than ~12% of the frame.
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        if len(rects) == 0:
            return []

        faces: list[Face] = []
        for (x, y, fw, fh) in rects:
            faces.append(
                Face(
                    cx=(x + fw / 2) / iw,
                    cy=(y + fh / 2) / ih,
                    w=fw / iw,
                    h=fh / ih,
                    score=1.0,
                )
            )
        # Largest first — gaze controller picks faces[0].
        faces.sort(key=lambda f: f.w * f.h, reverse=True)
        return faces


class FaceTracker:
    """Per-connection state: latest face list + new-face debounce."""

    def __init__(self) -> None:
        self.latest: list[Face] = []
        # 0 = never seen; first detection's gap is treated as "infinite"
        # so we fire a greeting on the very first face of a session.
        self.last_seen_s: float = 0.0

    def update(self, faces: list[Face]) -> bool:
        """Update state with a fresh detection. Returns True iff this
        update fires a new_face event (debounced)."""
        now = time.monotonic()
        prev_had_face = bool(self.latest)
        self.latest = faces

        if not faces:
            return False

        gap_s = now - self.last_seen_s if self.last_seen_s else float("inf")
        self.last_seen_s = now

        if prev_had_face:
            return False
        return gap_s >= NEW_FACE_DEBOUNCE_S
