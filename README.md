# stackchan

Agentic refactor of the M5StackChan Kawaii desktop robot.

- `firmware/` — ESP32-S3 PlatformIO project. Thin I/O: wakeword, mic,
  speaker, servos, face, camera.
- `brain/` — Python agent on a Jetson Orin Nano. Whisper STT, Claude
  tool-use loop (Haiku/Sonnet), Piper TTS, periodic YOLO/MediaPipe
  vision.

Single WebSocket on the LAN between them; mDNS discovery
(`stackchan-brain.local`).

See [the plan](../../.claude/plans/i-have-an-m5stackchan-floating-hartmanis.md)
for the full design. Per-subproject setup lives in each directory's README.
