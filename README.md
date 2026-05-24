# stackchan

Agentic refactor of the M5StackChan Kawaii desktop robot.

- `firmware/` — ESP32-S3 ESP-IDF project. Thin I/O: wakeword, mic,
  speaker, servos, face, camera.
- `brain/` — Python agent (Mac mini for now, Jetson Orin Nano later).
  faster-whisper STT, Claude tool-use loop (Haiku/Sonnet), Piper TTS,
  OpenCV-Haar face detection.

Single WebSocket on the LAN between them; mDNS discovery
(`stackchan-brain.local` in production; on Mac dev, point the firmware
at the host's own `.local` hostname since mDNS-on-macOS can't bind
:5353).

See [the plan](../../.claude/plans/i-have-an-m5stackchan-floating-hartmanis.md)
for the full design. Per-subproject setup lives in each directory's README.

## Dev workflow

Two terminals — one for the firmware (`idf.py`), one for the brain (Python).

**Terminal 1 — firmware (flash + serial monitor):**

```
source ~/esp/esp-idf/export.sh
cd firmware
idf.py build && idf.py -p /dev/cu.usbmodem21101 flash monitor
# Exit monitor with Ctrl+]
```

Skip `idf.py build` if you haven't touched firmware source. After
adding a new `.cpp` under `main/`, run `idf.py reconfigure` once so the
CMake glob picks it up.

**Terminal 2 — brain (websocket server):**

```
cd brain
.venv/bin/python agent_server.py 2>&1 | tee /tmp/brain.log
```

No `source` needed — `.venv/bin/python` runs in the venv directly.
`source .venv/bin/activate` works too if you prefer the activated shell.

Tail the log from a third terminal:

```
tail -f /tmp/brain.log
```

Order doesn't matter — whichever starts second waits for the other.
Firmware reconnects with exponential backoff; the brain just accepts
the first incoming WS.
