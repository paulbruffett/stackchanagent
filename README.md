# stackchan

Agentic refactor of the M5StackChan Kawaii desktop robot.

- `firmware/` — ESP32-S3 ESP-IDF project. Thin I/O: wakeword, mic,
  speaker, servos, face, camera.
- `brain/` — Python agent, running on the Jetson Orin Nano. faster-whisper
  STT, Claude tool-use loop (Haiku/Sonnet), Piper/Hume TTS, OpenCV-Haar
  face detection. Runs as a systemd user service that pulls `main` on every
  start — see `deploy/`.

Single WebSocket on the LAN between them; mDNS discovery
(`stackchan-brain.local` in production; on Mac dev, point the firmware
at the host's own `.local` hostname since mDNS-on-macOS can't bind
:5353).

Design decisions live in [`docs/adr/`](docs/adr/); deployment in
[`deploy/README.md`](deploy/README.md); per-subproject setup in each
directory's README.

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

The brain normally runs as a systemd user service on the Jetson, so there is
nothing to start by hand — `systemctl --user restart stackchan-brain` both
deploys (it resets the checkout to `origin/main`) and restarts:

```
ssh jetson
systemctl --user restart stackchan-brain
journalctl --user-unit=stackchan-brain -f
```

To run it in the foreground instead, stop the service first — otherwise the
two fight over :8765 and :8080:

```
systemctl --user stop stackchan-brain
cd ~/code/stackchanagent/brain
.venv/bin/python agent_server.py 2>&1 | tee /tmp/brain.log
```

Note the next `systemctl --user start` will reset the checkout to
`origin/main`, discarding local edits to tracked files.

Order doesn't matter — whichever starts second waits for the other.
Firmware reconnects with exponential backoff; the brain just accepts
the first incoming WS.
