# stackchan

A voice assistant on an M5Stack CoreS3 "Stack-chan" robot: say "computer,
turn off the office light" and the light goes off about a second later.

- `firmware/` — ESP32-S3 ESP-IDF project. Thin I/O only: the "computer"
  wake word (ESP-SR), microphone streaming (with 500 ms of pre-roll so the
  words right after the wake word survive), speaker playback, head servos
  and the face. Watchdogs return it to idle if the brain goes quiet
  mid-turn. Optional BLE link for Claude Desktop permission prompts
  (`BUDDY_ENABLED`, off by default).
- `brain/` — Python agent on a Jetson Orin Nano, running as a systemd user
  service (see [`deploy/`](deploy/README.md)). End-of-speech detection,
  faster-whisper STT (`small.en`, hinted with Home Assistant's device and
  room names), the Home Assistant fast path, the LLM tool loop via
  OpenRouter (default `openai/gpt-5.6-luna`, switchable in the console),
  MCP tools, Piper TTS, conversation memory and a web console on :8080.

The two talk over a single WebSocket on the LAN (brain on :8765, found via
mDNS as `stackchan-brain.local`).

## How a request flows

1. The firmware hears "computer", starts streaming mic audio (pre-roll
   first) and shows its listening face.
2. The brain detects the end of speech (a loudness threshold, `SPEECH_RMS`,
   plus a 700 ms silence tail) and transcribes it.
3. **Home Assistant fast path:** the transcript goes to HA's local intent
   matcher. A definite hit (a device action or a state answer) is spoken
   straight back — no LLM call, ~0.1 s.
4. Anything else goes to the LLM with its tools: Home Assistant (over MCP),
   weather, timers and reminders, and the robot's own face, head, dance,
   speaker volume, battery status, memory and goodbye. A device
   command the model handles takes one model round: it confirms in the same
   message as the tool call.
5. Replies stream sentence by sentence into Piper and back to the speaker.
   A short follow-up window (4.5 s) then lets you reply without the wake
   word.

Between conversations the brain folds older turns into summaries and
extracts durable facts; both are editable in the console.

## Docs

- [`docs/plan-responsiveness.md`](docs/plan-responsiveness.md) — why the
  system looks like this: the latency diagnosis, what was cut, and what's
  left.
- [`deploy/README.md`](deploy/README.md) — the Jetson service, logs,
  console access.
- [`brain/README.md`](brain/README.md), [`firmware/README.md`](firmware/README.md)
  — per-subproject setup.
- [`CONTEXT.md`](CONTEXT.md) — glossary; [`docs/adr/`](docs/adr/) — decisions.

## Dev workflow

**Brain.** It runs as a systemd user service on the Jetson that resets its
checkout to `origin/main` on every start, so merging to `main` and
restarting *is* the deploy:

```
ssh jetson
systemctl --user restart stackchan-brain
journalctl --user-unit=stackchan-brain -f
```

The update step skips `git fetch` if it fetched in the last 60 s, so two
restarts within a minute don't pull. To run the brain in the foreground,
stop the service first (otherwise both fight over :8765 and :8080) — and
note the next `systemctl --user start` resets the checkout again:

```
systemctl --user stop stackchan-brain
cd ~/code/stackchanagent/brain
.venv/bin/python agent_server.py
```

**Firmware.**

```
source ~/esp/esp-idf/export.sh
cd firmware
idf.py build && idf.py -p /dev/cu.usbmodem21101 flash
```

The CoreS3's automatic reset doesn't work over its native USB: hold the
bottom-left reset button ~2 s (green LED) to enter download mode before
flashing, then press it briefly to boot the new image. After adding a
`.cpp` under `main/`, run `idf.py reconfigure` once so the CMake glob picks
it up.

Logs come out of the USB-C port (115200 baud). Opening the port reboots
the robot, so open it once and keep it open. While listening, the firmware
logs one line per second — `uplink/s: read=51 dropped=0 sent=51` is a
healthy mic stream.

Order doesn't matter — the firmware reconnects with backoff, and the brain
accepts the connection whenever it arrives.
