# stackchan brain

Jetson-side Python agent. The ESP32 firmware connects to this over a single
WebSocket on the LAN; the brain runs STT, the Claude tool-use loop, TTS,
and periodic vision.

## Install

```bash
cd brain
python -m venv .venv && source .venv/bin/activate
pip install -e .              # phase 1: just websockets + zeroconf + anthropic
# pip install -e '.[voice]'   # phase 2+
# pip install -e '.[vision]'  # phase 4+
```

## Run

```bash
python agent_server.py
```

Advertises `stackchan-brain.local` via mDNS on port 8765. The firmware
resolves that name at boot — no IP config needed.

## Phase 1 verification

1. Start `agent_server.py` on the Jetson.
2. Flash phase-1 firmware on the CoreS3 (next firmware iteration; current
   firmware is phase 0 hardware bring-up only).
3. Speak — audio is echoed back through the StackChan's speaker with a
   small LAN latency. Confirms the wire protocol + clocking.

Once that passes we layer in `faster-whisper` and `piper-tts` in phase 2.

## Env

`ANTHROPIC_API_KEY` is required from phase 3 onward.
