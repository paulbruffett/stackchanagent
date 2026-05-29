# stackchan brain

Jetson-side Python agent. The ESP32 firmware connects to this over a single
WebSocket on the LAN; the brain runs STT, the Claude tool-use loop, TTS,
and periodic vision.

## Install

All runtime deps (websockets, anthropic, faster-whisper, piper-tts,
opencv-headless, numpy) are required and live in the base dependency
list — there are no optional groups to opt into.

With `uv` (recommended on the Jetson):

```bash
cd brain
uv sync   # auto-fetches CPython 3.12 if the system Python is older
```

Python 3.11+ is required (`onnxruntime`, pulled in by `piper-tts`,
dropped 3.10 wheels). On Jetson Ubuntu 22.04 the system Python is 3.10,
so `uv` downloads a managed `python-build-standalone` aarch64 build —
the system Python is left untouched.

Or with plain pip:

```bash
cd brain
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

`ANTHROPIC_API_KEY` is loaded from a `.env` at the repo root (one level
above `brain/`). Create it once:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > ../.env
```

First-run downloads (cached afterwards):
- Piper voice (`en_US-amy-medium`) → `~/.cache/piper-voices/`
- Whisper model (`small.en` int8) → `~/.cache/huggingface/`

## Run

```bash
uv run agent_server.py
# or, in the venv: python agent_server.py
```

Listens on `0.0.0.0:8765`. Also attempts to advertise
`stackchan-brain.local` via mDNS — non-fatal if it fails (the firmware
can be pointed at a literal IP via `idf.py menuconfig` → Stackchan
Brain → Brain host).

## Phase 1 verification

1. Start `agent_server.py` on the Jetson.
2. Flash phase-1 firmware on the CoreS3 (next firmware iteration; current
   firmware is phase 0 hardware bring-up only).
3. Speak — audio is echoed back through the StackChan's speaker with a
   small LAN latency. Confirms the wire protocol + clocking.

Once that passes we layer in `faster-whisper` and `piper-tts` in phase 2.

## Env

`ANTHROPIC_API_KEY` is required from phase 3 onward.
