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

Python 3.12 is required — `onnxruntime` (pulled in by `piper-tts`) dropped
3.10 wheels, and the locally-built CUDA `ctranslate2` wheel is cp312-only.
On Jetson Ubuntu 22.04 the system Python is 3.10, so `uv` downloads a
managed `python-build-standalone` aarch64 build — the system Python is
left untouched.

`uv.lock` is resolved **on the Jetson**, since the `ctranslate2` pin is an
aarch64 path wheel. Re-lock there, not on the Mac.

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

## Test

```bash
uv run --extra dev pytest        # or, in the venv: python -m pytest
```

`tests/` covers the conversation-state hardening (M6): atomic exchange
persistence, tool-dispatch recovery, the message-thread contract
(`validate_thread`) + read-time sanitizer, the startup integrity pass
(`repair_memory`), and the follow-up false-trigger gate. The suite uses a
scripted fake Anthropic client, so it runs offline with no API key or model
download.

## Phase 1 verification

1. Start `agent_server.py` on the Jetson.
2. Flash phase-1 firmware on the CoreS3 (next firmware iteration; current
   firmware is phase 0 hardware bring-up only).
3. Speak — audio is echoed back through the StackChan's speaker with a
   small LAN latency. Confirms the wire protocol + clocking.

Once that passes we layer in `faster-whisper` and `piper-tts` in phase 2.

## Env

`ANTHROPIC_API_KEY` is required from phase 3 onward.

### Rocky mode (optional)

A device-wide character mode: the robot adopts the Rocky persona (the
broken-English alien engineer from *Project Hail Mary*) and, when a Hume AI
voice is configured, switches speech to a Hume cloud voice. Toggle it in the
web Config tab (`ROCKY_MODE` under the **rocky** group) or by voice ("computer,
rocky mode" / "normal mode").

It degrades gracefully: with **no** Hume key the persona still works, spoken on
the normal Piper voice. To enable the cloud voice, add to `../.env`:

```
HUME_API_KEY=...
HUME_VOICE_ID=...          # a voice id, e.g. a cloned Rocky voice
HUME_VOICE_PROVIDER=CUSTOM_VOICE   # or HUME_AI for a library voice by id
HUME_FALLBACK_VOICE=...    # a Hume library voice *name*, used if no id is set
```

`ROCKY_SPEED` (rocky config group) sets the Hume speaking-rate multiplier
(Piper ignores it).
