# Session context, 2026-09-24

Notes from the review session that produced `plan-responsiveness.md`. Read
this before touching the voice path; it records what was found and where.

## State of the repo

- `origin/main` (decc3ca) is what runs on the Jetson. The local checkout had
  been sitting at the phase-0 commit (e9f663a), 117 commits behind, until this
  session fast-forwarded it.
- Prose in `README.md` and `brain/README.md` still describes phases 0–2. The
  code is well past that: MCP + A2A tool sources, web console, memory
  summarizer, Rocky mode, BLE buddy, camera/face behaviour, sleep mode.
- No Home Assistant integration exists in the repo. Lights go through either
  the bundled Hue stdio MCP server (`brain/mcp_servers/hue.py`) or an HA MCP
  server registered over http in the console (the last commit added bearer
  auth for that). Which one is live is only knowable from `memory.db` on the
  Jetson.

## The voice turn, as implemented

Firmware (`firmware/main/agent/`): ESP-SR wake word "computer" → JSON event
over one WebSocket → mic frames (16 kHz s16le, 20 ms, opcode 0x01) while in
LISTENING → PCM back (opcode 0x01) while in SPEAKING. State machine in
`state.cpp` pauses the wake word in LISTENING and SPEAKING. Brain commands are
queued and run on the main loop (`commands.cpp`).

Brain (`brain/agent_server.py`): RMS VAD (threshold 150, 200 ms lead, 700 ms
tail, 10 s cap) → faster-whisper `small.en` fp16 CUDA in a worker thread →
`AgentSession.respond` (Haiku 4.5, streaming, sentence-chunked to Piper) →
tool loop up to 6 rounds → follow-up window (8 s, wake word paused, extended
thinking on) → idle.

## Findings that drive the plan

1. Two LLM round trips per device command; the light flips between them.
2. `_turn_lock` is shared by the user turn, the proactive greeting
   (`agent_server.py` `proactive_greet`) and the summarizer
   (`claude_agent.py` `_maybe_summarize`, two LLM calls). A user turn can
   queue silently behind either.
3. The follow-up window pauses the wake word for ~9 s after every reply.
4. RMS VAD lets quiet speech run to the 10 s cap.
5. BLE buddy keeps NimBLE advertising alongside the Wi-Fi audio stream;
   `speaker_play.cpp` already logs playback-queue overflows.
6. Every service restart does `git reset --hard` + `uv sync` check + Whisper
   reload (`deploy/update-brain.sh`).

Nothing was measured: SSH and the HA API are locked out from the dev Mac.
The console's Transactions tab (`stt_ms`, `total_ms`) is the baseline source.

## Where things live

- Config knobs and defaults: `brain/config.py` `SPECS` (hot vs restart).
- MCP timeouts: `brain/mcp_client.py` (connect 20 s, call 30 s, dispatch 45 s).
- Wire protocol: docstring at the top of `brain/agent_server.py`.
- Deploy and console access: `deploy/README.md`.
- Jetson: `192.168.4.150`, brain on 8765, console on 8080 (token in `.env`),
  HA on 8123, Matter Server on 5580.
