# Plan: make Stack-chan respond fast and every time

Written 2026-09-24 against `origin/main` (decc3ca). The local `main` checkout is
117 commits behind that and still shows the phase-0 servo sweep, so everything
below refers to what is actually deployed on the Jetson.

Core use case, as stated: a voice assistant with a web interface and a
configurable LLM that can call tools/agents (Home Assistant, Hue). Everything
else is negotiable.

---

## 1. Where the time and the misses come from today

The path for "computer, turn on the office light":

| Stage | Where | What it costs / how it fails |
|---|---|---|
| Wake word "computer" | ESP-SR on the ESP32 | Fast and local. **But it is paused whenever the firmware is in LISTENING or SPEAKING**, which includes the 8 s follow-up window after every reply plus the playback wait. Say "computer …" inside that window and it is not a wake word; it becomes a "follow-up" utterance the model judges with extended thinking on. Reads as "ignored me" or "slow". |
| Utterance end | brain, RMS threshold 150 + 700 ms silence tail | 0.7 s floor. A quiet speaker who never crosses RMS 150 for 200 ms never trips speech-lead, so the capture runs to the 10 s `MAX_UTTERANCE_MS` cap before anything happens. Plausible source of the "sometimes 10 s, sometimes nothing" feel. |
| STT | faster-whisper `small.en` fp16 CUDA, one lock | Fine in isolation. Shares the Orin Nano with HA, Matter Server, Piper threads, and OpenCV face detection every 20 s. |
| LLM round 1 | Haiku 4.5, streaming | Prompt = persona + up to 200 facts + up to 12 summaries + up to 20 raw turns + 6 native tools + every MCP tool schema (HA's MCP server alone is a dozen large schemas) + A2A tools. Ends in `tool_use`; a canned filler is spoken. |
| Tool call | MCP child (Hue, stdio) or HA (streamable-http), 30 s call timeout, 45 s dispatch timeout | The light flips here, ~3–4 s after you stop talking. |
| LLM round 2 | Haiku again | Only exists to phrase "Done, the office light is on." |
| TTS | Piper on CPU, per sentence, paced at 0.8x real time | Adds ~0.3–0.8 s before first audio. |

Best case is roughly 5–7 s from end of speech to spoken confirmation. That is
structural, not a bug. I could not measure real numbers (no SSH, no console
token); the console's **Transactions** tab already records `stt_ms` and
`total_ms` per turn, so the baseline is a few minutes of reading.

Things that make a given attempt much worse, or silent:

1. **Turn-lock contention.** A proactive greeting (face detector saw a "new"
   face) and the background summarizer (two LLM calls: summary + fact
   extraction, every ~20 turns) take the same `_turn_lock` as your utterance.
   Your turn waits behind them with no feedback. `agent_server.py:632`,
   `claude_agent.py:1262`.
2. **The follow-up window.** 8 s of open mic after every reply, wake word
   paused, any noise transcribed and sent to the model with a 1024-token
   thinking budget. It also chains: a reply opens another window.
3. **Stranded firmware state.** Most of the recent firmware commits are fixes
   for the device being stuck in LISTENING/SPEAKING. The brain now forces IDLE
   on reconnect, but a wedged turn on a live connection (an MCP call riding
   out its 45 s timeout, six tool rounds) leaves the device deaf for that long.
4. **BLE + Wi-Fi on one radio.** The "buddy" BLE peripheral (Claude Desktop
   permission prompts) keeps NimBLE advertising alongside the Wi-Fi audio
   stream. ESP32-S3 coexistence is time-sliced; the magnitude here is
   unmeasured, but it is a known source of jitter and dropped frames, and
   `speaker_play` already logs queue overflows.
5. **Sleep mode.** Screen off after 5 min idle. Wake word stays armed, but any
   hiccup in the wake path now looks like a dead robot.
6. **Restart cost.** Every service restart runs `git fetch` + `reset --hard` +
   `uv sync` check, then reloads Whisper. A daytime restart is tens of seconds
   of dead air.

## 2. What to cut

Nothing below serves the stated core use case, and each one either takes the
turn lock, competes for the Jetson, keeps the radio busy, or widens the prompt.

| Remove | Files (origin/main) | Why it hurts |
|---|---|---|
| Camera, face detection, proactive greeting, look-around, centering, sleep | `brain/vision.py`, `brain/behavior.py`, greeting/sleep code in `agent_server.py`, `firmware/main/agent/camera_pump.cpp`, `describe_view` tool | Turn-lock contention, CPU on the Jetson, JPEG every 1.5 s over the same socket as audio |
| Rocky mode, Hume TTS, Rocky skin, sprite tooling | `brain/tts_hume.py`, `firmware/main/stackchan/avatar/skins/rocky/`, `firmware/tools/sprite_gen/`, ADR 0001 | Cloud TTS on the reply path with fallback logic; a second persona prompt; skin swap on connect |
| BLE buddy | `firmware/main/agent/buddy_ble*.{cpp,c,h}`, `hal_ble.cpp`, `bleprph/`, `CONFIG_BT_*` | Radio coexistence, tap-handler branching, sleep-timeout special cases |
| A2A client | `brain/a2a_client.py`, A2A tab, `a2a_servers` table | Extra tool defs every turn, 60 s call budget, nothing uses it for lights |
| Summaries + automatic fact extraction + consolidation | most of `brain/memory.py`, half of `claude_agent.py` | Two LLM calls under the turn lock; the history-repair machinery exists only because of this |
| Follow-up window with extended thinking | `agent_server.py` follow-up code, `stt.should_drop_follow_up`, `FOLLOWUP_*` knobs | Pauses the wake word; false triggers; thinking latency |

Keep: wake word, mic stream, VAD, STT, one LLM loop, MCP tools, Piper TTS,
facial expressions and head moves as tools (they are cheap and they are the
charm), the web console (config, MCP, transactions, logs), the deploy unit.

## 3. Two shapes for what remains

### Option A: prune the brain in place, keep the firmware

Evolutionary. The firmware works and the latency is not in it, apart from BLE
and the state machine. Delete the modules above, then restructure the turn:

- **One LLM round for device commands.** Ask the model to say the confirmation
  in the same message as the `tool_use` ("Turning on the office light.") and,
  for tools flagged fire-and-forget, do not go back to the model unless the
  tool errored. Halves LLM latency on the most common request. The
  `tool_result` still has to be staged and committed; the next user message
  carries it as its first block, which the API allows.
- **Home Assistant intent fast path.** After STT, `POST /api/conversation/process`
  with the transcript. HA's built-in matcher handles "turn on the office
  light" locally in well under a second and returns speech text. If
  `response_type` is `action_done`, speak it with Piper and never call Claude.
  Anything HA cannot match falls through to the LLM with tools as today.
  About forty lines. Needs an HA long-lived token in `.env`.
- **Better end-of-utterance.** Silero VAD ships inside faster-whisper
  (`faster_whisper.vad`); run it on the 20 ms frames instead of RMS, with a
  ~400 ms tail. Cheap on CPU. If that is more than you want, the minimum is
  shortening `SILENCE_TAIL_MS` and adding an adaptive noise floor.
- **Small prompt.** Last 6 turns, facts capped at ~20 and hand-edited in the
  console, tools limited to what is enabled. Keep the `MODEL` knob.
- **Timing in the console.** Add `vad_ms`, `llm_ttft_ms`, `tool_ms`, `tts_ms`
  next to the existing `stt_ms`/`total_ms` so regressions are visible.
- **Firmware, one reflash.** Remove buddy BLE and set `CONFIG_BT_ENABLED=n`,
  remove the camera pump and Rocky skin. Add one local safety: if LISTENING
  lasts more than ~15 s with no brain command, return to IDLE and re-arm the
  wake word, so a wedged turn never leaves the device deaf.
- **Deploy.** Stop pulling on every restart; pull on demand. Restarts become
  fast and deterministic.

### Option B: let Home Assistant be the assistant

A reset. The brain becomes a ~300-line bridge: wake-word event and audio in
from the firmware, out to HA's Assist pipeline over its WebSocket
(`assist_pipeline/run`, `start_stage: stt`, `end_stage: tts`), TTS audio back
to the firmware on the existing wire protocol. HA does STT (Wyoming whisper
container), local intent matching, LLM fallback (its Anthropic, OpenAI or
Ollama conversation integrations, chosen in the UI, with exposed entities as
tools and HA's MCP *client* integration for external servers), and TTS
(Wyoming piper). The web interface becomes HA's Assist settings plus its
pipeline debug view, which shows per-stage timing for every run.

What you give up: the persona and durable facts (HA agents take a prompt
template but have no memory), sentence-streamed TTS from Claude (HA has been
adding streaming, but I have not verified it end to end with Piper), and the
A2A agents. What you also give up: almost all of `brain/`. Firmware stays as
in Option A. An ESPHome voice-satellite firmware was considered and rejected:
it would lose the servos and the face.

Prerequisites I cannot do from here: an HA long-lived token, and Wyoming
whisper/piper running on the Jetson (containers if HA is in Docker, add-ons if
it is HA OS).

### Recommendation

Do the Option A prune, and include the HA fast path from day one. The prune is
needed under either option, the fast path gives the top request sub-2 s
without any LLM call, and everything else still has the configurable LLM and
MCP tools behind it. If after a few weeks the fast path is handling most of
what you say, Option B is a short step: the bridge already exists in the
fast-path code.

## 4. Phases

**Phase 0, baseline (an hour, no code).** Ten runs of the light command.
Record `stt_ms` and `total_ms` from the Transactions tab and note every miss.
Grep the journal for `dropped`, `queue overflow`, `sanitizing`, `busy for`.
Confirm which tool actually handles lights today (Hue stdio server or HA over
http) in the MCP tab. Turn on `STT_DEBUG_DUMP` for one session and listen to
two captures.

**Phase 1, config-only experiment (same day).** In the console, no restart:
`FOLLOW_UP_WINDOW_S=0`, `FOLLOW_UP_THINKING=0`, `SLEEP_TIMEOUT_S=0`,
`GREETING_COOLDOWN_S` and `RECENT_INTERACTION_S` and `LOOK_AROUND_INTERVAL_S`
very large, `SUMMARIZE_TRIGGER=100000`, `MAX_PROMPT_FACTS=20`,
`SILENCE_TAIL_MS=450`, `ROCKY_MODE=0`, disable unused MCP/A2A servers. Repeat
the ten runs. This isolates how much of the inconsistency is the background
behaviour before any code is deleted.

**Phase 2, brain prune + fast path (2–3 days).** Delete the modules in
section 2; remove every other holder of the turn lock; add the HA fast path;
single-round device tools; Silero VAD; timing columns. Tests: keep
`test_agent_loop`, `test_mcp_a2a` (MCP half), `test_webui`, `test_memory`
(facts + config); drop the summarizer, repair, sanitize and follow-up suites
with the code they cover.

**Phase 3, firmware prune (1 day, one reflash).** Remove buddy BLE, camera
pump, Rocky skin; BT off; LISTENING watchdog. Keep the vendored xiaozhi slice
as the audio/wake-word HAL; replacing it with bare esp-sr is a separate
project with no latency payoff.

**Phase 4, deploy (an hour).** Pull on demand instead of every restart.
Re-run the ten-run baseline and compare.

Then decide on Option B with data.

## 5. Decisions needed

1. Which tool turns lights on today: the bundled Hue MCP server, or HA's MCP
   server over http? Where does the name "office light" live?
2. Can you create an HA long-lived access token? Is HA running as HA OS
   (add-ons available) or in Docker?
3. "Configurable LLM": Anthropic models only, as now, or should OpenAI/Ollama
   be selectable? The latter argues for Option B or a provider shim.
4. Anything in the cut list you want to keep? I have assumed expressions and
   head motion stay, and that camera, Rocky, BLE buddy and A2A go.
5. Access for me: a console token URL lets me run Phase 0 and 1 from the
   browser; an SSH key lets me deploy Phases 2–4. Otherwise I hand you the
   commands.
6. Fast-forward the local `main` to `origin/main`? The working tree is clean
   and strictly behind, so it is a no-op merge.
