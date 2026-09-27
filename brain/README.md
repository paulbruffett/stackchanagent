# stackchan brain

The Python agent that runs on the Jetson. The ESP32 firmware connects to it
over one WebSocket on the LAN (:8765); the brain does end-of-speech
detection, STT, the Home Assistant fast path, the LLM tool loop, TTS and
conversation memory, and serves the web console (:8080).

In production it runs as a systemd user service — see
[`../deploy/README.md`](../deploy/README.md). This file covers the code.

## Install

```bash
cd brain
uv sync --extra dev   # fetches CPython 3.12 if the system Python is older
```

Python 3.12 exactly: `onnxruntime` (via `piper-tts`) dropped 3.10 wheels, and
`ctranslate2` is pinned to a locally built CUDA aarch64 wheel in `wheels/`
that is cp312-only. So `uv.lock` can only be resolved **on the Jetson** —
re-lock there, never on the Mac (a Mac-resolved lock silently guts the venv).

## Environment

Read from `.env` at the repo root (one level above `brain/`):

| Variable | Needed for |
|---|---|
| `OPENROUTER_API_KEY` | The LLM (all models go through OpenRouter). |
| `HA_TOKEN` | Home Assistant: the fast path, STT device-name hints, and the `homeassistant` MCP server (registered with `env_ref: HA_TOKEN`). |
| `HA_URL` | Optional; defaults to `http://localhost:8123`. Deliberately not a console knob — the token goes wherever it points. |
| `CONSOLE_TOKEN` | Web console auth. If unset, one is generated and printed at startup. |
| `CONSOLE_BIND` | Optional console bind address. |

Most behaviour is tuned at runtime in the console's Config tab
(`config.py` lists every knob, its default and whether it needs a restart):
`MODEL` / `SUMMARY_MODEL` / `REASONING_EFFORT`, `STT_MODEL`, `SPEECH_RMS`,
`FOLLOW_UP_WINDOW_S`, `HA_FAST_PATH`, `BUDDY_ENABLED`, `SPEAKER_VOLUME`, `SYSTEM_PROMPT`, and so
on. Overrides persist in `~/.stackchan/memory.db` alongside the conversation,
summaries, facts and MCP registry.

First-run downloads (cached afterwards): the Piper voice
(`~/.cache/piper-voices/`) and the Whisper model (`~/.cache/huggingface/`).

## Layout

| Module | Role |
|---|---|
| `agent_server.py` | WebSocket server: VAD, turn orchestration, follow-up window, sleep, the idle ticker (summaries, buddy sync), TTS playback. |
| `stt.py` | faster-whisper wrapper, wake-word stripping, the follow-up noise gate. |
| `ha_fast_path.py` | Home Assistant intent fast path and the STT vocabulary fetch. |
| `claude_agent.py` | The LLM loop over OpenRouter (the name predates the move): streaming, tool rounds, the single-round device-command exit, history repair, summarizer. |
| `tools.py`, `mcp_client.py` | Native tools (face, head, dance, volume, battery status, memory, goodbye) and MCP servers (`mcp_servers/weather.py` bundled; Home Assistant over http). |
| `memory.py`, `config.py` | SQLite persistence and the knob registry. |
| `webui/` | The console. |

## Test

```bash
uv run --extra dev pytest
```

The suite fakes the OpenAI streaming client and Home Assistant, so it runs
offline with no key or model download. `agent_server` itself isn't imported
by the tests (it opens the live DB and loads models at import), so logic
worth testing lives in small pure functions (`policy.py`, `stt.py`,
`ha_fast_path.py`).
