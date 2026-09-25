"""WebSocket server the M5StackChan ESP32 connects to.

Phase 2: wakeword → LISTENING (accumulate PCM, RMS-based silence VAD) →
STT → TTS → SPEAKING (stream PCM back) → IDLE. Driven from a per-
connection state machine.

Wire protocol:
  - Binary frame, first byte = opcode:
      0x01  PCM audio frame (16 kHz, mono, s16le, 20 ms = 640 bytes)
  - Text frame: JSON control message, both directions.
      from ESP32: {"event": "boot"|"wakeword"|"vad_end", ...}
      to   ESP32: {"cmd": "stop_listening"|"start_speaking"|"stop_speaking"|
                          "set_expression"|"look_at"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import socket
import sys
import time
import wave
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import uvicorn
from dotenv import load_dotenv
from websockets.asyncio.server import ServerConnection, serve
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

# Load ANTHROPIC_API_KEY (and any other env) from the project root .env
# before importing the agent module (which constructs the Anthropic client).
load_dotenv(Path(__file__).parent.parent / ".env")

from claude_agent import AgentSession, maybe_summarize, repair_memory
from config import get_config, init_config
import ha_fast_path
from mcp_client import McpClient
from policy import effective_sleep_timeout
from memory import Memory
from stt import Transcriber, should_drop_follow_up, strip_wake_word
from tasks import spawn
from tts import Synthesizer
from webui.app import create_app
from webui.logbuf import LOGS, TURNS, WebUILogHandler, publish_turn

HOST = "0.0.0.0"
PORT = 8765
WEB_PORT = 8080
MDNS_NAME = "stackchan-brain"

OP_AUDIO = 0x01

# Audio assumptions (must match firmware).
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2                  # 640

# VAD / follow-up / sleep knobs are hot-editable via config.py (web UI).
# Read with get_config().get("SPEECH_RMS") etc. at the use sites below.
# Defaults live in config.py::SPECS.

log = logging.getLogger("brain")

# Lazy globals — one model load per process.
stt = Transcriber()
tts = Synthesizer()
# Single shared Memory across all WS connections, so the robot
# remembers conversations even after a disconnect/reconnect or process
# restart. Sqlite handles file locking; only one process should write.
memory = Memory()
# Shared MCP client (Phase 9b): one set of server connections for the
# whole process. Started in main(); tools merged into every agent turn.
mcp_client = McpClient(memory)
# Live agent sessions, one per connected firmware WebSocket (normally exactly
# one). The web console edits durable history; a session hydrates its thread
# once at construction, so the console needs a way to reach the in-memory copy
# — see resync_live_sessions.
live_sessions: set[AgentSession] = set()

# A session busy with a turn holds its turn lock for the whole exchange
# (model round trips + tool calls). Bound the console's wait on it rather than
# hanging the HTTP request behind a wedged turn — which is exactly the state
# the operator is usually trying to clear.
RESYNC_TIMEOUT_S = 15.0


@dataclass
class ConnState:
    listening: bool = False
    speaking: bool = False
    # True when listening was opened by the brain (post-reply window) rather
    # than by a firmware wakeword event. Drives the agent to use
    # respond_follow_up so it can stay silent on side conversation.
    follow_up: bool = False
    # Timeout task that closes the follow-up window if the user never speaks.
    # Cancelled the moment speech is detected (or another turn starts).
    follow_up_timeout: asyncio.Task | None = None
    speech_buf: bytearray = field(default_factory=bytearray)
    voiced_ms: int = 0
    trailing_silence_ms: int = 0
    started_at: float = 0.0
    agent: AgentSession | None = None
    # True while asleep: the screen is off until a wake word or head tap.
    # Driven by SLEEP_TIMEOUT_S.
    asleep: bool = False
    # monotonic time of the last interaction that should keep the device
    # awake (conversation, wake word, tap). Seeded at connect so a fresh
    # connection doesn't immediately sleep.
    last_activity_s: float = 0.0
    # last_activity_s of the idle stretch the summarizer already ran in, so a
    # long quiet spell folds the backlog once rather than every tick.
    summarized_for_activity_s: float = -1.0
    # The in-flight fold, if any, and when the last one started (spaces out
    # retries of the overflow fold if the LLM call keeps failing).
    summarize_task: asyncio.Task | None = None
    last_summarize_s: float = 0.0
    # monotonic time the most recent TTS playback is expected to finish on
    # the device. The brain sends audio faster than real time, so when the
    # speaker worker returns the device still has buffered audio playing;
    # we wait until past this before reopening the mic (else the robot
    # hears its own voice tail and replies to itself).
    est_playback_end_s: float = 0.0
    # True while a BLE buddy approve prompt is pending on the device (firmware
    # emits {"event":"buddy_prompt","pending":...}). Elongates the sleep
    # timeout so the device doesn't sleep out from under an unanswered prompt.
    buddy_prompt_pending: bool = False


def frame_rms(frame: bytes) -> float:
    if len(frame) < 2:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


async def send_pcm_stream(ws: ServerConnection, pcm: bytes) -> None:
    """Stream PCM bytes to firmware as OP_AUDIO frames at FRAME_BYTES each,
    paced to roughly real-time so the firmware speaker queue doesn't blow up.
    """
    # Pace by sleeping between frames; speaker queue is 1 s cap.
    for i in range(0, len(pcm), FRAME_BYTES):
        chunk = pcm[i : i + FRAME_BYTES]
        if not chunk:
            continue
        if len(chunk) < FRAME_BYTES:
            chunk = chunk + b"\x00" * (FRAME_BYTES - len(chunk))
        await ws.send(bytes([OP_AUDIO]) + chunk)
        # Pace at slightly faster than real-time so we lead the playback queue.
        await asyncio.sleep(FRAME_MS / 1000 * 0.8)


def ensure_agent(ws: ServerConnection, state: ConnState) -> AgentSession:
    if state.agent is None:
        state.agent = AgentSession(ws, memory=memory, mcp=mcp_client)
        live_sessions.add(state.agent)
    return state.agent


async def resync_live_sessions() -> int:
    """Re-hydrate every live session's in-memory thread from SQLite, so a
    console edit to the conversation tail reaches the connected device instead
    of only landing at the next brain restart. Returns how many sessions were
    re-synced; a session that stays busy past RESYNC_TIMEOUT_S is skipped and
    logged rather than blocking the console."""
    synced = 0
    for session in list(live_sessions):
        try:
            await asyncio.wait_for(session.resync_from_memory(), RESYNC_TIMEOUT_S)
            synced += 1
        except TimeoutError:
            log.warning(
                "live session busy for %.0fs — its in-memory thread still "
                "holds the old turns; restart the brain to clear it",
                RESYNC_TIMEOUT_S,
            )
    return synced


async def run_speaker(
    ws: ServerConnection,
    state: ConnState,
    queue: asyncio.Queue[str | None],
) -> None:
    """Drain `queue` until a None sentinel: synth each sentence with TTS
    and ship the PCM. Sends `start_speaking` lazily on the first sentence
    (so a tool-only turn with no spoken output doesn't toggle the
    speaking face) and `stop_speaking` only if we ever started."""
    started = False
    state.speaking = True
    play_start: float | None = None
    total_audio_s = 0.0
    try:
        while True:
            sentence = await queue.get()
            if sentence is None:
                break
            if not started:
                await ws.send(json.dumps({"cmd": "start_speaking"}))
                started = True
            t0 = time.monotonic()
            tts_pcm = await asyncio.to_thread(tts.synthesize, sentence)
            audio_s = len(tts_pcm) / (SAMPLE_RATE * 2)
            log.info(
                "tts: %d ms, %.2fs audio, %r",
                int((time.monotonic() - t0) * 1000), audio_s, sentence[:80],
            )
            # Playback starts ~when the first frame reaches the device.
            if play_start is None:
                play_start = time.monotonic()
            total_audio_s += audio_s
            await send_pcm_stream(ws, tts_pcm)
        if started:
            await ws.send(json.dumps({"cmd": "stop_speaking"}))
    finally:
        state.speaking = False
        # Device plays at real time from play_start; record when the last
        # sample will have left the speaker so respond() can wait it out.
        state.est_playback_end_s = (
            play_start + total_audio_s if play_start is not None else 0.0
        )


async def _drive_agent_turn(
    ws: ServerConnection,
    state: ConnState,
    run_agent: Callable[[Callable[[str], Awaitable[None]]], Awaitable[str]],
) -> str:
    """Wire a sentence-streaming agent run to a speaker worker. The
    agent calls `enqueue(sentence)` as each sentence completes; the
    speaker worker drains the queue in parallel so TTS pacing doesn't
    backpressure the LLM stream."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    speaker = asyncio.create_task(run_speaker(ws, state, queue))

    async def enqueue(sentence: str) -> None:
        await queue.put(sentence)

    try:
        return await run_agent(enqueue)
    finally:
        await queue.put(None)
        await speaker


def _dump_capture(pcm: bytes) -> None:
    """Write a captured utterance to ~/.stackchan/captures/*.wav (16 kHz mono
    s16le) so the raw STT input can be listened to. Gated by STT_DEBUG_DUMP;
    best-effort — never let a debug write break a turn."""
    try:
        out_dir = Path.home() / ".stackchan" / "captures"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"utt-{int(time.time() * 1000)}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)
        log.info("saved capture: %s (%.2fs)", path, len(pcm) / (SAMPLE_RATE * 2))
    except Exception:
        log.exception("capture dump failed")


async def respond(ws: ServerConnection, state: ConnState) -> None:
    """Run STT → streaming agent → sentence-chunked TTS, then either
    open a follow-up window (if we actually spoke) or go idle."""
    # Speech has been captured; the timeout task no longer needs to fire.
    _cancel_follow_up_timeout(state)

    pcm = bytes(state.speech_buf)
    state.speech_buf = bytearray()
    state.listening = False
    voiced_ms = state.voiced_ms  # snapshot before reset; used by the follow-up gate
    state.voiced_ms = 0
    state.trailing_silence_ms = 0

    # Snapshot + clear the follow-up flag now so the agent picks the
    # right entrypoint and we don't re-enter follow-up mode by accident.
    follow_up_turn = state.follow_up
    state.follow_up = False

    await ws.send(json.dumps({"cmd": "stop_listening"}))

    if get_config().get("STT_DEBUG_DUMP"):
        # Whole-file WAV write; off the loop like every other blocking call
        # here, so turning the debug dump on doesn't stall the console/mDNS.
        await asyncio.to_thread(_dump_capture, pcm)
    try:
        transcript = await stt.transcribe(pcm)
    except Exception:
        # A CUDA OOM here used to escape the message loop and drop the socket,
        # turning one failed utterance into a reconnect. The firmware has
        # already been sent stop_listening, so just treat it as "heard nothing".
        log.exception("stt failed — going idle")
        return
    transcript = dataclasses.replace(transcript, text=strip_wake_word(transcript.text))
    if not transcript.text:
        log.info("empty transcript — going idle")
        return

    # Follow-up false-trigger gate (M6.7). A follow-up turn needs no wakeword,
    # so a noise blip hallucinated into text would otherwise start a turn and
    # open yet another window (a self-perpetuating loop). Drop it if Whisper is
    # unconfident OR the capture is a clipping blip too short to be speech.
    # Wakeword turns are never gated here.
    if follow_up_turn:
        cfg = get_config()
        drop, reason = should_drop_follow_up(
            transcript,
            voiced_ms,
            max_no_speech_prob=cfg.get("FOLLOWUP_MAX_NO_SPEECH_PROB"),
            min_avg_logprob=cfg.get("FOLLOWUP_MIN_AVG_LOGPROB"),
            clip_peak_pct=cfg.get("FOLLOWUP_CLIP_PEAK_PCT"),
            min_voiced_ms=cfg.get("FOLLOWUP_MIN_VOICED_MS"),
        )
        if drop:
            log.info(
                "dropped follow-up (%s): %r (no_speech=%.2f avg_logprob=%.2f "
                "peak=%.1f%% voiced=%d ms)",
                reason,
                transcript.text,
                transcript.no_speech_prob,
                transcript.avg_logprob,
                transcript.peak_pct,
                voiced_ms,
            )
            return

    log.info(
        "transcript: %r (%d ms)%s",
        transcript.text,
        transcript.latency_ms,
        " [follow-up]" if follow_up_turn else "",
    )

    agent = ensure_agent(ws, state)
    tool_calls: list[dict[str, Any]] = []
    # Per-turn observer, not a slot on the session: _turn_lock is taken inside
    # respond*, so another turn can be in flight against the same AgentSession,
    # and a shared slot let one turn collect the other's tool calls and then
    # publish its own as empty.
    on_tool = lambda name, inp: tool_calls.append({"name": name, "input": inp})
    t0 = time.monotonic()
    # Home Assistant first: a device command or state question it matches
    # locally is answered in ~50 ms with no LLM call; a miss adds the same
    # ~50 ms before the LLM (published as ha_ms either way). Not on follow-up
    # turns: with no wake word, the model has to judge whether the words were
    # even meant for us (side conversation, TV, our own echo) before anything
    # in the house moves.
    fast = None if follow_up_turn else await ha_fast_path.try_handle(transcript.text)
    ha_hit = fast is not None and fast.speech is not None
    if ha_hit:
        speech = fast.speech

        async def run(spk: Callable[[str], Awaitable[None]]) -> str:
            await spk(speech)
            await agent.record_exchange(transcript.text, speech, follow_up=False)
            return speech
    elif follow_up_turn:
        run = lambda spk: agent.respond_follow_up(
            transcript.text, spk, on_tool=on_tool
        )
    else:
        run = lambda spk: agent.respond(transcript.text, spk, on_tool=on_tool)
    speak_text = await _drive_agent_turn(ws, state, run)
    total_ms = int((time.monotonic() - t0) * 1000)
    log.info("%s turn: %d ms total, %r", "ha" if ha_hit else "agent", total_ms, speak_text[:120])
    publish_turn({
        "ts": time.time(),
        "transcript": transcript.text,
        "follow_up": follow_up_turn,
        "path": "ha" if ha_hit else "llm",
        "tools": tool_calls,
        "reply": speak_text,
        "stt_ms": transcript.latency_ms,
        "ha_ms": fast.latency_ms if fast is not None else None,
        "total_ms": total_ms,
    })

    state.last_activity_s = time.monotonic()

    # If the agent chose to stay silent (typical on a follow-up that
    # wasn't directed at us), close out — no further window. Same for an
    # explicit end_conversation: the user said goodbye, so holding the mic
    # open for another FOLLOW_UP_WINDOW_S is exactly what they didn't ask for.
    if agent.conversation_ended:
        log.info("agent ended the conversation — re-arming wakeword")
    elif speak_text.strip():
        await _open_follow_up_window(ws, state)
    else:
        log.info("agent stayed silent — ending conversation, re-arming wakeword")


async def _open_follow_up_window(ws: ServerConnection, state: ConnState) -> None:
    """Tell the firmware to keep streaming mic audio (wakeword paused),
    arm the brain for VAD-driven capture, and schedule a timeout that
    closes the window if no speech arrives. Idempotent — cancels any
    previous pending timeout first."""
    # Wait out any TTS still playing on the device before reopening the
    # mic, plus a small guard, so the robot doesn't capture the tail of
    # its own voice and reply to itself.
    guard = get_config().get("FOLLOW_UP_GUARD_S")
    residual = state.est_playback_end_s - time.monotonic()
    wait = residual + guard
    if wait > 0:
        log.info("follow-up: waiting %.2fs for playback to finish", wait)
        await asyncio.sleep(wait)

    _cancel_follow_up_timeout(state)

    state.speech_buf = bytearray()
    state.voiced_ms = 0
    state.trailing_silence_ms = 0
    state.started_at = time.monotonic()
    state.listening = True
    state.follow_up = True
    await ws.send(json.dumps({"cmd": "start_listening"}))
    log.info("follow-up window opened (%.1fs)", get_config().get("FOLLOW_UP_WINDOW_S"))

    # Through spawn() even though we track and cancel this one ourselves: it
    # ends in a `ws.send` on a socket that may have died since, and a bare
    # create_task turns that ConnectionClosed into a context-free "Task
    # exception was never retrieved" at GC time.
    state.follow_up_timeout = spawn(
        _follow_up_timeout_task(ws, state), "follow_up_timeout"
    )


async def _follow_up_timeout_task(
    ws: ServerConnection, state: ConnState
) -> None:
    """Sleep the window, then if no voiced speech has been captured yet,
    close the window and re-arm the wakeword. If the user did start
    speaking, the existing VAD path handles end-of-utterance and this
    task is cancelled by respond() before this branch runs."""
    try:
        await asyncio.sleep(get_config().get("FOLLOW_UP_WINDOW_S"))
    except asyncio.CancelledError:
        return
    if state.voiced_ms >= get_config().get("SPEECH_LEAD_MS"):
        # User started talking — let the normal VAD path finish.
        return
    log.info("follow-up window timed out (no speech) — closing")
    state.listening = False
    state.follow_up = False
    state.speech_buf = bytearray()
    state.voiced_ms = 0
    state.trailing_silence_ms = 0
    await ws.send(json.dumps({"cmd": "stop_listening"}))


def _cancel_follow_up_timeout(state: ConnState) -> None:
    if state.follow_up_timeout is not None and not state.follow_up_timeout.done():
        state.follow_up_timeout.cancel()
    state.follow_up_timeout = None


def _should_sleep(state: ConnState) -> bool:
    """True when the inactivity timeout has elapsed and we're idle. A
    SLEEP_TIMEOUT_S of 0 disables sleeping entirely."""
    if state.asleep or state.listening or state.speaking:
        return False
    timeout = get_config().get("SLEEP_TIMEOUT_S")
    if not timeout or timeout <= 0:
        return False
    # Hold off sleeping while a BLE buddy approve prompt is waiting on the
    # device — the brain's idle timer is otherwise blind to it.
    timeout = effective_sleep_timeout(
        timeout,
        get_config().get("BUDDY_PROMPT_SLEEP_TIMEOUT_S"),
        state.buddy_prompt_pending,
    )
    return time.monotonic() - state.last_activity_s >= timeout


IDLE_CHECK_INTERVAL_S = 5.0


# A conversation that never goes quiet for SUMMARIZE_IDLE_S would otherwise
# never fold, and an ever-growing backlog ends in an over-length 400 whose
# recovery deletes the whole unsummarized history. Past this multiple of
# SUMMARIZE_TRIGGER, fold between turns anyway.
SUMMARIZE_OVERFLOW_FACTOR = 2
SUMMARIZE_RETRY_S = 60.0


def _should_summarize(state: ConnState) -> bool:
    """Fold once per idle stretch, SUMMARIZE_IDLE_S after the last turn — or
    between turns if the backlog has overflowed. Never while a mic is open or
    a fold is already running."""
    if state.listening or state.speaking:
        return False
    if state.summarize_task is not None and not state.summarize_task.done():
        return False
    cfg = get_config()
    now = time.monotonic()
    trigger = int(cfg.get("SUMMARIZE_TRIGGER"))
    if memory.unsummarized_count() >= trigger * SUMMARIZE_OVERFLOW_FACTOR:
        return now - state.last_summarize_s >= SUMMARIZE_RETRY_S
    if state.summarized_for_activity_s == state.last_activity_s:
        return False
    return now - state.last_activity_s >= float(cfg.get("SUMMARIZE_IDLE_S"))


async def _idle_ticker(ws: ServerConnection, state: ConnState) -> None:
    """Housekeeping that runs between conversations, on its own clock:
    idle → sleep, and folding the turn backlog into a summary. Cancelled by
    handle() on disconnect."""
    while True:
        await asyncio.sleep(IDLE_CHECK_INTERVAL_S)
        if _should_summarize(state):
            state.summarized_for_activity_s = state.last_activity_s
            state.last_summarize_s = time.monotonic()
            # A fresh connection (reconnect, brain restart) has no session
            # until its first turn; the backlog it inherited still needs
            # folding, so make one.
            agent = ensure_agent(ws, state)
            state.summarize_task = spawn(maybe_summarize(agent), "summarize")
        if _should_sleep(state):
            await go_to_sleep(ws, state)


async def go_to_sleep(ws: ServerConnection, state: ConnState) -> None:
    """Enter sleep: tell the firmware to turn the screen off (it sets a
    sleepy face first). The wake word and head tap stay armed on the
    firmware as the only way out."""
    state.asleep = True
    # Persist so a brain restart while asleep resumes in the asleep state
    # rather than treating a still-dark firmware screen as awake.
    memory.set_runtime_state("asleep", True)
    log.info("sleeping (idle %.0fs)", time.monotonic() - state.last_activity_s)
    try:
        await ws.send(json.dumps({"cmd": "sleep"}))
    except Exception:
        log.exception("sleep cmd send failed")


def wake_up(state: ConnState) -> None:
    """Clear the sleep state on a wake word / tap. The firmware relights
    its own screen locally on the same trigger (instant, offline-safe), so
    no wake command is sent from here — we just clear the flag."""
    if state.asleep:
        log.info("waking")
    state.asleep = False
    memory.set_runtime_state("asleep", False)


def _on_wake_trigger(state: ConnState) -> None:
    """Shared handling for a wake word OR a head tap: wake if asleep and arm
    a listening capture. The firmware has already transitioned itself to
    LISTENING (and relit the screen), so the brain only sets up VAD state."""
    _cancel_follow_up_timeout(state)
    wake_up(state)
    state.listening = True
    state.follow_up = False
    state.speech_buf = bytearray()
    state.voiced_ms = 0
    state.trailing_silence_ms = 0
    state.started_at = time.monotonic()
    state.last_activity_s = time.monotonic()


async def handle(ws: ServerConnection) -> None:
    log.info("esp32 connected: %s", ws.remote_address)
    # Firmware turn-state recovery (brain-only; "Fix C"). If the brain was
    # killed mid-turn, the firmware is stranded in LISTENING/SPEAKING — which
    # pauses the wakeword and gates out head-tap (the firmware gates tap to
    # IDLE), leaving the device unresponsive until a manual reboot even though
    # it auto-reconnects. A new WS connection means no turn can be in progress,
    # so force the firmware back to IDLE: stop_speaking transitions to IDLE from
    # any mode (re-arming the wakeword, re-enabling tap) and leaves the screen
    # untouched, so it's a no-op on a fresh boot and safe while asleep. The
    # durable self-heal (firmware → IDLE on WS disconnect, "Fix A") is queued
    # for the next reflash.
    await ws.send(json.dumps({"cmd": "stop_speaking"}))
    # Rocky mode is gone, but a firmware that was showing the Rocky skin when
    # the old brain went away keeps it until told otherwise.
    await ws.send(json.dumps({"cmd": "set_skin", "value": "default"}))
    state = ConnState()
    # Seed the sleep clock at connect so a fresh link doesn't immediately
    # sleep before any interaction.
    state.last_activity_s = time.monotonic()
    # Restore the persisted sleep flag: if the device was asleep when the
    # brain last ran (or restarted), stay dormant and let only a wake word /
    # head tap (which the firmware lights locally) bring it back. The firmware is still backlit-off from its earlier
    # `sleep`, so the two stay consistent without sending any command.
    if bool(memory.get_runtime_state("asleep", False)):
        state.asleep = True
        log.info("restored sleep state on connect: asleep")
    idle_ticker = spawn(_idle_ticker(ws, state), "idle_ticker")
    try:
        async for msg in ws:
            if isinstance(msg, bytes) and msg and msg[0] == OP_AUDIO:
                if not state.listening:
                    continue
                frame = msg[1:]
                if len(frame) % 2:
                    # np.frombuffer(int16) raises on a partial sample, which
                    # would escape the message loop and drop the connection.
                    # Our firmware always sends whole frames, so this is a
                    # malformed peer on the open port, not us.
                    log.warning("dropping odd-length audio frame (%d bytes)",
                                len(frame))
                    continue
                state.speech_buf.extend(frame)
                cfg = get_config()
                rms = frame_rms(frame)
                if rms >= cfg.get("SPEECH_RMS"):
                    state.voiced_ms += FRAME_MS
                    state.trailing_silence_ms = 0
                else:
                    state.trailing_silence_ms += FRAME_MS
                    # Decay un-established speech: if a full silence-tail
                    # passes without ever reaching the speech-lead threshold,
                    # the voiced frames so far were noise (a creak, a distant
                    # voice, the TTS tail) — not a real utterance onset. Drop
                    # them so scattered blips can't accumulate to the lead
                    # threshold and falsely end an otherwise-silent follow-up
                    # window seconds early (the window's own timer should
                    # govern when the user stays quiet). Once real speech
                    # establishes (voiced_ms >= lead) this no longer fires and
                    # end_by_silence below governs end-of-utterance as before.
                    if (
                        state.voiced_ms < cfg.get("SPEECH_LEAD_MS")
                        and state.trailing_silence_ms >= cfg.get("SILENCE_TAIL_MS")
                    ):
                        state.voiced_ms = 0
                        state.trailing_silence_ms = 0

                elapsed_ms = int((time.monotonic() - state.started_at) * 1000)

                end_by_silence = (
                    state.voiced_ms >= cfg.get("SPEECH_LEAD_MS")
                    and state.trailing_silence_ms >= cfg.get("SILENCE_TAIL_MS")
                )
                end_by_timeout = elapsed_ms >= cfg.get("MAX_UTTERANCE_MS")

                if end_by_silence or end_by_timeout:
                    log.info(
                        "utterance end: %s (voiced=%d ms, tail=%d ms, total=%d ms)",
                        "silence" if end_by_silence else "timeout",
                        state.voiced_ms,
                        state.trailing_silence_ms,
                        elapsed_ms,
                    )
                    await respond(ws, state)
            elif isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    log.warning("bad json from esp32: %r", msg[:120])
                    continue
                log.info("event: %s", payload)
                event = payload.get("event")
                if event in ("wakeword", "tap"):
                    # Wake word or head tap: wake (if asleep) and start a
                    # listening capture, overriding any follow-up window. The
                    # firmware has already switched itself to LISTENING and
                    # relit the screen on the same trigger.
                    _on_wake_trigger(state)
                elif event == "buddy_prompt":
                    # The firmware's BLE buddy reports whether a permission
                    # prompt is waiting on the device, so _should_sleep can
                    # elongate the idle timeout while it's pending.
                    state.buddy_prompt_pending = bool(payload.get("pending"))
                    log.info("buddy prompt pending: %s", state.buddy_prompt_pending)
            elif isinstance(msg, bytes):
                # Any other opcode is ignored. A zero-length binary frame is
                # legal WebSocket and lands here too (the audio branch
                # requires a non-empty msg). Log arguments are evaluated
                # eagerly, so an unguarded msg[0] would IndexError out of the
                # message loop and tear the connection down.
                log.debug(
                    "unknown binary frame: opcode %s, %d bytes",
                    f"0x{msg[0]:02x}" if msg else "none", len(msg),
                )
            else:
                log.warning("unexpected frame type: %r", type(msg).__name__)
    except Exception:
        log.exception("connection error")
    finally:
        if state.agent is not None:
            live_sessions.discard(state.agent)
        # The follow-up timer outlives the socket otherwise: it wakes up to
        # FOLLOW_UP_WINDOW_S later, mutates a ConnState nothing owns any more
        # (pinning its AgentSession and speech buffer alive with it) and then
        # sends stop_listening down a dead connection. An in-flight turn is
        # deliberately left to finish — cancelling it at an arbitrary await is
        # how the M6.1 half-written tool_use corruption happens.
        _cancel_follow_up_timeout(state)
        idle_ticker.cancel()
        log.info("esp32 disconnected")


def lan_ip() -> str:
    """The address the ESP32 can actually reach us at.

    Not `gethostbyname(gethostname())`: on Debian/Ubuntu that resolves the
    hostname through /etc/hosts, which maps it to the 127.0.1.1 loopback
    alias. Advertising that over mDNS points the firmware at its own
    loopback. Connecting a UDP socket sends no packets — it only asks the
    routing table which source address would be used to reach the outside
    world, which is exactly the interface the ESP32 is on.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 53))
        return s.getsockname()[0]
    finally:
        s.close()


def _console_bind() -> str:
    """Interface for the web console.

    The LAN address, not 0.0.0.0: the console is an admin surface (its MCP
    tab persists a command line the brain then executes as this user), so it
    has no business being offered on every interface the Jetson happens to
    have — a VPN/tunnel one included. CONSOLE_BIND overrides, e.g. 127.0.0.1
    for ssh-tunnel-only access or 0.0.0.0 to go back to everything. Note the
    LAN bind means `curl localhost:8080` on the Jetson no longer works; use
    the LAN address.
    """
    override = (os.environ.get("CONSOLE_BIND") or "").strip()
    if override:
        return override
    try:
        return lan_ip()
    except OSError as exc:
        # No default route yet (DHCP still coming up under a user unit that
        # can't order on network-online.target). Falling back to every
        # interface keeps the console reachable once the link is up; the
        # token is what actually guards it.
        log.warning("no LAN address yet (%s) — console binds %s", exc, HOST)
        return HOST


def _console_token() -> tuple[str, bool]:
    """(token, was_generated) for the web console.

    From CONSOLE_TOKEN in the repo-root .env, which survives the deploy
    script's `git reset --hard` (it's untracked). Absent, mint one for this
    run rather than serving the console open — a per-run secret that gets
    logged is still a secret, and it means a Jetson that has never been
    configured is protected but not locked out of its own console.
    """
    configured = (os.environ.get("CONSOLE_TOKEN") or "").strip()
    if configured:
        return configured, False
    return secrets.token_urlsafe(16), True


async def advertise_mdns() -> tuple[AsyncZeroconf, ServiceInfo] | tuple[None, None]:
    """Register stackchan-brain.local via zeroconf.

    Async on purpose. The synchronous `Zeroconf` API deadlocks when it is
    constructed from inside a running event loop: it attaches to *our* loop,
    then blocks the calling thread on `run_coroutine_threadsafe(...).result()`
    against that same loop, and every registration dies with EventLoopBlocked
    ~10 s later. AsyncZeroconf drives the same machinery through our loop and
    registers in about a second.

    Still non-fatal. On macOS port 5353 is held by the system mdnsresponder
    and registration can genuinely time out — for local testing the firmware
    can be pointed at the host's existing `.local` hostname directly.
    """
    aiozc = None
    try:
        ip = lan_ip()
        # Bind the LAN interface only, so we don't also announce on lo and
        # docker0 (nothing the ESP32 can use, and it clutters avahi-browse).
        aiozc = AsyncZeroconf(interfaces=[ip])
        info = ServiceInfo(
            type_="_ws._tcp.local.",
            name=f"{MDNS_NAME}._ws._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=PORT,
            server=f"{MDNS_NAME}.local.",
        )
        await aiozc.async_register_service(info)
        log.info("mDNS: advertising %s.local at %s:%d", MDNS_NAME, ip, PORT)
        return aiozc, info
    except Exception as exc:
        # Close on the way out. A half-registered responder keeps answering
        # queries for the rest of the process's life, so leaking it here means
        # serving an address we just decided was unusable — which is how a
        # failed registration still managed to publish 127.0.1.1.
        if aiozc is not None:
            try:
                await aiozc.async_close()
            except Exception:
                log.debug("zeroconf close after failed registration", exc_info=True)
        log.warning(
            "mDNS registration failed (%s: %s). The brain still listens on :%d; "
            "point the firmware at the host's existing .local hostname.",
            type(exc).__name__,
            exc,
            PORT,
        )
        return None, None


def _seed_default_mcp_servers() -> None:
    """Register the bundled weather server once (empty registry). Uses the
    running interpreter and an absolute script path so it works regardless
    of cwd."""
    if memory.list_mcp_servers():
        return
    srv_dir = Path(__file__).parent / "mcp_servers"
    memory.add_mcp_server(
        "weather", "stdio", sys.executable,
        args=[str(srv_dir / "weather.py")], enabled=True,
    )
    log.info("seeded default MCP server: weather")


async def main() -> None:
    global stt, tts
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Load persisted config overrides, then apply the restart-bound knobs
    # (TTS voice, STT device/compute/model). These objects are lazy — no
    # model is loaded until first use — so reconstructing here is cheap and
    # picks up any web-UI override saved on a previous run.
    cfg = init_config(memory)
    # M6.5: heal any durable conversation-state corruption (dangling tool_use
    # from a pre-M6.1 crash, etc.) before the first turn replays it.
    repair_memory(memory)
    tts = Synthesizer(voice=cfg.get("PIPER_VOICE"))
    stt = Transcriber(
        model_name=cfg.get("STT_MODEL"),
        device=cfg.get("STT_DEVICE"),
        compute_type=cfg.get("STT_COMPUTE_TYPE"),
    )
    # Pull the whisper load off the first utterance. The Jetson restarts the
    # brain on every deploy, so "first utterance" is a routine event, and the
    # load is seconds of dead air on top of a turn the user is waiting on.
    # Background, not awaited: the socket should be accepting connections
    # while the model comes up.
    spawn(stt.warm(), "stt_warm")

    # MCP servers (Phase 9b): seed the local weather server on first run so
    # it works out of the box. Then connect — best-effort, a down server just
    # contributes no tools.
    _seed_default_mcp_servers()
    await mcp_client.start()

    # Web console: tee brain.* logs to the live feed and serve the
    # FastAPI app in-process on WEB_PORT, sharing memory + config + mcp.
    loop = asyncio.get_running_loop()
    LOGS.bind_loop(loop)
    TURNS.bind_loop(loop)
    logging.getLogger("brain").addHandler(WebUILogHandler())
    console_token, console_generated = _console_token()
    web_host = _console_bind()
    web = uvicorn.Server(
        uvicorn.Config(
            create_app(memory, cfg, mcp_client,
                       token=console_token,
                       resync_sessions=resync_live_sessions),
            host=web_host, port=WEB_PORT, loop="none", log_level="warning",
        )
    )
    web_task = asyncio.create_task(web.serve())

    aiozc, info = await advertise_mdns()
    try:
        # ping_interval=None: the 78/esp-ml307 WebSocket on the firmware
        # doesn't reply to pings, so server-side keepalive trips the
        # connection every ~50 s. We accept the lost dead-conn detection.
        async with serve(
            handle, HOST, PORT, max_size=2**20, ping_interval=None
        ):
            log.info("brain listening on ws://%s:%d", HOST, PORT)
            if console_generated:
                # Nobody can reach the console without this, and a fresh
                # Jetson has no CONSOLE_TOKEN yet — so print the URL that
                # hands the token to the browser (it stores it and drops it
                # from the address bar).
                log.warning(
                    "web console on http://%s:%d/#token=%s — no CONSOLE_TOKEN "
                    "in .env, so this one is good only until the next restart",
                    web_host, WEB_PORT, console_token,
                )
            else:
                log.info("web console on http://%s:%d (token from .env)",
                         web_host, WEB_PORT)
            await asyncio.Future()
    finally:
        web.should_exit = True
        await web_task
        await mcp_client.aclose()
        await ha_fast_path.aclose()
        if aiozc is not None and info is not None:
            await aiozc.async_unregister_service(info)
            await aiozc.async_close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # systemd stops us with SIGINT (deploy/stackchan-brain.service) so that
        # main()'s finally can unregister mDNS and close the MCP client.
        # Letting KeyboardInterrupt escape would exit non-zero and leave the
        # unit sitting in `failed` after every ordinary stop.
        log.info("interrupted — shut down cleanly")
