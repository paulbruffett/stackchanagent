"""WebSocket server the M5StackChan ESP32 connects to.

Phase 2: wakeword → LISTENING (accumulate PCM, RMS-based silence VAD) →
STT → TTS → SPEAKING (stream PCM back) → IDLE. Driven from a per-
connection state machine.

Wire protocol:
  - Binary frame, first byte = opcode:
      0x01  PCM audio frame (16 kHz, mono, s16le, 20 ms = 640 bytes)
      0x02  JPEG camera frame (phase 4+)
  - Text frame: JSON control message, both directions.
      from ESP32: {"event": "boot"|"wakeword"|"vad_end", ...}
      to   ESP32: {"cmd": "stop_listening"|"start_speaking"|"stop_speaking"|
                          "set_expression"|"look_at"|"set_motion_rate"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import uvicorn
from dotenv import load_dotenv
from websockets.asyncio.server import ServerConnection, serve
from zeroconf import ServiceInfo, Zeroconf

# Load ANTHROPIC_API_KEY (and any other env) from the project root .env
# before importing the agent module (which constructs the Anthropic client).
load_dotenv(Path(__file__).parent.parent / ".env")

from behavior import IdleBehavior
from claude_agent import AgentSession, repair_memory
from config import get_config, init_config
from a2a_client import A2aClient
from mcp_client import McpClient
from policy import effective_sleep_timeout, skin_for_rocky_mode
from memory import Memory
from stt import Transcriber, should_drop_follow_up
from tasks import spawn
from tts import Synthesizer
from tts_hume import HumeSynthesizer
from vision import FaceDetector, FaceTracker
from webui.app import create_app
from webui.logbuf import LOGS, TURNS, WebUILogHandler, publish_turn

HOST = "0.0.0.0"
PORT = 8765
WEB_PORT = 8080
MDNS_NAME = "stackchan-brain"

OP_AUDIO = 0x01
OP_JPEG = 0x02

# Audio assumptions (must match firmware).
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2                  # 640

# VAD / greeting / follow-up / detect-cadence knobs are now hot-editable
# via config.py (web UI). Read with get_config().get("SPEECH_RMS") etc.
# at the use sites below. Defaults live in config.py::SPECS.
#
# DETECT_INTERVAL_LOOK_AROUND_S stays a constant — during a sweep we want
# detection on every frame, and 0.0 ("every frame") is not a useful knob.
DETECT_INTERVAL_LOOK_AROUND_S = 0.0    # 0 = every frame (~1.5 s)

log = logging.getLogger("brain")

# Lazy globals — one model load per process.
stt = Transcriber()
tts = Synthesizer()
# Rocky-mode cloud voice (Milestone 4). Lazy — no HTTP client until first
# use, and reports unavailable when no Hume key/voice is in .env, so this is
# harmless to construct even with no Hume account.
hume_tts = HumeSynthesizer()
# Debounce so a Rocky-mode-without-Hume run logs the degradation once, not
# per sentence.
_rocky_no_hume_warned = False
# Rolling Hume reliability tally (since process start). Reported on every
# fallback so the real flake rate is visible from the log without a separate
# metric — decide whether the mid-reply voice-flip is frequent enough to
# mitigate off data, not off a single 500.
_hume_ok = 0
_hume_fallback = 0


def _synthesize(sentence: str, use_rocky: bool, speed: float) -> bytes:
    """Synthesize one sentence to 16 kHz PCM in a worker thread. In Rocky
    mode with a Hume voice configured, use Hume; any Hume failure (network,
    quota, malformed stream) falls back to Piper for that sentence so audio
    is never dropped. Without Rocky mode — or without a Hume key — this is
    plain Piper."""
    global _rocky_no_hume_warned, _hume_ok, _hume_fallback
    if use_rocky:
        if hume_tts.available:
            try:
                pcm = hume_tts.synthesize(sentence, speed=speed)
                _hume_ok += 1
                return pcm
            except Exception:
                _hume_fallback += 1
                total = _hume_ok + _hume_fallback
                log.warning(
                    "hume tts failed; falling back to piper "
                    "(fallbacks %d/%d = %.1f%%)",
                    _hume_fallback, total, 100.0 * _hume_fallback / total,
                    exc_info=True,
                )
        elif not _rocky_no_hume_warned:
            log.warning(
                "ROCKY_MODE on but no Hume voice configured — persona active, "
                "voice stays Piper (set HUME_API_KEY + a voice in .env)"
            )
            _rocky_no_hume_warned = True
    return tts.synthesize(sentence)
# Single shared Memory across all WS connections, so the robot
# remembers conversations even after a disconnect/reconnect or process
# restart. Sqlite handles file locking; only one process should write.
memory = Memory()
# Shared MCP client (Phase 9b): one set of server connections for the
# whole process. Started in main(); tools merged into every agent turn.
mcp_client = McpClient(memory)
# Shared A2A client (Phase 9c): connects to Agent2Agent servers (e.g.
# Hermes) and surfaces their sub-agents as delegation tools in every turn.
a2a_client = A2aClient(memory)


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
    latest_jpeg: bytes | None = None
    face_detector: FaceDetector | None = None
    face_tracker: FaceTracker = field(default_factory=FaceTracker)
    behavior: IdleBehavior = field(default_factory=IdleBehavior)
    last_user_interaction_s: float = 0.0
    last_greeting_s: float = 0.0
    last_detect_s: float = 0.0
    # True while asleep: screen is off, look-around and face detection are
    # suspended. Cleared by a wake word or head tap. Driven by SLEEP_TIMEOUT_S.
    asleep: bool = False
    # monotonic time of the last interaction that should keep the device
    # awake (conversation, wake word, tap). Seeded at connect so a fresh
    # connection doesn't immediately sleep. Distinct from
    # last_user_interaction_s (which gates greetings) so sleep timing and
    # greeting suppression stay independent.
    last_activity_s: float = 0.0
    # monotonic time the most recent TTS playback is expected to finish on
    # the device. The brain sends audio faster than real time, so when the
    # speaker worker returns the device still has buffered audio playing;
    # we wait until past this before reopening the mic (else the robot
    # hears its own voice tail and replies to itself).
    est_playback_end_s: float = 0.0
    # Set while a vision/detection task is in flight, so we drop overlapping
    # frames rather than queuing detections behind a slow mediapipe call.
    detecting: bool = False
    # The avatar skin last pushed to the firmware ("rocky"/"default"). Tracks
    # ROCKY_MODE; synced on connect and whenever it changes (see
    # _maybe_sync_skin). None until the first sync on connect.
    last_skin: str | None = None
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
        state.agent = AgentSession(
            ws,
            memory=memory,
            get_latest_jpeg=lambda: state.latest_jpeg,
            on_external_head_move=state.behavior.notify_head_moved,
            mcp=mcp_client,
            a2a=a2a_client,
        )
    return state.agent


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
    # Pick the TTS backend once per turn (Rocky mode → Hume when configured).
    # Read here in the event loop, not in the worker thread, per the config
    # single-loop contract.
    cfg = get_config()
    use_rocky = bool(cfg.get("ROCKY_MODE"))
    speed = float(cfg.get("ROCKY_SPEED"))
    try:
        while True:
            sentence = await queue.get()
            if sentence is None:
                break
            if not started:
                await ws.send(json.dumps({"cmd": "start_speaking"}))
                started = True
            t0 = time.monotonic()
            tts_pcm = await asyncio.to_thread(_synthesize, sentence, use_rocky, speed)
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
        _dump_capture(pcm)
    transcript = stt.transcribe(pcm)
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
    agent.on_tool = lambda name, inp: tool_calls.append({"name": name, "input": inp})
    t0 = time.monotonic()
    try:
        if follow_up_turn:
            run = lambda spk: agent.respond_follow_up(transcript.text, spk)
        else:
            run = lambda spk: agent.respond(transcript.text, spk)
        speak_text = await _drive_agent_turn(ws, state, run)
    finally:
        agent.on_tool = None
    total_ms = int((time.monotonic() - t0) * 1000)
    log.info("agent turn: %d ms total, %r", total_ms, speak_text[:120])
    publish_turn({
        "ts": time.time(),
        "transcript": transcript.text,
        "follow_up": follow_up_turn,
        "tools": tool_calls,
        "reply": speak_text,
        "stt_ms": transcript.latency_ms,
        "total_ms": total_ms,
    })

    state.last_user_interaction_s = time.monotonic()
    state.last_activity_s = time.monotonic()

    # If the agent chose to stay silent (typical on a follow-up that
    # wasn't directed at us), close out — no further window.
    if speak_text.strip():
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

    state.follow_up_timeout = asyncio.create_task(
        _follow_up_timeout_task(ws, state)
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
    # Don't drop the screen mid-sweep, or the head would keep moving while
    # "asleep". A sweep is short (~16 s) and infrequent; just wait it out.
    if state.behavior.look_around_in_progress:
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


async def _maybe_sync_skin(ws: ServerConnection, state: ConnState) -> None:
    """Push the avatar skin to the firmware if ROCKY_MODE has changed since the
    last sync. ROCKY_MODE is the single source of truth (voice + face); this
    catches both the voice tool (set_persona_mode) and a web-console toggle
    with no callback machinery. Cheap to call on every camera frame."""
    want = skin_for_rocky_mode(get_config().get("ROCKY_MODE"))
    if want == state.last_skin:
        return
    try:
        await ws.send(json.dumps({"cmd": "set_skin", "value": want}))
    except Exception:
        # Leave last_skin unchanged so the next frame retries rather than
        # latching a skin we never actually pushed.
        log.exception("set_skin send failed")
        return
    state.last_skin = want
    log.info("skin → %s", want)


async def go_to_sleep(ws: ServerConnection, state: ConnState) -> None:
    """Enter sleep: tell the firmware to turn the screen off (it sets a
    sleepy face first), and stop look-around + face detection. The wake
    word and head tap stay armed on the firmware as the only way out."""
    state.asleep = True
    # Persist so a brain restart while asleep resumes in the asleep state
    # instead of re-running autonomous behavior (look-around / face-detect
    # greet) against a still-dark firmware screen.
    memory.set_runtime_state("asleep", True)
    log.info("sleeping (idle %.0fs)", time.monotonic() - state.last_activity_s)
    try:
        await ws.send(json.dumps({"cmd": "sleep"}))
    except Exception:
        log.exception("sleep cmd send failed")


def wake_up(state: ConnState) -> None:
    """Clear the sleep state on a wake word / tap. The firmware relights
    its own screen locally on the same trigger (instant, offline-safe), so
    no wake command is sent from here — we just resume brain-side behavior
    and reset the look-around clock so a sweep doesn't fire immediately."""
    if state.asleep:
        log.info("waking")
    state.asleep = False
    memory.set_runtime_state("asleep", False)
    state.behavior.last_look_around_s = time.monotonic()


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
    now = time.monotonic()
    state.last_user_interaction_s = now
    state.last_activity_s = now


async def proactive_greet(ws: ServerConnection, state: ConnState) -> None:
    """Run an agent turn triggered by a non-speech event (new face seen).
    Bypasses STT and uses a stage-direction message instead."""
    now = time.monotonic()
    if state.listening or state.speaking:
        log.info("greet skipped: busy (listening=%s speaking=%s)",
                 state.listening, state.speaking)
        return
    if now - state.last_greeting_s < get_config().get("GREETING_COOLDOWN_S"):
        log.info("greet skipped: cooldown (%.0fs since last)",
                 now - state.last_greeting_s)
        return
    if now - state.last_user_interaction_s < get_config().get("RECENT_INTERACTION_S"):
        log.info("greet skipped: recent interaction (%.0fs ago)",
                 now - state.last_user_interaction_s)
        return

    state.last_greeting_s = now
    log.info("proactive greeting: new face")

    agent = ensure_agent(ws, state)
    tool_calls: list[dict[str, Any]] = []
    agent.on_tool = lambda name, inp: tool_calls.append({"name": name, "input": inp})
    t0 = time.monotonic()
    try:
        speak_text = await _drive_agent_turn(
            ws,
            state,
            lambda spk: agent.respond_to_event(
                "A new person just appeared in front of you. Greet them in one "
                "short, friendly sentence.",
                spk,
            ),
        )
    finally:
        agent.on_tool = None
    total_ms = int((time.monotonic() - t0) * 1000)
    log.info("greet turn: %d ms, %r", total_ms, speak_text[:120])
    publish_turn({
        "ts": time.time(),
        "transcript": "[new face — proactive greeting]",
        "follow_up": False,
        "tools": tool_calls,
        "reply": speak_text,
        "stt_ms": None,
        "total_ms": total_ms,
    })


async def process_latest_jpeg(ws: ServerConnection, state: ConnState) -> None:
    """Run face detection + gaze update + proactive-greeting check on the
    most recent JPEG. Skipped if a previous detection is still running,
    so a slow frame doesn't queue up backlogged work — we always look at
    the newest available frame instead."""
    if state.detecting:
        return
    jpeg = state.latest_jpeg
    if jpeg is None:
        return

    if state.face_detector is None:
        try:
            state.face_detector = FaceDetector()
        except Exception:
            log.exception("FaceDetector init failed — disabling vision")
            return

    state.detecting = True
    try:
        faces = await asyncio.to_thread(state.face_detector.detect, jpeg)
    except Exception:
        log.exception("face detect failed")
        state.detecting = False
        return
    state.detecting = False

    new_face = state.face_tracker.update(faces)

    in_conversation = state.listening or state.speaking
    await state.behavior.tick(ws, faces, in_conversation)

    if new_face:
        spawn(proactive_greet(ws, state), "proactive_greet")


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
    state = ConnState()
    # Seed the sleep clock at connect so a fresh link doesn't immediately
    # sleep before any interaction.
    state.last_activity_s = time.monotonic()
    # Restore the persisted sleep flag: if the device was asleep when the
    # brain last ran (or restarted), stay dormant — keep look-around and
    # face detection suppressed so we don't act against a dark screen — and
    # let only a wake word / head tap (which the firmware lights locally)
    # bring it back. The firmware is still backlit-off from its earlier
    # `sleep`, so the two stay consistent without sending any command.
    if bool(memory.get_runtime_state("asleep", False)):
        state.asleep = True
        log.info("restored sleep state on connect: asleep")
    # Sync the avatar skin to the current ROCKY_MODE on connect (boot brings up
    # the default skin; this flips it to Rocky if ROCKY_MODE is on).
    await _maybe_sync_skin(ws, state)
    try:
        async for msg in ws:
            if isinstance(msg, bytes) and msg and msg[0] == OP_AUDIO:
                if not state.listening:
                    continue
                frame = msg[1:]
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
            elif isinstance(msg, bytes) and msg and msg[0] == OP_JPEG:
                jpeg = bytes(msg[1:])
                log.debug("jpeg frame, %d bytes", len(jpeg))
                # Always keep the latest frame around for describe_view,
                # but only RUN detection at the configured cadence.
                state.latest_jpeg = jpeg
                # Frames are the steady heartbeat (~every 1.5 s, even while
                # asleep), so use them to push a ROCKY_MODE skin change.
                await _maybe_sync_skin(ws, state)
                # While asleep: no look-around, no face detection — the wake
                # word / tap is the only way out. Frames keep arriving (cheap)
                # so describe_view still has a recent one after waking.
                if state.asleep:
                    continue
                # Awake + idle long enough → sleep. Checked on each frame
                # (~every 1.5 s), which is a fine cadence for a minutes-scale
                # timeout.
                if _should_sleep(state):
                    await go_to_sleep(ws, state)
                    continue
                interval = (
                    DETECT_INTERVAL_LOOK_AROUND_S
                    if state.behavior.look_around_in_progress
                    else get_config().get("DETECT_INTERVAL_IDLE_S")
                )
                if time.monotonic() - state.last_detect_s >= interval:
                    state.last_detect_s = time.monotonic()
                    spawn(process_latest_jpeg(ws, state), "process_jpeg")
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
                log.warning(
                    "unknown binary opcode 0x%02x, %d bytes", msg[0], len(msg)
                )
            else:
                log.warning("unexpected frame type: %r", type(msg).__name__)
    except Exception:
        log.exception("connection error")
    finally:
        log.info("esp32 disconnected")


def advertise_mdns() -> tuple[Zeroconf, ServiceInfo] | tuple[None, None]:
    """Register stackchan-brain.local via zeroconf.

    On macOS, port 5353 is held by the system mdnsresponder and zeroconf's
    register_service can time out. We treat that as non-fatal — for local
    testing the firmware can be pointed at the host's existing `.local`
    hostname (e.g. `Pauls-Mac-mini.local`) directly.
    """
    try:
        zc = Zeroconf()
        ip = socket.gethostbyname(socket.gethostname())
        info = ServiceInfo(
            type_="_ws._tcp.local.",
            name=f"{MDNS_NAME}._ws._tcp.local.",
            addresses=[socket.inet_aton(ip)],
            port=PORT,
            server=f"{MDNS_NAME}.local.",
        )
        zc.register_service(info)
        log.info("mDNS: advertising %s.local at %s:%d", MDNS_NAME, ip, PORT)
        return zc, info
    except Exception as exc:
        log.warning(
            "mDNS registration failed (%s). The brain still listens on :%d; "
            "point the firmware at the host's existing .local hostname.",
            exc,
            PORT,
        )
        return None, None


def _seed_default_mcp_servers() -> None:
    """Register the bundled weather + Hue servers once (empty registry).
    Uses the running interpreter and absolute script paths so it works
    regardless of cwd. Weather is enabled; Hue is disabled until the user
    sets HUE_BRIDGE_IP/HUE_TOKEN in .env and toggles it on."""
    if memory.list_mcp_servers():
        return
    srv_dir = Path(__file__).parent / "mcp_servers"
    memory.add_mcp_server(
        "weather", "stdio", sys.executable,
        args=[str(srv_dir / "weather.py")], enabled=True,
    )
    memory.add_mcp_server(
        "hue", "stdio", sys.executable,
        args=[str(srv_dir / "hue.py")], env_ref="HUE_TOKEN", enabled=False,
    )
    log.info("seeded default MCP servers: weather (on), hue (off)")


def _seed_default_a2a_servers() -> None:
    """Register the Hermes A2A endpoint once (empty registry), disabled
    until the user confirms the URL and toggles it on (like Hue)."""
    if memory.list_a2a_servers():
        return
    memory.add_a2a_server(
        "hermes", "http://192.168.4.30:8080", enabled=False,
    )
    log.info("seeded default A2A server: hermes (off)")


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

    # MCP servers (Phase 9b): seed the two local servers on first run so
    # weather works out of the box and Hue is one toggle + .env away.
    # Then connect — best-effort, a down server just contributes no tools.
    _seed_default_mcp_servers()
    await mcp_client.start()

    # A2A servers (Phase 9c): seed Hermes (disabled) on first run, then
    # connect any enabled endpoints — best-effort, like MCP.
    _seed_default_a2a_servers()
    await a2a_client.start()

    # Web console: tee brain.* logs to the live feed and serve the
    # FastAPI app in-process on WEB_PORT, sharing memory + config + mcp.
    loop = asyncio.get_running_loop()
    LOGS.bind_loop(loop)
    TURNS.bind_loop(loop)
    logging.getLogger("brain").addHandler(WebUILogHandler())
    web = uvicorn.Server(
        uvicorn.Config(
            create_app(memory, cfg, mcp_client, a2a_client),
            host=HOST, port=WEB_PORT, loop="none", log_level="warning",
        )
    )
    web_task = asyncio.create_task(web.serve())

    zc, info = advertise_mdns()
    try:
        # ping_interval=None: the 78/esp-ml307 WebSocket on the firmware
        # doesn't reply to pings, so server-side keepalive trips the
        # connection every ~50 s. We accept the lost dead-conn detection.
        async with serve(
            handle, HOST, PORT, max_size=2**20, ping_interval=None
        ):
            log.info("brain listening on ws://%s:%d", HOST, PORT)
            log.info("web console on http://%s:%d", HOST, WEB_PORT)
            await asyncio.Future()
    finally:
        web.should_exit = True
        await web_task
        await mcp_client.aclose()
        await a2a_client.aclose()
        if zc is not None and info is not None:
            zc.unregister_service(info)
            zc.close()


if __name__ == "__main__":
    asyncio.run(main())
