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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import numpy as np
from dotenv import load_dotenv
from websockets.asyncio.server import ServerConnection, serve
from zeroconf import ServiceInfo, Zeroconf

# Load ANTHROPIC_API_KEY (and any other env) from the project root .env
# before importing the agent module (which constructs the Anthropic client).
load_dotenv(Path(__file__).parent.parent / ".env")

from behavior import IdleBehavior
from claude_agent import AgentSession
from memory import Memory
from stt import Transcriber
from tts import Synthesizer
from vision import FaceDetector, FaceTracker

HOST = "0.0.0.0"
PORT = 8765
MDNS_NAME = "stackchan-brain"

OP_AUDIO = 0x01
OP_JPEG = 0x02

# Audio assumptions (must match firmware).
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 320
FRAME_BYTES = FRAME_SAMPLES * 2                  # 640

# VAD: a frame's RMS must clear SPEECH_RMS to be "voiced".
# Speech ends after SILENCE_TAIL_MS of consecutive sub-threshold frames,
# but only once we've actually heard SPEECH_LEAD_MS of voiced frames.
# 150 is a quiet-room threshold; bump it if the device picks up too much
# background hum and refuses to end utterances.
SPEECH_RMS = 150
SILENCE_TAIL_MS = 700
SPEECH_LEAD_MS = 200
MAX_UTTERANCE_MS = 10000

# Proactive greeting gates.
GREETING_COOLDOWN_S = 1800.0  # at most one greeting per 30 minutes per conn
RECENT_INTERACTION_S = 90.0   # silent if user has spoken in the last 90 s

# Face-detection cadence. The camera_pump on firmware fires every ~1.5 s
# regardless (so describe_view always has a recent frame), but we only
# RUN detection at these intervals to save CPU + avoid noisy tracking.
# During a look-around sweep we detect every frame, since the robot is
# actively looking around and a fresh detection should redirect it.
DETECT_INTERVAL_IDLE_S = 20.0          # default: 20 s between face checks
DETECT_INTERVAL_LOOK_AROUND_S = 0.0    # 0 = every frame (~1.5 s)

log = logging.getLogger("brain")

# Lazy globals — one model load per process.
stt = Transcriber()
tts = Synthesizer()
# Single shared Memory across all WS connections, so the robot
# remembers conversations even after a disconnect/reconnect or process
# restart. Sqlite handles file locking; only one process should write.
memory = Memory()


@dataclass
class ConnState:
    listening: bool = False
    speaking: bool = False
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
    # Set while a vision/detection task is in flight, so we drop overlapping
    # frames rather than queuing detections behind a slow mediapipe call.
    detecting: bool = False


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
            log.info(
                "tts: %d ms, %.2fs audio, %r",
                int((time.monotonic() - t0) * 1000),
                len(tts_pcm) / (SAMPLE_RATE * 2),
                sentence[:80],
            )
            await send_pcm_stream(ws, tts_pcm)
        if started:
            await ws.send(json.dumps({"cmd": "stop_speaking"}))
    finally:
        state.speaking = False


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


async def respond(ws: ServerConnection, state: ConnState) -> None:
    """Run STT → streaming agent → sentence-chunked TTS, then go idle."""
    pcm = bytes(state.speech_buf)
    state.speech_buf = bytearray()
    state.listening = False
    state.voiced_ms = 0
    state.trailing_silence_ms = 0

    await ws.send(json.dumps({"cmd": "stop_listening"}))

    transcript = stt.transcribe(pcm)
    if not transcript.text:
        log.info("empty transcript — going idle")
        return

    log.info("transcript: %r (%d ms)", transcript.text, transcript.latency_ms)

    agent = ensure_agent(ws, state)
    t0 = time.monotonic()
    speak_text = await _drive_agent_turn(
        ws, state, lambda spk: agent.respond(transcript.text, spk)
    )
    log.info(
        "agent turn: %d ms total, %r",
        int((time.monotonic() - t0) * 1000),
        speak_text[:120],
    )

    state.last_user_interaction_s = time.monotonic()


async def proactive_greet(ws: ServerConnection, state: ConnState) -> None:
    """Run an agent turn triggered by a non-speech event (new face seen).
    Bypasses STT and uses a stage-direction message instead."""
    now = time.monotonic()
    if state.listening or state.speaking:
        log.info("greet skipped: busy (listening=%s speaking=%s)",
                 state.listening, state.speaking)
        return
    if now - state.last_greeting_s < GREETING_COOLDOWN_S:
        log.info("greet skipped: cooldown (%.0fs since last)",
                 now - state.last_greeting_s)
        return
    if now - state.last_user_interaction_s < RECENT_INTERACTION_S:
        log.info("greet skipped: recent interaction (%.0fs ago)",
                 now - state.last_user_interaction_s)
        return

    state.last_greeting_s = now
    log.info("proactive greeting: new face")

    agent = ensure_agent(ws, state)
    t0 = time.monotonic()
    speak_text = await _drive_agent_turn(
        ws,
        state,
        lambda spk: agent.respond_to_event(
            "A new person just appeared in front of you. Greet them in one "
            "short, friendly sentence.",
            spk,
        ),
    )
    log.info(
        "greet turn: %d ms, %r",
        int((time.monotonic() - t0) * 1000), speak_text[:120],
    )


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
        asyncio.create_task(proactive_greet(ws, state))


async def handle(ws: ServerConnection) -> None:
    log.info("esp32 connected: %s", ws.remote_address)
    state = ConnState()
    try:
        async for msg in ws:
            if isinstance(msg, bytes) and msg and msg[0] == OP_AUDIO:
                if not state.listening:
                    continue
                frame = msg[1:]
                state.speech_buf.extend(frame)
                rms = frame_rms(frame)
                if rms >= SPEECH_RMS:
                    state.voiced_ms += FRAME_MS
                    state.trailing_silence_ms = 0
                else:
                    state.trailing_silence_ms += FRAME_MS

                elapsed_ms = int((time.monotonic() - state.started_at) * 1000)

                end_by_silence = (
                    state.voiced_ms >= SPEECH_LEAD_MS
                    and state.trailing_silence_ms >= SILENCE_TAIL_MS
                )
                end_by_timeout = elapsed_ms >= MAX_UTTERANCE_MS

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
                interval = (
                    DETECT_INTERVAL_LOOK_AROUND_S
                    if state.behavior.look_around_in_progress
                    else DETECT_INTERVAL_IDLE_S
                )
                if time.monotonic() - state.last_detect_s >= interval:
                    state.last_detect_s = time.monotonic()
                    asyncio.create_task(process_latest_jpeg(ws, state))
            elif isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    log.warning("bad json from esp32: %r", msg[:120])
                    continue
                log.info("event: %s", payload)
                if payload.get("event") == "wakeword":
                    state.listening = True
                    state.speech_buf = bytearray()
                    state.voiced_ms = 0
                    state.trailing_silence_ms = 0
                    state.started_at = time.monotonic()
                    state.last_user_interaction_s = time.monotonic()
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


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    zc, info = advertise_mdns()
    try:
        # ping_interval=None: the 78/esp-ml307 WebSocket on the firmware
        # doesn't reply to pings, so server-side keepalive trips the
        # connection every ~50 s. We accept the lost dead-conn detection.
        async with serve(
            handle, HOST, PORT, max_size=2**20, ping_interval=None
        ):
            log.info("brain listening on ws://%s:%d", HOST, PORT)
            await asyncio.Future()
    finally:
        if zc is not None and info is not None:
            zc.unregister_service(info)
            zc.close()


if __name__ == "__main__":
    asyncio.run(main())
