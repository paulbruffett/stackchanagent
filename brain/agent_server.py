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

import numpy as np
from dotenv import load_dotenv
from websockets.asyncio.server import ServerConnection, serve
from zeroconf import ServiceInfo, Zeroconf

# Load ANTHROPIC_API_KEY (and any other env) from the project root .env
# before importing the agent module (which constructs the Anthropic client).
load_dotenv(Path(__file__).parent.parent / ".env")

from claude_agent import AgentSession
from stt import Transcriber
from tts import Synthesizer

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

log = logging.getLogger("brain")

# Lazy globals — one model load per process.
stt = Transcriber()
tts = Synthesizer()


@dataclass
class ConnState:
    listening: bool = False
    speech_buf: bytearray = field(default_factory=bytearray)
    voiced_ms: int = 0
    trailing_silence_ms: int = 0
    started_at: float = 0.0
    agent: AgentSession | None = None


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


async def respond(ws: ServerConnection, state: ConnState) -> None:
    """Run STT → TTS → playback on the buffered speech, then go back to IDLE."""
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

    # Claude tool-use turn. Tools fire as side effects (WS commands).
    if state.agent is None:
        state.agent = AgentSession(ws)
    t0 = time.monotonic()
    speak_text = await state.agent.respond(transcript.text)
    agent_ms = int((time.monotonic() - t0) * 1000)
    log.info("agent: %d ms, %r", agent_ms, speak_text[:120])

    if not speak_text:
        log.info("empty agent reply — going idle")
        return

    t0 = time.monotonic()
    tts_pcm = tts.synthesize(speak_text)
    tts_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "tts ready: %d ms, %d bytes (%.2f s of audio)",
        tts_ms,
        len(tts_pcm),
        len(tts_pcm) / (SAMPLE_RATE * 2),
    )

    await ws.send(json.dumps({"cmd": "start_speaking"}))
    await send_pcm_stream(ws, tts_pcm)
    await ws.send(json.dumps({"cmd": "stop_speaking"}))


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
                log.debug("jpeg frame, %d bytes", len(msg) - 1)
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
        async with serve(handle, HOST, PORT, max_size=2**20):
            log.info("brain listening on ws://%s:%d", HOST, PORT)
            await asyncio.Future()
    finally:
        if zc is not None and info is not None:
            zc.unregister_service(info)
            zc.close()


if __name__ == "__main__":
    asyncio.run(main())
