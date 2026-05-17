"""WebSocket server the M5StackChan ESP32 connects to.

Phase 1: connection lifecycle + audio echo. Receives binary opcode-0x01
PCM frames from the ESP32 and immediately echoes them back so we can
confirm the wire protocol end-to-end before adding STT/TTS in phase 2.

Wire protocol (kept tiny on purpose):
  - Binary frame, first byte = opcode:
      0x01  PCM audio frame (16 kHz, mono, s16le, 20 ms = 640 bytes payload)
      0x02  JPEG camera frame (phase 4+)
  - Text frame: JSON control message, both directions.
      from ESP32: {"event": "wakeword" | "vad_end" | "boot"}
      to   ESP32: {"cmd": "set_expression"|"look_at"|"set_motion_rate"|
                          "start_speaking"|"stop_speaking"|"stop_listening"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket

from websockets.asyncio.server import ServerConnection, serve
from zeroconf import ServiceInfo, Zeroconf

HOST = "0.0.0.0"
PORT = 8765
MDNS_NAME = "stackchan-brain"

OP_AUDIO = 0x01
OP_JPEG = 0x02

log = logging.getLogger("brain")


async def handle(ws: ServerConnection) -> None:
    log.info("esp32 connected: %s", ws.remote_address)
    audio_frames = 0
    last_audio_log = 0.0
    try:
        async for msg in ws:
            if isinstance(msg, bytes) and msg and msg[0] == OP_AUDIO:
                # Phase 1: echo audio back so the speaker plays what the mic hears.
                await ws.send(msg)
                audio_frames += 1
                now = asyncio.get_event_loop().time()
                if now - last_audio_log >= 2.0:
                    log.info(
                        "audio echo: %d frames so far (last frame %d bytes)",
                        audio_frames,
                        len(msg),
                    )
                    last_audio_log = now
            elif isinstance(msg, bytes) and msg and msg[0] == OP_JPEG:
                log.debug("jpeg frame, %d bytes", len(msg) - 1)
            elif isinstance(msg, str):
                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    log.warning("bad json from esp32: %r", msg[:120])
                    continue
                log.info("event: %s", payload)
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
