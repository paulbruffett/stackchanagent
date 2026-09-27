"""Over-the-air firmware updates: image checks, storage, the one-time
download URL, and the update's progress as the robot reports it.

Dependency-free (no FastAPI, no websockets) so it is unit-testable offline.
The console (webui/app.py) uploads an image and asks for it to be sent;
agent_server's idle ticker sends the `ota` command between conversations and
feeds the firmware's `ota` / `boot` events back in here.

The robot fetches the image with a plain HTTP GET and can't easily attach the
console token, so the download URL itself is the credential: an unguessable
path, good for one GET, expiring after DOWNLOAD_TTL_S, regenerated for every
send.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import struct
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("brain.ota")

# firmware/partitions.csv: ota_0 and ota_1 are 0x4f0000 each.
OTA_PARTITION_SIZE = 0x4F0000

# esp_image_header_t (24 bytes) + first esp_image_segment_header_t (8 bytes),
# then esp_app_desc_t — ESP-IDF components/bootloader_support/include/
# esp_app_format.h and esp_app_desc.h.
IMAGE_MAGIC = 0xE9
IMAGE_HEADER_LEN = 24
SEGMENT_HEADER_LEN = 8
APP_DESC_OFFSET = IMAGE_HEADER_LEN + SEGMENT_HEADER_LEN
APP_DESC_MAGIC = 0xABCD5432
APP_DESC_LEN = 256
CHIP_ID_OFFSET = 12
ESP32S3_CHIP_ID = 9
EXPECTED_PROJECT = "stack-chan"

DOWNLOAD_TTL_S = 600.0
# Sent, or mid-download, with no word from the robot for this long: give up
# on it in the console. The firmware's HTTP read has its own timeout well
# below this, so a healthy update always reports first.
STALL_TIMEOUT_S = 120.0


class ImageError(ValueError):
    """The upload is not a flashable app image for this robot."""


@dataclass(frozen=True)
class ImageInfo:
    size: int
    sha256: str
    project: str
    version: str
    date: str
    time: str
    idf: str
    # Non-fatal oddities the console should show (e.g. another project's app).
    warnings: tuple[str, ...] = ()

    @property
    def built(self) -> str:
        return f"{self.date} {self.time}"


def _cstr(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace")


def parse_image(data: bytes, max_size: int = OTA_PARTITION_SIZE) -> ImageInfo:
    """Validate an ESP32-S3 app image and read its esp_app_desc_t. Raises
    ImageError with an operator-readable reason."""
    if len(data) < APP_DESC_OFFSET + APP_DESC_LEN:
        raise ImageError(f"too small for an app image ({len(data)} bytes)")
    if len(data) > max_size:
        raise ImageError(
            f"{len(data)} bytes does not fit the {max_size:#x}-byte OTA partition"
        )
    if data[0] != IMAGE_MAGIC:
        raise ImageError(f"not an ESP app image (first byte {data[0]:#04x}, want 0xe9)")
    (chip_id,) = struct.unpack_from("<H", data, CHIP_ID_OFFSET)
    if chip_id != ESP32S3_CHIP_ID:
        raise ImageError(f"built for chip id {chip_id}, not the ESP32-S3 ({ESP32S3_CHIP_ID})")
    (desc_magic,) = struct.unpack_from("<I", data, APP_DESC_OFFSET)
    if desc_magic != APP_DESC_MAGIC:
        raise ImageError(
            "no app description after the image header — a bootloader or "
            "partition table rather than the app (build/stack-chan.bin)?"
        )
    d = data[APP_DESC_OFFSET:APP_DESC_OFFSET + APP_DESC_LEN]
    project = _cstr(d[48:80])
    warnings = ()
    if project != EXPECTED_PROJECT:
        warnings = (f"project is {project!r}, not {EXPECTED_PROJECT!r}",)
    return ImageInfo(
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        project=project,
        version=_cstr(d[16:48]),
        time=_cstr(d[80:96]),
        date=_cstr(d[96:112]),
        idf=_cstr(d[112:144]),
        warnings=warnings,
    )


def _describe(info: ImageInfo | None) -> dict[str, Any] | None:
    return None if info is None else asdict(info) | {"built": info.built}


class FirmwareStore:
    """The one uploaded image (only the latest is kept) plus its parsed info,
    under ~/.stackchan/firmware/."""

    IMAGE = "stack-chan.bin"
    META = "stack-chan.json"

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path.home() / ".stackchan" / "firmware"

    def save(self, data: bytes) -> ImageInfo:
        """Validate and store `data`, replacing any earlier image. Raises
        ImageError, leaving the previous image in place."""
        info = parse_image(data)
        self.root.mkdir(parents=True, exist_ok=True)
        for name, payload in (
            (self.IMAGE, data),
            (self.META, json.dumps(asdict(info)).encode()),
        ):
            tmp = self.root / (name + ".tmp")
            tmp.write_bytes(payload)
            os.replace(tmp, self.root / name)
        return info

    def info(self) -> ImageInfo | None:
        try:
            meta = json.loads((self.root / self.META).read_text())
            meta["warnings"] = tuple(meta.get("warnings") or ())
            return ImageInfo(**meta)
        except (OSError, ValueError, TypeError):
            return None

    def read(self) -> bytes | None:
        try:
            return (self.root / self.IMAGE).read_bytes()
        except OSError:
            return None


@dataclass
class Grant:
    token: str
    expires_at: float
    used: bool = False


class OtaManager:
    """Shared between the console and the WebSocket handler: what's stored,
    whether a send has been requested, the live download grant, and the
    update's progress as the firmware reports it."""

    def __init__(self, store: FirmwareStore | None = None) -> None:
        self.store = store or FirmwareStore()
        self.base_url = ""  # http://<lan ip>:<console port>, set by agent_server
        self.requested = False
        self.grant: Grant | None = None
        # idle | queued | sent | downloading | rebooting | done | failed
        self.state = "idle"
        self.pct = 0
        self.error = ""
        self.updated_at = 0.0
        self.target: ImageInfo | None = None
        # What the robot said it is running in its last boot event.
        self.robot_fw: dict[str, str] | None = None

    # --- console side ---------------------------------------------------
    def request_send(self, now: float | None = None) -> ImageInfo:
        """Queue the stored image for the robot. The ticker sends it at the
        next quiet moment. Raises LookupError with nothing stored and
        RuntimeError while an update is already under way."""
        info = self.store.info()
        if info is None:
            raise LookupError("no firmware uploaded")
        if self.in_progress:
            raise RuntimeError(f"an update is already {self.state}")
        self.requested = True
        self.target = info
        self._set("queued", now=now)
        log.info("ota queued: %s %s (%s, %d bytes)", info.project, info.version,
                 info.built, info.size)
        return info

    @property
    def in_progress(self) -> bool:
        return self.state in ("queued", "sent", "downloading", "rebooting")

    def status(self, now: float | None = None) -> dict[str, Any]:
        self.check_stall(now)
        info = self.store.info()
        return {
            "state": self.state,
            "pct": self.pct,
            "error": self.error,
            "updated_at": self.updated_at,
            "stored": _describe(info),
            "target": _describe(self.target),
            "robot": self.robot_fw,
        }

    # --- download URL -----------------------------------------------------
    def issue_grant(self, now: float | None = None) -> Grant:
        """A fresh download token; any earlier one stops working."""
        now = time.monotonic() if now is None else now
        self.grant = Grant(secrets.token_urlsafe(24), now + DOWNLOAD_TTL_S)
        return self.grant

    def claim(self, token: str, now: float | None = None) -> bool:
        """True exactly once per issued token, and only before it expires."""
        now = time.monotonic() if now is None else now
        g = self.grant
        if g is None or g.used or now > g.expires_at:
            return False
        if not secrets.compare_digest(token.encode(), g.token.encode()):
            return False
        g.used = True
        return True

    # --- robot side -------------------------------------------------------
    def build_command(self, now: float | None = None) -> dict[str, Any] | None:
        """The `ota` command for a requested send, with a fresh grant; None if
        nothing is requested or the stored image vanished."""
        if not self.requested:
            return None
        info = self.store.info()
        if info is None:
            self.requested = False
            self._set("failed", error="stored firmware disappeared", now=now)
            return None
        grant = self.issue_grant(now)
        return {
            "cmd": "ota",
            "url": f"{self.base_url}/firmware/{grant.token}.bin",
            "size": info.size,
            "sha256": info.sha256,
        }

    def mark_sent(self, now: float | None = None) -> None:
        self.requested = False
        self._set("sent", now=now)

    def on_event(self, payload: dict[str, Any], now: float | None = None) -> None:
        """Firmware {"event":"ota","state":downloading|rebooting|failed, …}."""
        state = payload.get("state")
        if state == "downloading":
            pct = payload.get("pct")
            self.pct = int(pct) if isinstance(pct, (int, float)) else self.pct
            self._set("downloading", now=now)
        elif state == "rebooting":
            self.pct = 100
            self._set("rebooting", now=now)
            log.info("ota: image written, robot rebooting")
        elif state == "failed":
            err = str(payload.get("error") or "unknown error")
            self._set("failed", error=err, now=now)
            log.warning("ota failed on the robot: %s", err)
        else:
            log.warning("ota: unexpected state %r", state)

    def on_boot(self, fw: str | None, built: str | None, now: float | None = None) -> None:
        """Boot event after (re)connect. Closes out a pending update: the new
        image only gets this far once it reached the brain, which is also when
        it marks itself valid."""
        if fw is not None or built is not None:
            self.robot_fw = {"version": fw or "", "built": built or ""}
        if self.state not in ("rebooting", "downloading", "sent"):
            return
        t = self.target
        if t is not None and built == t.built and fw == t.version:
            self._set("done", now=now)
            log.info("ota: robot is running %s (%s)", fw, built)
        elif self.state == "rebooting":
            self._set("failed", error=(
                f"robot came back running {fw or '?'} ({built or '?'}), not the "
                "uploaded image — rolled back?"), now=now)
            log.warning("ota: %s", self.error)
        else:
            # Reconnected mid-download (link drop, or the robot reset itself):
            # the download died with the old connection.
            self._set("failed", error="robot reconnected before the update finished",
                      now=now)
            log.warning("ota: %s", self.error)

    def check_stall(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state in ("sent", "downloading", "rebooting") and \
                now - self.updated_at > STALL_TIMEOUT_S:
            self._set("failed", error=f"no word from the robot for {STALL_TIMEOUT_S:.0f}s",
                      now=now)
            log.warning("ota: %s", self.error)

    def _set(self, state: str, *, error: str = "", now: float | None = None) -> None:
        self.state = state
        self.error = error
        if state in ("done", "failed"):
            self.grant = None  # a finished update's URL has nothing left to serve
        if state in ("queued", "sent"):
            self.pct = 0
        self.updated_at = time.monotonic() if now is None else now
