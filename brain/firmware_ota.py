"""Over-the-air firmware updates: image checks, storage, the one-time
download URL, and the update's progress as the robot reports it.

Dependency-free (no FastAPI, no websockets) so it is unit-testable offline.
The console (webui/app.py) uploads an image and queues it; agent_server's idle
ticker sends the `ota` command between conversations and feeds the firmware's
`ota` / `boot` events back in here.

The robot fetches the image with a plain HTTP GET and can't easily attach the
console token, so the download URL itself is the credential: an unguessable
path, good for one GET, expiring after DOWNLOAD_TTL_S, regenerated for every
send. The image itself is signed (firmware/README.md) and the robot refuses
one that isn't signed with its own key, so the URL only has to keep the
image private, not make it trustworthy.

Lifecycle of one update (`state`):

    idle/done/failed/cancelled ─request_send→ queued ─(ticker, robot idle)→ sent
    sent → downloading → rebooting → confirming → done
                       (new image boots, reconnects; its boot event carries
                        its ELF SHA-256 and whether it is still on probation)

Anything can end in failed (robot error, silence, rollback); queued can be
cancelled or expire. Every send has an attempt id the firmware echoes on its
progress events, so a late event from an earlier attempt can't move a newer
one along.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import os
import secrets
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

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

# Secure-boot-v2-format signature (CONFIG_SECURE_SIGNED_APPS_RSA_SCHEME): the
# image is padded to a 4 KB boundary and followed by one 4 KB sector whose
# first 1216-byte block starts E7 02 00 00, then the SHA-256 of everything
# before the sector, …, and a CRC32 of the first 1196 bytes at 1196.
SIG_SECTOR = 4096
SIG_BLOCK_LEN = 1216
SIG_CRC_OFFSET = 1196
SIG_BLOCK_HEAD = bytes([0xE7, 0x02, 0x00, 0x00])

DOWNLOAD_TTL_S = 600.0
# Shortest reported ELF-hash prefix accepted as identifying a build.
MIN_SHA_PREFIX = 8
# Sent or mid-download with no word from the robot for this long: give up on
# it. The firmware's HTTP read times out well before this.
STALL_TIMEOUT_S = 120.0
# Rebooting/confirming: the new image has 5 min from boot to prove itself
# (firmware/main/agent/ota.cpp) before it rolls back, then reconnects on the
# old one. Wait past that so the console can say "rolled back".
CONFIRM_TIMEOUT_S = 420.0
# A queued update nobody could send (robot offline or never idle) goes stale.
QUEUE_TTL_S = 3600.0

IN_FLIGHT = ("sent", "downloading", "rebooting", "confirming")


class ImageError(ValueError):
    """The upload is not a flashable app image for this robot."""


@dataclass(frozen=True)
class ImageInfo:
    size: int
    sha256: str
    elf_sha256: str
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


def _check_signature(data: bytes) -> None:
    unsigned = ImageError(
        "image is not signed — build it with the signing key "
        "(CONFIG_SECURE_SIGNED_APPS_NO_SECURE_BOOT, see firmware/README.md)"
    )
    if len(data) % SIG_SECTOR or len(data) < 2 * SIG_SECTOR:
        raise unsigned
    body_len = len(data) - SIG_SECTOR
    block = data[body_len:body_len + SIG_BLOCK_LEN]
    if block[:4] != SIG_BLOCK_HEAD:
        raise unsigned
    (crc,) = struct.unpack_from("<I", block, SIG_CRC_OFFSET)
    if crc != zlib.crc32(block[:SIG_CRC_OFFSET]) & 0xFFFFFFFF:
        raise ImageError("signature block is corrupt (CRC mismatch)")
    if block[4:36] != hashlib.sha256(data[:body_len]).digest():
        raise ImageError("signature block doesn't belong to this image (digest mismatch)")


def parse_image(data: bytes, max_size: int = OTA_PARTITION_SIZE) -> ImageInfo:
    """Validate a signed ESP32-S3 app image and read its esp_app_desc_t.
    Raises ImageError with an operator-readable reason. Whether the signature
    is from the right key only the robot can tell (it holds the public key)."""
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
    _check_signature(data)
    d = data[APP_DESC_OFFSET:APP_DESC_OFFSET + APP_DESC_LEN]
    project = _cstr(d[48:80])
    warnings = ()
    if project != EXPECTED_PROJECT:
        warnings = (f"project is {project!r}, not {EXPECTED_PROJECT!r}",)
    return ImageInfo(
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        elf_sha256=d[144:176].hex(),
        project=project,
        version=_cstr(d[16:48]),
        time=_cstr(d[80:96]),
        date=_cstr(d[96:112]),
        idf=_cstr(d[112:144]),
        warnings=warnings,
    )


def firmware_base_url(bind_host: str, port: int, lan_ip: Callable[[], str]) -> str:
    """http://host:port the robot downloads from: the console's own bind
    address, or the LAN address when it's bound to every interface (or a
    name). Raises ValueError when the console is loopback-only — the robot
    can't reach it, and quietly serving from elsewhere would widen what the
    operator chose to expose."""
    host = bind_host.strip().strip("[]")
    if host.lower() == "localhost":
        raise ValueError("the console is bound to loopback only (CONSOLE_BIND); "
                         "the robot can't download firmware from it")
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None and addr.is_loopback:
        raise ValueError(f"the console is bound to loopback only ({host}); "
                         "the robot can't download firmware from it")
    if addr is None or addr.is_unspecified:
        addr = ipaddress.ip_address(lan_ip())
    shown = f"[{addr}]" if addr.version == 6 else str(addr)
    return f"http://{shown}:{port}"


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
    the live download grant, and the update's progress as the firmware
    reports it."""

    def __init__(
        self,
        store: FirmwareStore | None = None,
        base_url: Callable[[], str] | None = None,
    ) -> None:
        self.store = store or FirmwareStore()
        # () -> "http://<ip>:<console port>"; raises ValueError when the
        # robot can't be served (loopback-only console). Set by agent_server.
        self.base_url = base_url
        self.grant: Grant | None = None
        self.state = "idle"
        self.attempt = 0
        self.pct = 0
        self.error = ""
        self.updated_at = 0.0
        self.target: ImageInfo | None = None
        # What the robot said it is running in its last boot event.
        self.robot_fw: dict[str, Any] | None = None

    # --- predicates -------------------------------------------------------
    @property
    def requested(self) -> bool:
        """A send is waiting for the robot to be idle."""
        return self.state == "queued"

    @property
    def device_busy(self) -> bool:
        """The robot is downloading, flashing, rebooting into, or still on
        probation with a new image. Brain-initiated actions wait: a set_buddy
        reboot now would roll the update back."""
        return self.state in IN_FLIGHT

    @property
    def in_progress(self) -> bool:
        return self.requested or self.device_busy

    # --- console side -----------------------------------------------------
    def request_send(self, now: float | None = None) -> ImageInfo:
        """Queue the stored image. Raises LookupError with nothing stored,
        RuntimeError while an update is under way, ValueError when the robot
        couldn't download it (loopback-only console)."""
        info = self.store.info()
        if info is None:
            raise LookupError("no firmware uploaded")
        if self.in_progress:
            raise RuntimeError(f"an update is already {self.state}")
        if self.base_url is None:
            raise ValueError("firmware serving is not configured")
        self.base_url()  # fail now, in the console, rather than at send time
        self.attempt += 1
        self.target = info
        self._set("queued", now=now)
        log.info("ota #%d queued: %s %s (%s, elf %s, %d bytes)", self.attempt,
                 info.project, info.version, info.built, info.elf_sha256[:12], info.size)
        return info

    def cancel(self, now: float | None = None) -> None:
        """Drop a queued update. Raises RuntimeError once it has been sent
        (the robot is already on it)."""
        if not self.requested:
            raise RuntimeError(f"nothing queued (update is {self.state})")
        self._set("cancelled", now=now)
        log.info("ota #%d cancelled", self.attempt)

    def fail(self, error: str, now: float | None = None) -> None:
        """End the current update with `error` (e.g. the robot can't be
        served after all)."""
        self._set("failed", error=error, now=now)
        log.warning("ota #%d: %s", self.attempt, error)

    def status(self, now: float | None = None) -> dict[str, Any]:
        self.check_stall(now)
        return {
            "state": self.state,
            "attempt": self.attempt,
            "pct": self.pct,
            "error": self.error,
            "updated_at": self.updated_at,
            "stored": _describe(self.store.info()),
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
        """The `ota` command for a queued update, with a fresh grant; None if
        nothing is queued. Raises ValueError if the robot can't be served."""
        if not self.requested:
            return None
        info = self.store.info()
        if info is None or self.target is None or info.sha256 != self.target.sha256:
            self._set("failed", error="the stored firmware changed or vanished", now=now)
            return None
        base = self.base_url() if self.base_url else ""
        grant = self.issue_grant(now)
        return {
            "cmd": "ota",
            "id": self.attempt,
            "url": f"{base}/firmware/{grant.token}.bin",
            "size": info.size,
            "sha256": info.sha256,
        }

    def mark_sent(self, now: float | None = None) -> None:
        if self.requested:
            self._set("sent", now=now)

    def on_event(self, payload: dict[str, Any], now: float | None = None) -> None:
        """Firmware {"event":"ota","id":n,"state":downloading|rebooting|failed}
        and, from the new image, {"state":"confirmed","fw_sha":…}."""
        state = payload.get("state")
        if state == "confirmed":
            if self.state in ("rebooting", "confirming") and self._is_target(payload.get("fw_sha")):
                self._set("done", now=now)
                log.info("ota #%d: new image confirmed by the robot", self.attempt)
            return
        pid = payload.get("id")
        if pid is not None and pid != self.attempt:
            log.info("ota: ignoring %s event for attempt %s (current #%d)",
                     state, pid, self.attempt)
            return
        if self.state not in ("sent", "downloading"):
            log.info("ota: ignoring late %s event (update is %s)", state, self.state)
            return
        if state == "downloading":
            pct = payload.get("pct")
            self.pct = int(pct) if isinstance(pct, (int, float)) else self.pct
            self._set("downloading", now=now)
        elif state == "rebooting":
            self.pct = 100
            self._set("rebooting", now=now)
            log.info("ota #%d: image written, robot rebooting", self.attempt)
        elif state == "failed":
            err = str(payload.get("error") or "unknown error")
            self._set("failed", error=err, now=now)
            log.warning("ota #%d failed on the robot: %s", self.attempt, err)
        else:
            log.warning("ota: unexpected state %r", state)

    def on_boot(self, fw: str | None, built: str | None, fw_sha: str | None,
                fw_valid: bool | None, now: float | None = None) -> None:
        """Boot event after a (re)connect: record what the robot runs, and
        judge a pending update by the ELF SHA-256 — version strings and build
        dates don't reliably tell two builds apart."""
        if fw is not None or fw_sha is not None:
            self.robot_fw = {"version": fw or "", "built": built or "",
                             "elf_sha256": fw_sha or ""}
        if self.state in ("sent", "downloading"):
            # The download dies with the connection it was reporting on.
            self._set("failed", error="robot reconnected before the update finished", now=now)
            log.warning("ota #%d: %s", self.attempt, self.error)
        elif self.state in ("rebooting", "confirming"):
            if not self._is_target(fw_sha):
                self._set("failed", error=(
                    f"robot came back running {fw or '?'} ({(fw_sha or '?')[:12]}), not "
                    "the uploaded image — rolled back?"), now=now)
                log.warning("ota #%d: %s", self.attempt, self.error)
            elif fw_valid:
                self._set("done", now=now)
                log.info("ota #%d: robot is running the new image", self.attempt)
            else:
                # On probation: the firmware marks it valid after 30 s of
                # brain link and reports "confirmed".
                self._set("confirming", now=now)
                log.info("ota #%d: new image up, waiting for it to confirm", self.attempt)

    def check_stall(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        age = now - self.updated_at
        if self.state == "queued" and age > QUEUE_TTL_S:
            self._set("failed", error="queued update expired (robot never free for an hour)",
                      now=now)
        elif self.state in ("sent", "downloading") and age > STALL_TIMEOUT_S:
            self._set("failed", error=f"no word from the robot for {STALL_TIMEOUT_S:.0f}s",
                      now=now)
        elif self.state in ("rebooting", "confirming") and age > CONFIRM_TIMEOUT_S:
            self._set("failed", error="the new image never confirmed itself", now=now)
        else:
            return
        log.warning("ota #%d: %s", self.attempt, self.error)

    def _is_target(self, fw_sha: Any) -> bool:
        """Whether the robot's reported ELF hash is the image we sent. Firmware
        built before CONFIG_APP_RETRIEVE_LEN_ELF_SHA=64 reports only the first
        9 hex characters, so accept a prefix of at least MIN_SHA_PREFIX."""
        if not isinstance(fw_sha, str) or self.target is None:
            return False
        reported = fw_sha.lower()
        return (len(reported) >= MIN_SHA_PREFIX
                and self.target.elf_sha256.startswith(reported))

    def _set(self, state: str, *, error: str = "", now: float | None = None) -> None:
        self.state = state
        self.error = error
        if state in ("queued", "sent"):
            self.pct = 0
        if state not in ("sent", "downloading"):
            self.grant = None  # nothing left to download
        self.updated_at = time.monotonic() if now is None else now
