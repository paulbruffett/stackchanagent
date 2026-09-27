"""Firmware OTA: image validation (incl. the signature block), the one-time
download URL, the progress state machine (attempt ids, ELF-hash success,
probation, cancel/expiry), the "robot is free" gate, the served base URL, and
the console endpoints."""

from __future__ import annotations

import hashlib
import struct
import zlib

import httpx
import pytest

from config import get_config
from firmware_ota import (
    APP_DESC_MAGIC,
    CONFIRM_TIMEOUT_S,
    DOWNLOAD_TTL_S,
    OTA_PARTITION_SIZE,
    QUEUE_TTL_S,
    STALL_TIMEOUT_S,
    FirmwareStore,
    ImageError,
    OtaManager,
    firmware_base_url,
    parse_image,
)
from policy import ota_send_ready
from webui.app import TOKEN_HEADER, create_app

TOKEN = "test-console-token"
AUTH = {TOKEN_HEADER: TOKEN}
BASE = "http://192.168.1.9:8080"


def sign(body: bytes) -> bytes:
    """Append a secure-boot-v2-format signature sector (dummy RSA fields,
    real digest + CRC — all the brain can check without the key)."""
    body = body + b"\xff" * (-len(body) % 4096)
    block = bytearray(1216)
    block[:4] = bytes([0xE7, 0x02, 0x00, 0x00])
    block[4:36] = hashlib.sha256(body).digest()
    struct.pack_into("<I", block, 1196, zlib.crc32(bytes(block[:1196])) & 0xFFFFFFFF)
    return body + bytes(block) + b"\xff" * (4096 - 1216)


def fake_image(
    *,
    version: str = "1.4.1",
    project: str = "stack-chan",
    date: str = "Sep 26 2026",
    time_: str = "10:11:12",
    elf: str = "ab" * 32,
    chip_id: int = 9,
    desc_magic: int = APP_DESC_MAGIC,
    body: int = 4096,
    signed: bool = True,
) -> bytes:
    """esp_image_header_t + one segment header + esp_app_desc_t, padded to
    `body` bytes, then (by default) a signature sector."""
    hdr = bytearray(24)
    hdr[0] = 0xE9
    hdr[1] = 1  # segment count
    struct.pack_into("<H", hdr, 12, chip_id)
    seg = struct.pack("<II", 0x3C000020, 256)
    desc = bytearray(256)
    struct.pack_into("<I", desc, 0, desc_magic)
    for off, n, s in ((16, 32, version), (48, 32, project), (80, 16, time_),
                      (96, 16, date), (112, 32, "v5.5.4")):
        raw = s.encode()[: n - 1]
        desc[off:off + len(raw)] = raw
    desc[144:176] = bytes.fromhex(elf)
    img = bytes(hdr) + seg + bytes(desc)
    img += b"\xff" * (body - len(img))
    return sign(img) if signed else img


class TestParseImage:
    def test_reads_the_app_description(self):
        info = parse_image(fake_image(elf="12" * 32))
        assert (info.project, info.version, info.built, info.idf) == (
            "stack-chan", "1.4.1", "Sep 26 2026 10:11:12", "v5.5.4")
        assert info.elf_sha256 == "12" * 32
        assert info.size == 8192 and len(info.sha256) == 64
        assert info.warnings == ()

    def test_rejects_non_image(self):
        with pytest.raises(ImageError, match="0xe9"):
            parse_image(b"\x7fELF" + fake_image()[4:])

    def test_rejects_other_chip(self):
        with pytest.raises(ImageError, match="chip"):
            parse_image(fake_image(chip_id=0))

    def test_rejects_missing_app_desc(self):
        # e.g. the bootloader image: same magic byte, no esp_app_desc_t
        with pytest.raises(ImageError, match="app description"):
            parse_image(fake_image(desc_magic=0))

    def test_rejects_truncated(self):
        with pytest.raises(ImageError, match="too small"):
            parse_image(fake_image()[:100])

    def test_size_limit_is_the_partition(self):
        fits = fake_image(body=OTA_PARTITION_SIZE - 4096)
        assert parse_image(fits).size == OTA_PARTITION_SIZE
        with pytest.raises(ImageError, match="partition"):
            parse_image(fake_image(body=OTA_PARTITION_SIZE))

    def test_rejects_unsigned(self):
        with pytest.raises(ImageError, match="not signed"):
            parse_image(fake_image(signed=False))
        with pytest.raises(ImageError, match="not signed"):
            parse_image(fake_image(signed=False, body=8192))

    def test_rejects_a_signature_for_another_image(self):
        img = bytearray(fake_image())
        img[200] ^= 0xFF  # body changed after signing
        with pytest.raises(ImageError, match="digest"):
            parse_image(bytes(img))

    def test_rejects_a_corrupt_signature_block(self):
        img = bytearray(fake_image())
        img[4096 + 100] ^= 0xFF
        with pytest.raises(ImageError, match="CRC"):
            parse_image(bytes(img))

    def test_other_project_is_a_warning(self):
        assert parse_image(fake_image(project="xiaozhi")).warnings


def test_store_keeps_only_the_latest_and_survives_a_bad_upload(tmp_path):
    store = FirmwareStore(tmp_path)
    store.save(fake_image(version="1"))
    store.save(fake_image(version="2"))
    with pytest.raises(ImageError):
        store.save(b"junk" * 100)
    assert store.info().version == "2"
    assert store.read() == fake_image(version="2")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["stack-chan.bin", "stack-chan.json"]


class TestBaseUrl:
    def lan(self):
        return "192.168.1.9"

    def test_specific_bind_is_used_as_is(self):
        assert firmware_base_url("192.168.1.20", 8080, self.lan) == "http://192.168.1.20:8080"

    def test_wildcard_uses_the_lan_address(self):
        assert firmware_base_url("0.0.0.0", 8080, self.lan) == "http://192.168.1.9:8080"
        assert firmware_base_url("::", 8080, self.lan) == "http://192.168.1.9:8080"

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.1.1", "::1", "[::1]", "localhost"])
    def test_loopback_is_refused(self, host):
        with pytest.raises(ValueError, match="loopback"):
            firmware_base_url(host, 8080, self.lan)

    def test_ipv6_is_bracketed(self):
        assert firmware_base_url("fd00::5", 8080, self.lan) == "http://[fd00::5]:8080"


@pytest.fixture
def mgr(tmp_path) -> OtaManager:
    return OtaManager(FirmwareStore(tmp_path), base_url=lambda: BASE)


class TestDownloadGrant:
    def test_single_use(self, mgr):
        g = mgr.issue_grant(now=0.0)
        assert mgr.claim(g.token, now=1.0)
        assert not mgr.claim(g.token, now=2.0)

    def test_expires(self, mgr):
        g = mgr.issue_grant(now=0.0)
        assert not mgr.claim(g.token, now=DOWNLOAD_TTL_S + 1)

    def test_wrong_token(self, mgr):
        mgr.issue_grant(now=0.0)
        assert not mgr.claim("guess", now=1.0)
        assert not OtaManager(mgr.store).claim("", now=0.0)  # nothing issued

    def test_regenerated_per_send(self, mgr):
        old = mgr.issue_grant(now=0.0).token
        new = mgr.issue_grant(now=1.0).token
        assert old != new
        assert not mgr.claim(old, now=2.0)
        assert mgr.claim(new, now=2.0)


def _sent(mgr: OtaManager, **img) -> dict:
    """Upload, queue and send; returns the ota command."""
    mgr.store.save(fake_image(**img))
    mgr.request_send(now=0.0)
    cmd = mgr.build_command(now=0.0)
    mgr.mark_sent(now=0.0)
    return cmd


class TestOtaFlow:
    def test_command_carries_id_url_size_and_hash(self, mgr):
        info = mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        assert mgr.requested and not mgr.device_busy
        cmd = mgr.build_command(now=0.0)
        assert cmd["cmd"] == "ota" and cmd["size"] == info.size and cmd["id"] == 1
        assert cmd["sha256"] == info.sha256
        assert cmd["url"] == f"{BASE}/firmware/{mgr.grant.token}.bin"
        mgr.mark_sent(now=0.0)
        assert not mgr.requested and mgr.device_busy
        assert mgr.build_command() is None  # sent once

    def test_nothing_stored(self, mgr):
        with pytest.raises(LookupError):
            mgr.request_send()

    def test_loopback_console_refuses_at_queue_time(self, tmp_path):
        def refuse():
            raise ValueError("the console is bound to loopback only")
        m = OtaManager(FirmwareStore(tmp_path), base_url=refuse)
        m.store.save(fake_image())
        with pytest.raises(ValueError, match="loopback"):
            m.request_send()
        assert m.state == "idle"

    def test_no_second_send_while_one_runs(self, mgr):
        mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        with pytest.raises(RuntimeError):
            mgr.request_send(now=1.0)

    def test_success_is_judged_by_elf_hash_and_probation(self, mgr):
        _sent(mgr, elf="cd" * 32)
        mgr.on_event({"id": 1, "state": "downloading", "pct": 40}, now=5.0)
        assert (mgr.state, mgr.pct) == ("downloading", 40)
        mgr.on_event({"id": 1, "state": "rebooting"}, now=30.0)
        # Same version and build date as anything else; only the hash counts.
        mgr.on_boot("1.4.1", "Sep 26 2026 10:11:12", "cd" * 32, False, now=45.0)
        assert mgr.state == "confirming" and mgr.device_busy
        assert mgr.robot_fw["elf_sha256"] == "cd" * 32
        mgr.on_event({"state": "confirmed", "fw_sha": "cd" * 32}, now=80.0)
        assert mgr.state == "done" and not mgr.device_busy
        assert mgr.grant is None

    def test_reconnect_after_confirming_counts_too(self, mgr):
        _sent(mgr, elf="cd" * 32)
        mgr.on_event({"id": 1, "state": "rebooting"}, now=30.0)
        mgr.on_boot("1.4.1", "x", "cd" * 32, False, now=45.0)
        # "confirmed" lost with a link drop; the next boot says it's valid.
        mgr.on_boot("1.4.1", "x", "cd" * 32, True, now=90.0)
        assert mgr.state == "done"

    def test_same_version_different_build_is_a_rollback(self, mgr):
        _sent(mgr, elf="cd" * 32)
        mgr.on_event({"id": 1, "state": "rebooting"}, now=30.0)
        mgr.on_boot("1.4.1", "Sep 26 2026 10:11:12", "ef" * 32, True, now=100.0)
        assert mgr.state == "failed" and "rolled back" in mgr.error

    def test_rollback_after_probation_started(self, mgr):
        _sent(mgr, elf="cd" * 32)
        mgr.on_event({"id": 1, "state": "rebooting"}, now=30.0)
        mgr.on_boot("1.4.1", "x", "cd" * 32, False, now=45.0)
        mgr.on_boot("1.4.0", "y", "01" * 32, True, now=400.0)
        assert mgr.state == "failed" and "rolled back" in mgr.error

    def test_robot_failure_is_reported(self, mgr):
        _sent(mgr)
        mgr.on_event({"id": 1, "state": "failed", "error": "sha256 mismatch"}, now=3.0)
        assert (mgr.state, mgr.error) == ("failed", "sha256 mismatch")
        mgr.request_send(now=4.0)  # can retry after a failure
        assert mgr.state == "queued"

    def test_stale_event_from_an_earlier_attempt_is_ignored(self, mgr):
        _sent(mgr)
        mgr.on_event({"id": 1, "state": "failed", "error": "boom"}, now=3.0)
        mgr.request_send(now=4.0)  # attempt 2 queued
        mgr.on_event({"id": 1, "state": "failed", "error": "late echo"}, now=5.0)
        assert mgr.state == "queued" and mgr.attempt == 2
        # …and even an id-less event can't move a queued update along.
        mgr.on_event({"state": "rebooting"}, now=6.0)
        assert mgr.state == "queued"

    def test_confirmed_for_another_image_is_ignored(self, mgr):
        _sent(mgr, elf="cd" * 32)
        mgr.on_event({"id": 1, "state": "rebooting"}, now=30.0)
        mgr.on_event({"state": "confirmed", "fw_sha": "ef" * 32}, now=31.0)
        assert mgr.state == "rebooting"

    def test_download_silence_times_out(self, mgr):
        _sent(mgr)
        mgr.check_stall(now=STALL_TIMEOUT_S - 1)
        assert mgr.state == "sent"
        mgr.check_stall(now=STALL_TIMEOUT_S + 1)
        assert mgr.state == "failed"

    def test_probation_that_never_ends_fails(self, mgr):
        _sent(mgr)
        mgr.on_event({"id": 1, "state": "rebooting"}, now=0.0)
        mgr.check_stall(now=CONFIRM_TIMEOUT_S - 1)
        assert mgr.state == "rebooting"
        mgr.check_stall(now=CONFIRM_TIMEOUT_S + 1)
        assert mgr.state == "failed"

    def test_reconnect_mid_download_fails_it(self, mgr):
        _sent(mgr)
        mgr.on_boot("1.4.1", "x", "ab" * 32, True, now=10.0)
        assert mgr.state == "failed" and "before the update finished" in mgr.error

    def test_boot_without_update_just_records_the_version(self, mgr):
        mgr.on_boot("1.4.1", "Sep 1 2026 00:00:00", "ab" * 32, True)
        assert mgr.state == "idle" and mgr.robot_fw["version"] == "1.4.1"


class TestQueue:
    def test_cancel(self, mgr):
        mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        mgr.cancel(now=1.0)
        assert mgr.state == "cancelled" and not mgr.requested
        assert mgr.build_command() is None

    def test_cannot_cancel_once_sent(self, mgr):
        _sent(mgr)
        with pytest.raises(RuntimeError):
            mgr.cancel()

    def test_queued_update_expires(self, mgr):
        mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        mgr.check_stall(now=QUEUE_TTL_S - 1)
        assert mgr.requested
        mgr.check_stall(now=QUEUE_TTL_S + 1)
        assert mgr.state == "failed" and "expired" in mgr.error
        assert mgr.build_command() is None

    def test_image_replaced_while_queued_is_not_sent(self, mgr, tmp_path):
        mgr.store.save(fake_image(version="1"))
        mgr.request_send(now=0.0)
        FirmwareStore(tmp_path).save(fake_image(version="2"))  # behind its back
        assert mgr.build_command(now=1.0) is None
        assert mgr.state == "failed"


class TestSendGate:
    def test_sends_when_the_robot_is_free(self):
        assert ota_send_ready(requested=True, boot_seen=True, busy=False)

    def test_waits_while_busy(self):
        assert not ota_send_ready(requested=True, boot_seen=True, busy=True)

    def test_waits_for_the_boot_report(self):
        assert not ota_send_ready(requested=True, boot_seen=False, busy=False)

    def test_nothing_requested(self):
        assert not ota_send_ready(requested=False, boot_seen=True, busy=False)


# --- console endpoints ------------------------------------------------------

@pytest.fixture
async def console(mem, mgr):
    app = create_app(mem, get_config(), token=TOKEN, ota=mgr)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as c:
        yield c


def _upload(data: bytes) -> dict:
    return {"files": {"file": ("stack-chan.bin", data, "application/octet-stream")}}


async def test_upload_needs_the_token(console, mgr):
    r = await console.post("/api/firmware", **_upload(fake_image()))
    assert r.status_code == 401
    assert mgr.store.info() is None
    assert (await console.post("/api/firmware/send")).status_code == 401
    assert (await console.post("/api/firmware/cancel")).status_code == 401
    assert (await console.get("/api/firmware")).status_code == 401


async def test_upload_send_and_download(console, mgr):
    r = await console.post("/api/firmware", headers=AUTH, **_upload(fake_image(version="9.9")))
    assert r.status_code == 200
    assert r.json()["stored"]["version"] == "9.9"

    r = await console.post("/api/firmware/send", headers=AUTH)
    assert r.json()["state"] == "queued"
    cmd = mgr.build_command()
    path = cmd["url"].removeprefix(BASE)

    # No console token: the one-time path is the credential.
    r = await console.get(path)
    assert r.status_code == 200 and r.content == fake_image(version="9.9")
    assert (await console.get(path)).status_code == 404  # single use
    assert (await console.get("/firmware/guess.bin")).status_code == 404


async def test_cancel_endpoint(console, mgr):
    await console.post("/api/firmware", headers=AUTH, **_upload(fake_image()))
    assert (await console.post("/api/firmware/cancel", headers=AUTH)).status_code == 409
    await console.post("/api/firmware/send", headers=AUTH)
    r = await console.post("/api/firmware/cancel", headers=AUTH)
    assert r.status_code == 200 and r.json()["state"] == "cancelled"


async def test_unsigned_upload_is_refused(console, mgr):
    r = await console.post("/api/firmware", headers=AUTH, **_upload(fake_image(signed=False)))
    assert r.status_code == 400 and "not signed" in r.json()["detail"]
    assert mgr.store.info() is None


async def test_send_without_upload(console):
    assert (await console.post("/api/firmware/send", headers=AUTH)).status_code == 404
