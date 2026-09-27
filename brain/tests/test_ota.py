"""Firmware OTA: image validation, the one-time download URL, the progress
state machine, the "between turns only" gate, and the console endpoints."""

from __future__ import annotations

import struct

import httpx
import pytest

from config import get_config
from firmware_ota import (
    APP_DESC_MAGIC,
    DOWNLOAD_TTL_S,
    OTA_PARTITION_SIZE,
    STALL_TIMEOUT_S,
    FirmwareStore,
    ImageError,
    OtaManager,
    parse_image,
)
from policy import ota_send_ready
from webui.app import TOKEN_HEADER, create_app

TOKEN = "test-console-token"
AUTH = {TOKEN_HEADER: TOKEN}
BASE = "http://192.168.1.9:8080"


def fake_image(
    *,
    version: str = "1.4.1",
    project: str = "stack-chan",
    date: str = "Sep 26 2026",
    time_: str = "10:11:12",
    chip_id: int = 9,
    desc_magic: int = APP_DESC_MAGIC,
    total: int = 4096,
) -> bytes:
    """esp_image_header_t + one segment header + esp_app_desc_t, padded."""
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
    img = bytes(hdr) + seg + bytes(desc)
    return img + b"\xff" * (total - len(img))


class TestParseImage:
    def test_reads_the_app_description(self):
        info = parse_image(fake_image())
        assert (info.project, info.version, info.built, info.idf) == (
            "stack-chan", "1.4.1", "Sep 26 2026 10:11:12", "v5.5.4")
        assert info.size == 4096 and len(info.sha256) == 64
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
        assert parse_image(fake_image(total=OTA_PARTITION_SIZE)).size == OTA_PARTITION_SIZE
        with pytest.raises(ImageError, match="partition"):
            parse_image(fake_image(total=OTA_PARTITION_SIZE + 1))

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


@pytest.fixture
def mgr(tmp_path) -> OtaManager:
    m = OtaManager(FirmwareStore(tmp_path))
    m.base_url = "http://192.168.1.9:8080"
    return m


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


class TestOtaFlow:
    def test_command_carries_url_size_and_hash(self, mgr):
        info = mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        cmd = mgr.build_command(now=0.0)
        assert cmd["cmd"] == "ota" and cmd["size"] == info.size
        assert cmd["sha256"] == info.sha256
        assert cmd["url"] == f"http://192.168.1.9:8080/firmware/{mgr.grant.token}.bin"
        mgr.mark_sent(now=0.0)
        assert mgr.build_command() is None  # sent once

    def test_nothing_stored(self, mgr):
        with pytest.raises(LookupError):
            mgr.request_send()

    def test_no_second_send_while_one_runs(self, mgr):
        mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        with pytest.raises(RuntimeError):
            mgr.request_send(now=1.0)

    def test_success_is_the_boot_report_of_the_new_image(self, mgr):
        mgr.store.save(fake_image(version="2.0"))
        mgr.request_send(now=0.0)
        mgr.build_command(now=0.0)
        mgr.mark_sent(now=0.0)
        mgr.on_event({"event": "ota", "state": "downloading", "pct": 40}, now=5.0)
        assert (mgr.state, mgr.pct) == ("downloading", 40)
        mgr.on_event({"event": "ota", "state": "rebooting"}, now=30.0)
        mgr.on_boot("2.0", "Sep 26 2026 10:11:12", now=45.0)
        assert mgr.state == "done"
        assert mgr.robot_fw == {"version": "2.0", "built": "Sep 26 2026 10:11:12"}
        assert mgr.grant is None

    def test_old_image_after_reboot_is_a_rollback(self, mgr):
        mgr.store.save(fake_image(version="2.0"))
        mgr.request_send(now=0.0)
        mgr.mark_sent(now=0.0)
        mgr.on_event({"state": "rebooting"}, now=30.0)
        mgr.on_boot("1.0", "Jan  1 2026 00:00:00", now=100.0)
        assert mgr.state == "failed" and "rolled back" in mgr.error

    def test_robot_failure_is_reported(self, mgr):
        mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        mgr.mark_sent(now=0.0)
        mgr.on_event({"state": "failed", "error": "sha256 mismatch"}, now=3.0)
        assert (mgr.state, mgr.error) == ("failed", "sha256 mismatch")
        mgr.request_send(now=4.0)  # can retry after a failure
        assert mgr.state == "queued"

    def test_silence_times_out(self, mgr):
        mgr.store.save(fake_image())
        mgr.request_send(now=0.0)
        mgr.mark_sent(now=0.0)
        mgr.check_stall(now=STALL_TIMEOUT_S - 1)
        assert mgr.state == "sent"
        mgr.check_stall(now=STALL_TIMEOUT_S + 1)
        assert mgr.state == "failed"

    def test_boot_without_update_just_records_the_version(self, mgr):
        mgr.on_boot("1.4.1", "Sep 1 2026 00:00:00")
        assert mgr.state == "idle" and mgr.robot_fw["version"] == "1.4.1"


class TestSendGate:
    def test_sends_between_turns(self):
        assert ota_send_ready(requested=True, boot_seen=True, busy=False)

    def test_waits_for_a_conversation_to_end(self):
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


async def test_bad_upload_is_refused(console, mgr):
    r = await console.post("/api/firmware", headers=AUTH, **_upload(b"\0" * 5000))
    assert r.status_code == 400
    assert mgr.store.info() is None


async def test_send_without_upload(console):
    assert (await console.post("/api/firmware/send", headers=AUTH)).status_code == 404
