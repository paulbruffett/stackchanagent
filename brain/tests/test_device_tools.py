"""The robot's own device tools: set_volume writes the SPEAKER_VOLUME knob and
sends set_volume at once (the sent value becomes the device's volume, so the
idle sync has nothing to repeat), get_device_status answers from the cached
report, dance sends the command."""

from __future__ import annotations

from conftest import FakeWs
from config import get_config
from policy import DeviceStatus, volume_sync_action
from tools import ToolContext, dispatch, is_error_result


def _ctx(mem, device: DeviceStatus | None = None) -> ToolContext:
    return ToolContext(ws=FakeWs(), memory=mem, device=device)


async def test_set_volume_down_from_reported_value(mem):
    dev = DeviceStatus(volume=50)
    ctx = _ctx(mem, dev)
    out = await dispatch("set_volume", {"change": "down"}, ctx)
    assert out == "Volume set to 35 (was 50)."
    assert ctx.ws.cmds("set_volume") == [{"cmd": "set_volume", "value": 35}]
    assert get_config().get("SPEAKER_VOLUME") == 35
    assert dev.volume == 35
    # The idle ticker's sync sees nothing to send.
    assert volume_sync_action(35, dev.volume, busy=False) is None


async def test_same_turn_calls_see_the_sent_volume(mem):
    dev = DeviceStatus()
    dev.update({"battery": 80, "charging": False, "volume": 50}, now=1.0)
    ctx = _ctx(mem, dev)
    await dispatch("set_volume", {"change": "down"}, ctx)
    assert await dispatch("set_volume", {"change": "down"}, ctx) == "Volume set to 20 (was 35)."
    assert "Volume at 20 out of 100." in await dispatch("get_device_status", {}, ctx)


async def test_set_volume_clamps_and_takes_a_level(mem):
    ctx = _ctx(mem, DeviceStatus(volume=90))
    assert await dispatch("set_volume", {"change": "up"}, ctx) == "Volume set to 100 (was 90)."
    assert await dispatch("set_volume", {"level": 20}, ctx) == "Volume set to 20 (was 100)."


async def test_set_volume_rejects_bool_level(mem):
    ctx = _ctx(mem, DeviceStatus(volume=50))
    assert is_error_result(await dispatch("set_volume", {"level": True}, ctx))
    assert ctx.ws.sent == []


async def test_set_volume_up_at_max_is_a_failure_round(mem):
    ctx = _ctx(mem, DeviceStatus(volume=100))
    out = await dispatch("set_volume", {"change": "up"}, ctx)
    assert is_error_result(out) and "maximum" in out
    assert ctx.ws.sent == []


async def test_set_volume_without_args_is_an_error(mem):
    out = await dispatch("set_volume", {}, _ctx(mem, DeviceStatus(volume=50)))
    assert is_error_result(out)


async def test_set_volume_on_firmware_without_volume_support(mem):
    for dev in (None, DeviceStatus()):
        ctx = _ctx(mem, dev)
        out = await dispatch("set_volume", {"change": "up"}, ctx)
        assert is_error_result(out) and "doesn't support volume" in out
        assert ctx.ws.sent == []
    assert not get_config().is_set("SPEAKER_VOLUME")


async def test_get_device_status_unknown_and_known(mem):
    assert await dispatch("get_device_status", {}, _ctx(mem)) == \
        "No status from the robot's body yet."
    dev = DeviceStatus()
    dev.update({"battery": None, "charging": None, "volume": 70}, now=1.0)
    assert await dispatch("get_device_status", {}, _ctx(mem, dev)) == \
        "Battery level unknown. Volume at 70 out of 100."


async def test_dance_sends_command_and_rejects_unknown_style(mem):
    ctx = _ctx(mem)
    assert await dispatch("dance", {"style": "robot"}, ctx) == "Dancing (robot)."
    assert ctx.ws.cmds("dance") == [{"cmd": "dance", "style": "robot"}]
    assert is_error_result(await dispatch("dance", {"style": "tango"}, ctx))
