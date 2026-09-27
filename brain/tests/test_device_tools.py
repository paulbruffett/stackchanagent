"""The robot's own device tools: set_volume writes the SPEAKER_VOLUME knob and
sends set_volume at once (recording it so the idle sync won't repeat it),
get_device_status answers from the cached report, dance sends the command."""

from __future__ import annotations

import json

from config import get_config
from policy import DeviceStatus, volume_sync_action
from tools import ToolContext, dispatch, is_error_result


class FakeWs:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _ctx(mem, device: DeviceStatus | None = None) -> ToolContext:
    return ToolContext(ws=FakeWs(), memory=mem, device=device)


async def test_set_volume_down_from_reported_value(mem):
    dev = DeviceStatus(volume=50)
    ctx = _ctx(mem, dev)
    out = await dispatch("set_volume", {"change": "down"}, ctx)
    assert out == "Volume set to 35 (was 50)."
    assert ctx.ws.sent == [{"cmd": "set_volume", "value": 35}]
    assert get_config().get("SPEAKER_VOLUME") == 35
    assert dev.volume_sent == 35
    # The idle ticker's sync sees it as already sent — no second command.
    assert volume_sync_action(35, dev.volume, dev.volume_sent, busy=False) is None


async def test_set_volume_falls_back_to_the_knob_and_clamps(mem):
    get_config().set("SPEAKER_VOLUME", 90)
    ctx = _ctx(mem)  # no device report
    assert await dispatch("set_volume", {"change": "up"}, ctx) == "Volume set to 100 (was 90)."
    assert await dispatch("set_volume", {"level": 20}, ctx) == "Volume set to 20 (was 100)."


async def test_set_volume_up_at_max_is_a_failure_round(mem):
    ctx = _ctx(mem, DeviceStatus(volume=100))
    out = await dispatch("set_volume", {"change": "up"}, ctx)
    assert is_error_result(out) and "maximum" in out
    assert ctx.ws.sent == []


async def test_set_volume_without_args_is_an_error(mem):
    out = await dispatch("set_volume", {}, _ctx(mem, DeviceStatus(volume=50)))
    assert is_error_result(out)


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
    assert ctx.ws.sent == [{"cmd": "dance", "style": "robot"}]
    assert is_error_result(await dispatch("dance", {"style": "tango"}, ctx))
