"""Head double tap → Home Assistant light toggle, against a fake MCP client."""

from __future__ import annotations

import json

from light_toggle import light_state, toggle_light

CONTEXT = json.dumps({
    "success": True,
    "result": (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: Desk Lamp\n"
        "  domain: light\n"
        "  state: 'on'\n"
        "- names: Office Light, Ceiling\n"
        "  domain: light\n"
        "  state: 'off'\n"
        "  areas: Office\n"
    ),
})


class FakeMcp:
    def __init__(self, tools: list[str], context: str = CONTEXT) -> None:
        self.tools = tools
        self.context = context
        self.calls: list[tuple[str, dict]] = []

    def find_tool(self, raw: str) -> str | None:
        return f"mcp__ha__{raw}" if raw in self.tools else None

    async def dispatch(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return self.context if name.endswith("GetLiveContext") else "ok"


def test_light_state_reads_named_entity():
    assert light_state(CONTEXT, "Office Light") == "off"
    assert light_state(CONTEXT, "office light") == "off"
    assert light_state(CONTEXT, "Ceiling") == "off"
    assert light_state(CONTEXT, "Desk Lamp") == "on"
    assert light_state(CONTEXT, "Kitchen") is None


async def test_uses_toggle_tool_when_present():
    mcp = FakeMcp(["HassToggle", "HassTurnOn", "HassTurnOff", "GetLiveContext"])
    await toggle_light(mcp, "Office Light")
    assert mcp.calls == [("mcp__ha__HassToggle", {"name": "Office Light"})]


async def test_turns_on_when_off():
    mcp = FakeMcp(["HassTurnOn", "HassTurnOff", "GetLiveContext"])
    await toggle_light(mcp, "Office Light")
    assert mcp.calls[-1] == ("mcp__ha__HassTurnOn", {"name": "Office Light"})


async def test_turns_off_when_on():
    mcp = FakeMcp(["HassTurnOn", "HassTurnOff", "GetLiveContext"])
    await toggle_light(mcp, "Desk Lamp")
    assert mcp.calls[-1] == ("mcp__ha__HassTurnOff", {"name": "Desk Lamp"})


async def test_unknown_light_does_nothing():
    mcp = FakeMcp(["HassTurnOn", "HassTurnOff", "GetLiveContext"])
    result = await toggle_light(mcp, "Kitchen")
    assert "not found" in result
    assert [c[0] for c in mcp.calls] == ["mcp__ha__GetLiveContext"]


async def test_no_ha_tools():
    mcp = FakeMcp([])
    assert "no Home Assistant" in await toggle_light(mcp, "Office Light")
    assert mcp.calls == []
