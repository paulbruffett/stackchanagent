"""Blank optional arguments the model fills in are dropped before an MCP call:
Home Assistant rejects the whole intent over floor="" or temperature=0."""

from __future__ import annotations

from tools import clean_mcp_args


def test_blank_values_are_dropped():
    args = {"name": "Office", "area": "Office", "floor": "", "domain": ["light"],
            "device_class": [], "extra": None, "opts": {}}
    assert clean_mcp_args("mcp__homeassistant__HassTurnOn", args) == {
        "name": "Office", "area": "Office", "domain": ["light"]}


def test_ha_zero_colour_temperature_dropped_but_brightness_zero_kept():
    args = {"name": "office light", "temperature": 0, "brightness": 0}
    assert clean_mcp_args("mcp__homeassistant__HassLightSet", args) == {
        "name": "office light", "brightness": 0}


def test_non_ha_tools_keep_zero_values():
    assert clean_mcp_args("mcp__weather__get_weather", {"days": 0, "location": ""}) == {"days": 0}


def test_ha_name_miss_is_retried_as_area():
    from tools import ha_name_fallback

    miss = "[tool error] Error calling tool: <MatchFailedError ... MatchFailedReason.NAME: 1>"
    assert ha_name_fallback("mcp__homeassistant__HassTurnOff",
                            {"name": "Office", "domain": ["light"]}, miss) == {
        "domain": ["light"], "area": "Office"}
    # A name beside an area: drop the name, keep the area.
    assert ha_name_fallback("mcp__homeassistant__HassTurnOff",
                            {"name": "lamp", "area": "Office"}, miss) == {"area": "Office"}


def test_ha_name_fallback_only_on_name_mismatch():
    from tools import ha_name_fallback

    other = "[tool error] Error calling tool: Failed to call turn_on for: ['light.office']"
    assert ha_name_fallback("mcp__homeassistant__HassTurnOff", {"name": "Office"}, other) is None
    assert ha_name_fallback("mcp__homeassistant__HassTurnOff", {"name": "Office"}, "ok") is None
    miss = "[tool error] MatchFailedReason.NAME"
    assert ha_name_fallback("mcp__weather__get_weather", {"name": "x"}, miss) is None
    assert ha_name_fallback("mcp__homeassistant__HassTurnOff", {"area": "Office"}, miss) is None
