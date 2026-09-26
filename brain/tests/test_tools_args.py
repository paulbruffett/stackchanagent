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
