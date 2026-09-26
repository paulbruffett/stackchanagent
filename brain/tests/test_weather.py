"""Spoken multi-day forecast from Open-Meteo's daily arrays."""

from __future__ import annotations

from mcp_servers.weather import _daily_forecast

DAILY = {
    "time": ["2026-09-26", "2026-09-27", "2026-09-28"],
    "weather_code": [0, 61, 3],
    "temperature_2m_max": [70.2, 64.4, 66.0],
    "temperature_2m_min": [51.0, 55.6, 52.2],
    "precipitation_probability_max": [0, 70, 10],
}


def test_forecast_names_tomorrow_then_weekdays_and_skips_today():
    out = _daily_forecast(DAILY, 3)
    assert out.startswith("Tomorrow: ")
    assert "56 to 64, 70% chance of rain." in out
    assert "Monday: " in out  # 2026-09-28
    assert "10%" not in out  # low rain chances aren't worth saying


def test_single_day_has_no_forecast_sentences():
    assert _daily_forecast(DAILY, 1) == ""
