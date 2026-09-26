"""Weather MCP server (Phase 9b.2) — stdio FastMCP over Open-Meteo.

One tool, `get_weather(location?)`. Geocodes a place name with Open-Meteo's
free geocoding API, then fetches the current conditions + today's range
from the forecast API. Both endpoints are keyless, so there is no secret
to manage. When no location is given it falls back to $DEFAULT_LOCATION
(set by the brain from the config knob), else "Seattle, Washington".

Run standalone: `python mcp_servers/weather.py` (speaks MCP over stdio).
The brain registers it as a stdio server in the MCP tab.
"""

from __future__ import annotations

import datetime
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 12.0

# WMO weather-code → short phrase (the codes Open-Meteo returns).
WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def _default_location() -> str:
    return os.environ.get("DEFAULT_LOCATION") or "Seattle, Washington"


MAX_DAYS = 7


@mcp.tool()
async def get_weather(location: str | None = None, days: int = 1) -> str:
    """Get the weather for a location (city/place name). If no location is
    given, uses the robot's configured default location.

    days=1 (default): current conditions plus today's high/low.
    days=2..7: also a day-by-day forecast — use days=2 for "tomorrow",
    up to 7 for "this week". Temperatures in Fahrenheit. Returns a short
    spoken-friendly summary."""
    place = (location or _default_location()).strip()
    days = max(1, min(MAX_DAYS, int(days or 1)))
    async with httpx.AsyncClient(timeout=TIMEOUT) as http:
        # Open-Meteo's geocoder matches a single place name, not
        # "City, State". Try the full string, then fall back to the
        # part before the first comma (e.g. "Seattle, Washington" → "Seattle").
        candidates = [place]
        if "," in place:
            candidates.append(place.split(",", 1)[0].strip())
        results: list = []
        for cand in candidates:
            geo = await http.get(
                GEOCODE_URL, params={"name": cand, "count": 1, "language": "en"}
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if results:
                break
        if not results:
            return f"I couldn't find a place called {place}."
        g = results[0]
        lat, lon = g["latitude"], g["longitude"]
        label = ", ".join(
            x for x in (g.get("name"), g.get("admin1"), g.get("country")) if x
        )

        fc = await http.get(FORECAST_URL, params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
            "forecast_days": days,
        })
        fc.raise_for_status()
        data = fc.json()

    cur = data.get("current", {})
    daily = data.get("daily", {})
    cond = WMO.get(cur.get("weather_code"), "unknown conditions")
    temp = cur.get("temperature_2m")
    wind = cur.get("wind_speed_10m")
    hi = (daily.get("temperature_2m_max") or [None])[0]
    lo = (daily.get("temperature_2m_min") or [None])[0]

    parts = [f"In {label} it's currently {cond}"]
    if temp is not None:
        parts[0] += f" and {round(temp)}°F"
    if hi is not None and lo is not None:
        parts.append(f"with a high of {round(hi)} and a low of {round(lo)}")
    if wind is not None and wind >= 12:
        parts.append(f"winds around {round(wind)} mph")
    out = ", ".join(parts) + "."
    if days > 1:
        out += " " + _daily_forecast(daily, days)
    return out


def _daily_forecast(daily: dict, days: int) -> str:
    """One short sentence per day after today: "Tomorrow: light rain, 58 to
    64, 70% chance of rain." """
    dates = daily.get("time") or []
    codes = daily.get("weather_code") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    rain = daily.get("precipitation_probability_max") or []
    out = []
    for i in range(1, min(days, len(dates))):
        name = "Tomorrow" if i == 1 else datetime.date.fromisoformat(dates[i]).strftime("%A")
        line = f"{name}: {WMO.get(codes[i] if i < len(codes) else None, 'unknown conditions')}"
        if i < len(highs) and i < len(lows) and highs[i] is not None and lows[i] is not None:
            line += f", {round(lows[i])} to {round(highs[i])}"
        if i < len(rain) and rain[i] is not None and rain[i] >= 20:
            line += f", {round(rain[i])}% chance of rain"
        out.append(line + ".")
    return " ".join(out)


if __name__ == "__main__":
    mcp.run()
