"""Philips Hue MCP server (Phase 9b.3) — stdio FastMCP over the CLIP v2 API.

Talks directly to the Hue Bridge's local HTTPS API (no cloud, no extra
dependency beyond httpx). Two env vars, both set once in .env:

  HUE_BRIDGE_IP  — the bridge's LAN IP (passes through the child env)
  HUE_TOKEN      — the application key from pairing (injected via the
                   server's env_ref so other brain secrets aren't exposed)

Pairing is a one-time manual step — see mcp_servers/README.md.

The bridge serves a self-signed cert, so TLS verification is disabled
(`verify=False`); this is a LAN-local device on a trusted network.

Tools: list_lights, set_light, set_room.
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hue")

TIMEOUT = 8.0


def _bridge() -> tuple[str, str] | None:
    ip = os.environ.get("HUE_BRIDGE_IP", "").strip()
    token = os.environ.get("HUE_TOKEN", "").strip()
    if not ip or not token:
        return None
    return ip, token


def _client(ip: str, token: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=f"https://{ip}/clip/v2",
        headers={"hue-application-key": token},
        verify=False,
        timeout=TIMEOUT,
    )


def _need_config() -> str:
    return (
        "The Hue bridge isn't configured. Set HUE_BRIDGE_IP and HUE_TOKEN "
        "in the brain's .env (see mcp_servers/README.md for pairing)."
    )


def _match(name: str, items: list[dict]) -> dict | None:
    """Find a resource whose metadata.name matches `name` (exact first,
    then case-insensitive substring)."""
    name_l = name.strip().lower()
    exact = [i for i in items if i.get("metadata", {}).get("name", "").lower() == name_l]
    if exact:
        return exact[0]
    subs = [i for i in items if name_l in i.get("metadata", {}).get("name", "").lower()]
    return subs[0] if subs else None


def _payload(on: bool | None, brightness_pct: float | None) -> dict:
    body: dict = {}
    if on is not None:
        body["on"] = {"on": bool(on)}
    if brightness_pct is not None:
        body["dimming"] = {"brightness": max(0.0, min(100.0, float(brightness_pct)))}
    return body


@mcp.tool()
async def list_lights() -> str:
    """List the Hue lights and rooms, with each light's on/off state and
    brightness. Use this to discover what names are available before
    calling set_light or set_room."""
    cfg = _bridge()
    if cfg is None:
        return _need_config()
    ip, token = cfg
    async with _client(ip, token) as http:
        lr = await http.get("/resource/light")
        lr.raise_for_status()
        lights = lr.json().get("data", [])
        rr = await http.get("/resource/room")
        rr.raise_for_status()
        rooms = [r.get("metadata", {}).get("name", "?") for r in rr.json().get("data", [])]

    if not lights:
        return "No Hue lights found on the bridge."
    lines = []
    for li in lights:
        nm = li.get("metadata", {}).get("name", "?")
        on = li.get("on", {}).get("on")
        br = li.get("dimming", {}).get("brightness")
        state = "on" if on else "off"
        if on and br is not None:
            state += f" at {round(br)}%"
        lines.append(f"{nm}: {state}")
    out = "Lights — " + "; ".join(lines)
    if rooms:
        out += ". Rooms — " + ", ".join(rooms) + "."
    return out


@mcp.tool()
async def set_light(
    name: str, on: bool | None = None, brightness_pct: float | None = None
) -> str:
    """Turn a single light on or off and/or set its brightness (0-100%).
    `name` matches a light by name (case-insensitive). Provide `on`,
    `brightness_pct`, or both. Setting brightness implies turning on."""
    cfg = _bridge()
    if cfg is None:
        return _need_config()
    ip, token = cfg
    if brightness_pct is not None and on is None:
        on = True
    async with _client(ip, token) as http:
        r = await http.get("/resource/light")
        r.raise_for_status()
        light = _match(name, r.json().get("data", []))
        if light is None:
            return f"No light named '{name}'. Try list_lights."
        body = _payload(on, brightness_pct)
        if not body:
            return "Nothing to change — specify on and/or brightness_pct."
        put = await http.put(f"/resource/light/{light['id']}", json=body)
        put.raise_for_status()
    nm = light.get("metadata", {}).get("name", name)
    return f"OK, updated {nm}."


@mcp.tool()
async def set_room(
    name: str, on: bool | None = None, brightness_pct: float | None = None
) -> str:
    """Turn all lights in a room on or off and/or set their brightness
    (0-100%). `name` matches a room by name (case-insensitive). Setting
    brightness implies turning on."""
    cfg = _bridge()
    if cfg is None:
        return _need_config()
    ip, token = cfg
    if brightness_pct is not None and on is None:
        on = True
    async with _client(ip, token) as http:
        r = await http.get("/resource/room")
        r.raise_for_status()
        room = _match(name, r.json().get("data", []))
        if room is None:
            return f"No room named '{name}'. Try list_lights."
        grouped = next(
            (s["rid"] for s in room.get("services", [])
             if s.get("rtype") == "grouped_light"),
            None,
        )
        if grouped is None:
            return f"Room '{name}' has no controllable light group."
        body = _payload(on, brightness_pct)
        if not body:
            return "Nothing to change — specify on and/or brightness_pct."
        put = await http.put(f"/resource/grouped_light/{grouped}", json=body)
        put.raise_for_status()
    nm = room.get("metadata", {}).get("name", name)
    return f"OK, updated the {nm}."


if __name__ == "__main__":
    mcp.run()
