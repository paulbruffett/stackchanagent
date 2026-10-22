"""Head double tap → toggle a Home Assistant light, with no LLM step.

Uses the tools Home Assistant's MCP server already exposes to the agent.
HA's Assist API may not offer a toggle tool, so when HassToggle is missing
we read the light's state from GetLiveContext and call HassTurnOn or
HassTurnOff ourselves.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("brain.light_toggle")


def light_state(context: str, name: str) -> str | None:
    """'on', 'off', or None if `name` isn't in GetLiveContext's output.

    The output is a JSON envelope ({"success": ..., "result": "..."}) whose
    result is a header line plus a YAML list of entities:
        - names: Office Light
          domain: light
          state: 'on'
    Parsed by hand to avoid a YAML dependency."""
    try:
        data = json.loads(context)
        if isinstance(data, dict) and isinstance(data.get("result"), str):
            context = data["result"]
    except json.JSONDecodeError:
        pass
    want = name.strip().lower()
    # One chunk per list entry ("- names: ...").
    for entry in re.split(r"^- ", context, flags=re.MULTILINE)[1:]:
        names = re.search(r"^names:\s*(.+)$", entry, flags=re.MULTILINE)
        if not names:
            continue
        aliases = [n.strip().strip("'\"").lower() for n in names.group(1).split(",")]
        if want not in aliases:
            continue
        state = re.search(r"^\s*state:\s*['\"]?(\w+)", entry, flags=re.MULTILINE)
        return state.group(1).lower() if state else None
    return None


async def toggle_light(mcp: Any, name: str) -> str:
    """Toggle the light called `name`. Returns a short log-friendly result."""
    toggle = mcp.find_tool("HassToggle")
    if toggle:
        return await mcp.dispatch(toggle, {"name": name})

    ctx_tool = mcp.find_tool("GetLiveContext")
    on_tool = mcp.find_tool("HassTurnOn")
    off_tool = mcp.find_tool("HassTurnOff")
    if not (ctx_tool and on_tool and off_tool):
        return "no Home Assistant light tools connected"

    state = light_state(await mcp.dispatch(ctx_tool, {}), name)
    if state is None:
        return f"'{name}' not found in Home Assistant's live context"
    tool = off_tool if state == "on" else on_tool
    return await mcp.dispatch(tool, {"name": name})
