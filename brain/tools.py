"""Tool definitions for the Claude tool-use loop.

Each tool maps to a JSON command the firmware understands. Handlers
take a WebSocket plus the tool input and emit the command. Most return
a brief acknowledgement string for the assistant turn.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from websockets.asyncio.server import ServerConnection

log = logging.getLogger("brain.tools")


# Schemas exposed to Claude. Keep tight — descriptions are what drive
# tool selection, so be explicit about when to call each.
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "set_expression",
        "description": (
            "Change the robot's facial expression. Use sparingly to react to the "
            "conversation: 'happy' on good news, 'sad' on bad news, 'surprised' "
            "on unexpected information, 'sleepy' when asked to wind down. Defaults "
            "to 'neutral'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "enum": [
                        "neutral",
                        "happy",
                        "sad",
                        "sleepy",
                        "angry",
                        "surprised",
                    ],
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "look_at",
        "description": (
            "Point the head at a target. Yaw is left/right in degrees "
            "(-128 to +128, negative is left). Pitch is up/down in degrees "
            "(3 to 87, larger looks down). Use for natural conversational gestures."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "yaw_deg": {"type": "number", "minimum": -128, "maximum": 128},
                "pitch_deg": {"type": "number", "minimum": 3, "maximum": 87},
            },
            "required": ["yaw_deg", "pitch_deg"],
        },
    },
    {
        "name": "set_motion_rate",
        "description": (
            "Adjust how often the robot makes idle look-around movements. "
            "0 = perfectly still, 10 = constantly fidgeting. Default is 4. "
            "Use when the user asks the robot to 'move less' or 'be still'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "per_minute": {"type": "integer", "minimum": 0, "maximum": 30}
            },
            "required": ["per_minute"],
        },
    },
    {
        "name": "end_conversation",
        "description": (
            "End the conversation gracefully. Use when the user says goodbye or "
            "indicates they're done talking."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def dispatch(
    name: str, input_: dict[str, Any], ws: ServerConnection
) -> str:
    """Send a tool's JSON command to the firmware. Returns the
    tool_result content string for the next agent turn."""
    if name == "set_expression":
        await ws.send(
            json.dumps({"cmd": "set_expression", "value": input_["expression"]})
        )
        return f"Expression set to {input_['expression']}."
    if name == "look_at":
        await ws.send(
            json.dumps(
                {
                    "cmd": "look_at",
                    "yaw_deg": input_["yaw_deg"],
                    "pitch_deg": input_["pitch_deg"],
                }
            )
        )
        return f"Looking at yaw={input_['yaw_deg']}, pitch={input_['pitch_deg']}."
    if name == "set_motion_rate":
        await ws.send(
            json.dumps(
                {"cmd": "set_motion_rate", "per_minute": input_["per_minute"]}
            )
        )
        return f"Motion rate set to {input_['per_minute']} per minute."
    if name == "end_conversation":
        # No firmware-side cmd needed; the agent's reply is the goodbye.
        return "Conversation ended."
    log.warning("unknown tool: %s", name)
    return f"Unknown tool {name}"
